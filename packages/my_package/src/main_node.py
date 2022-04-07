#!/usr/bin/env python3

# Sources:
#
# Line following:
#   Programming Robots with ROS: A Practical Introduction to the Robot Operating System
#       link: https://books.google.ca/books?id=G3v5CgAAQBAJ&pg=PT293&lpg=PT293&dq=ros+opencv+moments&source=bl&ots=P2Hi66daEH&sig=ACfU3U35pUY5_vp-5qNnyFQM7nTjPr0lcg&hl=en&sa=X&ved=2ahUKEwievtDgz_H1AhX7DjQIHWbsA8cQ6AF6BAghEAM#v=onepage&q=ros%20opencv%20moments&f=false
#   http://edu.gaitech.hk/turtlebot/line-follower.html
#   https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
#
# RGB to HSV:
#   https://www.rapidtables.com/convert/color/rgb-to-hsv.html
#
# CompressedImage processing:
#   http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
#
# Duckiebot setup:
#   https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
#   https://github.com/duckietown/dt-ros-commons/tree/daffy/packages/duckietown_msgs/msg
#
# Apriltag detection
#   https://github.com/duckietown/lib-dt-apriltags

import rospy

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from dt_apriltags import Detector

from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class MoveHandler():
    def __init__(self):
        self.camera = CameraReader()
    
        self.angle_divisor = 40
        self.linear_speed = -0.5
        self.stopped = False

        self.topic = '/csc22911/car_cmd_switch_node/cmd'
        self.register()
    
    def register(self):
        self.publisher = rospy.Publisher(self.topic, Twist2DStamped, queue_size=10)

    def unregister(self):
        self.publisher.unregister()
        self.camera.unregister()
    
    def publish(self, move_cmd):
        self.publisher.publish(move_cmd)

    def stop(self):
        self.publish(Twist2DStamped())
        self.stopped = True
        rospy.sleep(1.0)

    # move forward accounting for the direction of the path
    def follow_path(self):
        # early out and stop if camera detects stop april tag
        if (self.camera.stop):
            self.stop()
            return

        # initialize movement command
        move_cmd = Twist2DStamped()
        move_cmd.v = self.linear_speed

        # set angular velocity to a proportion of the distance from the path to the center of vision
        angle_adjust = -float(self.camera.dist_to_center) / self.angle_divisor
        move_cmd.omega = angle_adjust

        self.publish(move_cmd)

class CameraReader():
    def __init__(self):
        self.dist_to_center = 0
        self.detect_counter = 0
        self.stop = False

        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        
        self.topic = '/csc22911/camera_node/image/compressed'
        self.debug_topic = '/csc22911/camera_node/output/raw/compressed'
        self.register()

    def register(self):
        self.subscriber = rospy.Subscriber(self.topic, CompressedImage, self.callback, queue_size=1)
        self.debug_image = rospy.Publisher(self.debug_topic, CompressedImage)
        rospy.sleep(0.1)

    def unregister(self):
        self.subscriber.unregister()
        rospy.sleep(0.1)
        self.debug_image.unregister()

    def callback(self, msg):
        # convert to cv2
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # convert to hsv
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # run detection every 5 ticks
        if (self.detect_counter == 0):
            self.detect_april_tags(cv_image)
            self.detect_counter = 5
        self.detect_counter -= 1

        # rgb to hsv values found using experimentation
        path_low = np.array([20, 63, 63])
        path_high = np.array([40, 255, 255])

        # create a mask of the yellow path
        path_mask = cv2.inRange(hsv, path_low, path_high)

        h, w, d = cv_image.shape

        # duplicate mask and crop to analyze the bottom of the robots vision
        path_mask_bottom = path_mask.copy()
        path_mask_bottom[0:(3 * h // 4), 0:w] = 0
        
        # duplicate mask and crop to analyze a narrow vertical slice of the robots vision,
        # in order to find the path after the intersection
        path_mask_crossing = path_mask.copy()
        path_mask_crossing[0:h, 0:(1 * w // 4)] = 0
        path_mask_crossing[0:h, (3 * w // 4):w] = 0
        
        # crop original mask to just a narrow row in front of the robot for line follow
        path_mask[0:(2 * h // 4), 0:w] = 0
        path_mask[(3 * h // 4):h, 0:w] = 0

        # if there is not enough yellow in the bottom mask, intersection detected
        # follow the centroid from the mask that captures the path after the intersection
        if (np.sum(path_mask_bottom) < 10000):
            path_mask = path_mask_crossing

        # update dist_to_center if yellow path is detected in the mask
        if (self.check_mask(path_mask)):
            self.dist_to_center = self.find_dist_to_center(path_mask, w)

        # publish mask for debugging
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', path_mask)[1]).tostring()

        self.debug_image.publish(msg)
        

    def find_dist_to_center(self, mask, w):
        # find the coordinate of the centroid from the first order moments in the x and y
        M = cv2.moments(mask)
        if (M['m00'] > 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            # set the distance from the x value of the centroid to the center of the camera
            return cx - w/2
        else:
            return 0        # this shouldn't ever return, its checked outside of this function first
    
    def check_mask(self, mask):
        M = cv2.moments(mask)
        return M['m00'] > 0

    def detect_april_tags(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(grey_img, estimate_tag_pose=False, camera_params=None, tag_size=None)

        if len(tags) and tags[0].tag_id == 166:
            self.stop = True
            #rospy.loginfo("detected")
        else:
            #rospy.loginfo("not detected")
            pass


class MainNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MainNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # create MoveHandler
        self.move_handler = MoveHandler()

    def run(self):
        rate = rospy.Rate(10)
        while not self.move_handler.stopped and not rospy.is_shutdown():
            self.move_handler.follow_path()
            rate.sleep()

    def on_shutdown(self):
        self.move_handler.stop()
        self.move_handler.unregister()
        rospy.loginfo("Shutting down")
        rospy.sleep(1.0)
        super(MainNode, self).on_shutdown()
        

if __name__ == '__main__':
    # create the node
    node = MainNode(node_name='main_node')
    # run node
    node.run()
    # keep spinning
    #rospy.spin()