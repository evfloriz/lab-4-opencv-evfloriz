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
# Feature detection:
#   https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#basics-of-brute-force-matcher
#
# CompressedImage processing:
#   http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
#
# Duckiebot setup:
#   https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
#   https://github.com/duckietown/dt-ros-commons/tree/daffy/packages/duckietown_msgs/msg
#
# Apriltags
#   https://github.com/duckietown/lib-dt-apriltags

import os
import queue
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String

import numpy as np

from duckietown_msgs.msg import Twist2DStamped

from sensor_msgs.msg import CompressedImage

import cv2

from dt_apriltags import Detector


class MoveHandler():
    # Class structure adapted from OdometryReader:
    # https://github.com/UAlberta-CMPUT412/lab-2-turtlebot-kinematics-evfloriz

    def __init__(self):
        self.camera = CameraReader()
    
        self.max_angular_speed = 0.1
        self.angle_divisor = 40
        self.linear_speed = -0.5
        self.stopped = False

        self.topic = '/csc22911/car_cmd_switch_node/cmd'
        self.register()
    
    def register(self):
        # set up publisher to communicate with robot
        self.publisher = rospy.Publisher(self.topic, Twist2DStamped, queue_size=10)

    def unregister(self):
        self.publisher.unregister()
    
    def publish(self, move_cmd):
        self.publisher.publish(move_cmd)

    def stop(self):
        #stop robot
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
        # and clamp so it doesn't rotate too fast
        angle_adjust = -float(self.camera.dist_to_center) / self.angle_divisor
        #move_cmd.omega = np.clip(angle_adjust, -self.max_angular_speed, self.max_angular_speed)
        move_cmd.omega = angle_adjust

        self.publish(move_cmd)

class CameraReader():
    def __init__(self):
        
        self.dist_to_center = 0
        self.yellow = False
        self.intersection = False

        self.stop = False

        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

        self.detect_counter = 0
        
        self.topic = '/csc22911/camera_node/image/compressed'
        self.register()

    def add_debug_publishers(self):
        self.debug_raw_image = rospy.Publisher('/csc22911/camera_node/output/raw/compressed', CompressedImage)

    def register(self):
        self.subscriber = rospy.Subscriber(self.topic, CompressedImage, self.callback, queue_size=1)
        self.add_debug_publishers()
        rospy.sleep(0.1)

    def unregister(self):
        self.subscriber.unregister()

    def callback(self, msg):
        # convert to cv2
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # convert to hsv
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # counter for detection
        if (self.detect_counter == 0):
            self.detect_april_tags(cv_image)
            self.detect_counter = 10
        self.detect_counter -= 1

        # rgb to hsv values found using experimentation and this source:
        # https://www.rapidtables.com/convert/color/rgb-to-hsv.html
        path_low = np.array([20, 63, 63])
        path_high = np.array([40, 255, 255])
        path_mask = cv2.inRange(hsv, path_low, path_high)

        # source path processing code and information:
        # Programming Robots with ROS: A Practical Introduction to the Robot Operating System
        # link to section on Google Books is at the top of this file

        h, w, d = cv_image.shape

        # get a crop of the bottom of the robots vision
        path_mask_bottom = path_mask.copy()
        path_mask_bottom[0:(3 * h // 4), 0:w] = 0

        
        # get a narrow crop of the path mask to snap to path after intersection
        search_left = 1 * w // 4
        search_right = 3 * w // 4
        
        path_mask_crossing = path_mask.copy()
        path_mask_crossing[0:h, 0:search_left] = 0
        path_mask_crossing[0:h, search_right:w] = 0
        
        # crop the masks to just a narrow row in front of the turtlebot
        search_top = 2 * h // 4
        search_bottom = 3 * h // 4
        
        path_mask[0:search_top, 0:w] = 0
        path_mask[search_bottom:h, 0:w] = 0

        # go straight by default
        #self.dist_to_center = 0

        # no yellow in the bottom quarter - crop to higher centroied
        
        if (np.sum(path_mask_bottom) < 10000):
            # intersection mode
            
            # crop the edges of path_mask
            path_mask = path_mask_crossing

            #rospy.loginfo("intersection mode")
                    
        # rospy.loginfo("yellow amount: %i", int(np.sum(path_mask)))
        
        # by default track the yellow path
        if (self.check_mask(path_mask)):
            self.dist_to_center = self.find_dist_to_center(path_mask, w)

        # intersection ahead
        # if masked band is under a threshold of yellow, move forward only
        # crop sides until threshold is great enough
        # eg 
        # --------
        # | yyyy |
        # --------
        # --------
        # |      |
        # --------
        #   ----
        #   |  |
        #   ----
        #   ----
        #   |yy|
        #   ----
        # --------
        # | yyyy |
        # --------

        

        # image matching

        # publish raw image for debugging
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', path_mask)[1]).tostring()

        self.debug_raw_image.publish(msg)
        

    def find_dist_to_center(self, mask, w):
        # find the coordinate of the centroid from the first order moments in the x and y
        M = cv2.moments(mask)
        if (M['m00'] > 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # draw red circle at centroid coordinates
            #cv2.circle(mask, (cx, cy), 10, (0,0,255), -1)
            
            # set the distance from the x value of the centroid to the center of the camera
            return cx - w/2
        else:
            return 0        # this shouldn't ever return, its checked outside of this function first
    
    def check_mask(self, mask):
        M = cv2.moments(mask)
        return M['m00'] > 0

    def match_feature(self, img):
        # feature matching with brute force matcher
        # source: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#basics-of-brute-force-matcher
        stop_img = cv2.imread('/home/eflorizo/catkin_ws/src/lab-3-opencv-evfloriz/world/stop.png')

        # initiate detector
        orb = cv2.ORB_create()

        # find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img, None)
        kp2, des2 = orb.detectAndCompute(stop_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # experimentally 50+ indicated a match
        if (len(matches) > 50):
            return True
        else:
            return False

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