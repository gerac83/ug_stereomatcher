#!/usr/bin/env python

"""
Temporal node to avoid the memory leak on the GPU matcher.

This opens, close and opens again an instance of the ROS node of
the GPU matcher. As soon as the memory leak bug is corrected, This
node will not longer be used.

Authors: Gerardo Aragon-Camarasa. 2014
"""

import roslib; roslib.load_manifest('ug_stereomatcher')
import rospy
import message_filters
from stereo_msgs.msg import DisparityImage
import signal
import subprocess
import sys
import time
from rospy import Time
import numpy as np
import whereIam
from ug_stereomatcher.msg import foveatedstack

def messagesCBF(msL,msR):
    global proc

    rospy.loginfo("Received foveated horizontal and vetical disparities")

    # Close matcher after 5 seconds
    time.sleep(5)
    proc.send_signal(signal.SIGINT)
    rospy.loginfo("GPU matcher closed...")
    # Start again macther after 5 seconds of being closed
    #time.sleep(5)
    proc = subprocess.Popen(['rosrun', 'ug_stereomatcher', 'UG_matcher_gpu'])
    rospy.loginfo("GPU matcher openned...")

def messagesCB(msL,msR):
    global proc

    rospy.loginfo("Received horizontal and vetical disparities")

    # Close matcher after 5 seconds
    time.sleep(5)
    proc.send_signal(signal.SIGINT)
    rospy.loginfo("GPU matcher closed...")
    # Start again macther after 5 seconds of being closed
    #time.sleep(5)
    proc = subprocess.Popen(['rosrun', 'ug_stereomatcher', 'UG_matcher_gpu'])
    rospy.loginfo("GPU matcher openned...")

# ********* MAIN *********
if __name__ == '__main__':
    global proc

    rospy.init_node('RHmatcher_hack', anonymous=True)
    ### subscribers
    subDV = message_filters.Subscriber('output_disparityV', DisparityImage)
    subDH = message_filters.Subscriber('output_disparityH', DisparityImage)
    ts = message_filters.TimeSynchronizer([subDH, subDV], 1)
    ts.registerCallback(messagesCB)
    
    subDVf = message_filters.Subscriber('output_stackV', foveatedstack)
    subDHf = message_filters.Subscriber('output_stackH', foveatedstack)
    tsf = message_filters.TimeSynchronizer([subDHf, subDVf], 1)
    tsf.registerCallback(messagesCBF)

    proc = subprocess.Popen(['rosrun', 'ug_stereomatcher', 'UG_matcher_gpu'])
    
    rospy.loginfo("matcher.py node ready!")
    rospy.spin()






