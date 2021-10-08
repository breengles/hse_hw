#!/usr/bin/env python

from math import atan2, acos, cos, sin, sqrt, pi

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import numpy as np


class Chaser:
    pos = np.zeros(2)
    n = np.array([1, 0])
    msg = Twist()

    eps = 1e-8

    def __call__(self):
        self.pub = rospy.Publisher("/chaser/cmd_vel", Twist, queue_size=10)

        # is there any better way to update pose?
        # probably via velocity; upd in __move_to
        self.__init_subscriber("/chaser/pose", self.__update_position)

        self.__init_subscriber("/turtle1/pose", self.__move_to)  # turtle1 == victim

    def __move_to(self, msg: Pose):
        v_pos = np.array([msg.x, msg.y])  # vector of the target coords

        d = v_pos - self.pos  # difference vector
        d_norm = np.linalg.norm(d)
        d /= d_norm  # normalize it

        self.msg.linear.x = d_norm / 2  # naive approach to stop when the target is approached

        angle = np.arccos(d.dot(self.n))  # get absolute angle

        # The sign of the dot product of cross-vector and plane norm ([0, 0, 1])
        # detemines (counter) clockwise rotation
        self.msg.angular.z = angle * np.sign(np.cross(self.n, d))

        self.pub.publish(self.msg)

    def __update_position(self, msg: Pose):
        self.pos[0] = msg.x
        self.pos[1] = msg.y
        self.n = np.array([np.cos(msg.theta), np.sin(msg.theta)])

    @staticmethod
    def __init_subscriber(name, callback):
        rospy.Subscriber(name, Pose, callback)


if __name__ == "__main__":
    rospy.init_node("chasing")

    chaser = Chaser()
    chaser()

    rospy.spin()
