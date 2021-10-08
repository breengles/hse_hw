#!/usr/bin/env python

from math import atan2

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


class Chaser:
    x = 0.0
    y = 0.0
    theta = 0.0
    msg = Twist()

    def __call__(self):
        self.pub = rospy.Publisher("/chaser/cmd_vel", Twist, queue_size=10)

        # is there any better way to update pose?
        # probably via velocity; upd in __move_to
        self.__init_subscriber("/chaser/pose", self.__update_position)

        self.__init_subscriber("/turtle1/pose", self.__move_to)  # turtle1 == victim

    def __move_to(self, msg: Pose):
        self.msg.linear.x = msg.x - self.x
        self.msg.linear.y = msg.y - self.y
        self.msg.angular.z = atan2(self.msg.linear.y, self.msg.linear.x) - self.theta

        self.pub.publish(self.msg)

    def __update_position(self, msg: Pose):
        self.x = msg.x
        self.y = msg.y
        self.theta = msg.theta

    @staticmethod
    def __init_subscriber(name, callback):
        rospy.Subscriber(name, Pose, callback)


if __name__ == "__main__":
    rospy.init_node("chasing")

    chaser = Chaser()
    chaser()

    rospy.spin()
