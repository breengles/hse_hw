#!/usr/bin/env python


from typing import List

from numpy import ma
import rospy
import numpy as np
from rospy.topics import Publisher
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point


class Robot:
    def __init__(self, window=1, threshold=0.1, res=0.1, r=5) -> None:
        rospy.init_node("filter")

        self.window = window
        self.threshold = threshold

        self.res = res
        self.r = r
        self.grid_size = 2 * int(r / res)

        self.marker = self.__init_mrk_msg()
        self.grid = self.__init_grid_msg()

        rospy.Subscriber("/base_scan", LaserScan, self.callback)
        self.pub_map = Publisher("/map", OccupancyGrid, queue_size=10)
        self.pub_mrk = Publisher("/visualization_marker", Marker, queue_size=10)

    def __init_mrk_msg(self):
        marker = Marker()

        marker.header.frame_id = "base_laser_link"

        marker.type = marker.POINTS

        marker.color.r = 0.5
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 0.5

        marker.scale.x = self.res
        marker.scale.y = self.res
        marker.scale.z = self.res

        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0

        return marker

    def __init_grid_msg(self):
        grid = OccupancyGrid()
        grid.info = MapMetaData()

        grid.header.frame_id = "base_laser_link"
        grid.info.resolution = self.res

        grid.info.width = self.grid_size
        grid.info.height = self.grid_size

        grid.info.origin.position.x = -self.r
        grid.info.origin.position.y = -self.r
        grid.info.origin.position.z = 0

        return grid

    def _filter(self, ranges: List):
        flt = np.ones_like(ranges, dtype=bool)

        for point_id, point in enumerate(ranges[self.window : -self.window], start=self.window):
            prvs = ranges[point_id - self.window : point_id]
            nxts = ranges[point_id + 1 : point_id + self.window]

            targets = np.hstack([prvs, nxts])

            r = abs(point - targets.mean()) / np.abs(targets).mean()

            flt[point_id] = r < self.threshold

        return flt

    def mrk_msg(self, xs, ys):
        self.marker.points = [Point(x, y, 0) for x, y in zip(xs, ys)]
        self.pub_mrk.publish(self.marker)

    def grid_msg(self, xs, ys):
        data = np.zeros((self.grid_size, self.grid_size), dtype=int)

        for x, y in zip(xs, ys):
            if x ** 2 + y ** 2 < self.r ** 2:
                i = int((x + self.r) / self.res)
                j = int((y + self.r) / self.res)

                data[j, i] = 100

        self.grid.data = data.ravel()
        self.pub_map.publish(self.grid)

    def callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)[self._filter(msg.ranges)]
        phis = np.linspace(msg.angle_min, msg.angle_max, ranges.shape[0], endpoint=True)

        xs = ranges * np.cos(phis)
        ys = ranges * np.sin(phis)

        # self.mrk_msg(xs, ys)
        self.grid_msg(xs, ys)


if __name__ == "__main__":
    robot = Robot(threshold=0.1)
    rospy.spin()
