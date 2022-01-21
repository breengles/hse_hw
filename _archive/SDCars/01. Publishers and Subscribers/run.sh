#!/usr/bin/env bash


catkin_make
source devel/setup.bash
cd src
roslaunch chasing.launch
