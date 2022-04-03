# HW on Self-driving cars


# How to create and compline project
* Init package and workspace
```
mkdir -r <workspace>/src
cd <workspace>/src
catkin_init_workspace
catkin_create_pkg <package_name> rospy
touch <package_name>/src/<name>.py
chmod +x <package_name>/src/<name>.py
```

* Compile, you do not need to reinit package after package modifications
```
cd ../
catkin_make
source devel/setup.bash
```

# run package
`rosrun <package_name> <name>.py`

do not forget to run `roscore`




# Tips
* Do not forget to run `catkin_make` in workspace folder
* To run `roslaunch <NAME>.launch` in the corresponding workspace `src`-folder
