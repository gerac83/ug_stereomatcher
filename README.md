# rh_stereo_proc
The rh_stereo_proc ROS package consists of the University of Glasgow GPU stereo matcher and nodes to compute point clouds from disparity maps for each operational mode as described below.

The matches has two operational modes:

1. it can compute disparity maps over a image pyramid at full resolution in 10 seconds on 16MP RGB images
2. it can compute foveated disparity maps for each level on the pyramid in 3 seconds on 16MP RGB images with a fixed fovea size of 615 by 407

###TODO:
* Add articles
* Add database as a reference benchmark
* Add dependencies
* Add test images
