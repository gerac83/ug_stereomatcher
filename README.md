# ug_stereomatcher

**Authors (see package.xml for contact info):** Paul Cockshott, Tian Xu, Gerardo Aragon-Camarasa, Susanne Oehler, Mozhgan Chimeh and J. Paul Siebert

**SEE LICENSE BEFORE USING THIS SOFTWARE**

**A tutorial can be found at https://github.com/gerac83/ug_stereomatcher/wiki**

The ug_stereomatcher ROS package comprises the University of Glasgow GPU stereo matcher and nodes to compute point clouds from disparity maps for each operational mode as described below.

The matcher has two operational modes:

1. Compute disparity maps over a image pyramid at full resolution in 10 seconds on 16MP RGB images
2. Compute foveated disparity maps for each level on the pyramid in 3 seconds on 16MP RGB images with a fixed fovea size of 615 by 407

**NOTE:** _You need a CUDA capable graphics card in order to compile and run the software. For 16MP images, you need at least 2GB of graphics card RAM. This code has been optimised for the NVIDIA Geforce GTX 750i and 970._

## Publications

If you use this ROS package please cite the following paper(s):

* Cockshott, P., Oehler, S., Xu, T., Siebert, J. P., and Aragon-Camarasa, G., 2012 “Parallel stereo vision algorithm” in: Many-Core Applications Research Community Symposium 2012, 29-30 Nov 2012, Aachen, Germany.

This ROS package has been used and demonstrated in the following papers:

* Sun, L., Aragon-Camarasa, G. and Siebert, J. P., 2014. “Improving Cloth Manipulation by Modelling Wrinkles” in: 2014 IEEE International Conference on Robotics and Automation (ICRA 2014), Advances in robot manipulation of clothes and flexible objects Workshop. June 1, 2014. Hong Kong, China.
* Sun, L., Aragon-Camarasa, G., Siebert, J.P. and Rogers, S., 2013 “A Heuristic-Based Approach for Flattening Wrinkled Clothes,” in ‘Towards Autonomous Robotic Systems, TAROS 2013’, University of Oxford, 20-30 August 2013.
* Sun, L., Aragon-Camarasa, G. Rogers, S., Siebert, J.P., 2015 "Accurate Garment Surface Analysis using an Active Stereo Robot Head with Application to Dual-Arm Flattening," in ICRA 2015 (Accepted)

## Database of stereo images

For benchmarking, we have produced a databse of 80 stereo RGB images; it can be downloaded for free at:

https://sites.google.com/site/ugstereodatabase/

Companion technical report:

http://arxiv.org/abs/1311.7295
