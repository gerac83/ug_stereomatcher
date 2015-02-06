# ROS message meta-data
# This includes:
#	seq: sequence ID
#	stamp: timestamp
#	frame_id: id of the image frame
# NOTE: "header" should hold the same values as the image which is processed
Header header

# Image
sensor_msgs/Image image_stack
# Image size number of colums
int16 im_width
# Image size number of rows
int16 im_height

# Foveated region of interest number of colums
int16 roi_width
# Foveated region of interest number of rows
int16 roi_height
# Number of stack levels
int16 num_levels
