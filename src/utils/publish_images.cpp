//
//  get_images.cpp
//  Helper function for the rh_calibration node
//
//  Created by Gerardo Aragon on November, 2012.
//  Copyright (c) 2012 Gerardo Aragon. All rights reserved.

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#include <ug_stereomatcher/settings.h>
#include <ug_stereomatcher/CamerasSync.h>
#include <sensor_msgs/CameraInfo.h>

bool showImage = true;

static const char WINDOW_LEFT[] = "Left image";
static const char WINDOW_RIGHT[] = "Right image";

static const char CAM_NAMEL[] = "left_camera";
static const char CAM_NAMER[] = "right_camera";

// Messages
static const char CAML_SUB[] = "input_left_image";
static const char CAMR_SUB[] = "input_right_image";
static const char CAM_ACQUIRE[] = "acquire_images";
static const char OUT_CAM_INFOL[] = "camera_info_left";
static const char OUT_CAM_INFOR[] = "camera_info_right";

// Paramter Server
static const char INPUT_XML[] = "image_list";
static const char CAMERA_INFOL[] = "camera_info_url_left";
static const char CAMERA_INFOR[] = "camera_info_url_right";

namespace enc = sensor_msgs::image_encodings;

class RHcam_node
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    ros::Subscriber sub_;
    image_transport::Publisher imageL_pub_;
    image_transport::Publisher imageR_pub_;
    ros::Publisher pub_, infoL_pub_, infoR_pub_;

	std::string urlL, urlR;
	Mat left, right;

public:

	Settings s_;
    sensor_msgs::CameraInfo infoL, infoR;

    int noFrames;

    RHcam_node() : it_(nh_)
    {

		s_.node_path = ros::package::getPath("ug_stereomatcher");
		nh_.getParam(CAMERA_INFOL, urlL);
		nh_.getParam(CAMERA_INFOR, urlR);

		ROS_INFO_STREAM(urlL);
		ROS_INFO_STREAM(urlR);

        if(nh_.hasParam(INPUT_XML))
        {
            nh_.getParam(INPUT_XML, s_.input);
        }else{
            ROS_ERROR_STREAM("Could not open image list: " << INPUT_XML);
        }
        s_.readList();
        ROS_INFO_STREAM("Input file read!");

		if (!s_.goodInput)
		{
		    ROS_ERROR("Invalid input detected. Application stopping.");
            nh_.shutdown();
		    return;
		}

		if(showImage)
		{
			cv::namedWindow(WINDOW_LEFT, CV_WINDOW_NORMAL);
			cv::resizeWindow(WINDOW_LEFT, 640, 480);
			cvMoveWindow(WINDOW_LEFT, 10, 10);

			cv::namedWindow(WINDOW_RIGHT, CV_WINDOW_NORMAL);
			cv::resizeWindow(WINDOW_RIGHT, 640, 480);
			cvMoveWindow(WINDOW_RIGHT, 650, 10);

			cv::startWindowThread();
			ROS_INFO("Windows initialised");
		}

        infoL = loadCameraInfo(urlL);
        infoR = loadCameraInfo(urlR);

        noFrames = 1;

        // Setup advertise and subscribe ROS messages
        //pub_ = nh_.advertise<sensor_msgs::JointState>(PTU_OUT, 1);
        imageL_pub_ = it_.advertise(CAML_SUB, 1);
        imageR_pub_ = it_.advertise(CAMR_SUB, 1);
        infoL_pub_ = nh_.advertise<sensor_msgs::CameraInfo>(OUT_CAM_INFOL,1);
        infoR_pub_ = nh_.advertise<sensor_msgs::CameraInfo>(OUT_CAM_INFOR,1);

        sub_ = nh_.subscribe(CAM_ACQUIRE, 1, &RHcam_node::captureImage, this);

        ROS_INFO("Node Initialised");

    }

    ~RHcam_node()
    {
        ROS_INFO("Exit node... ");
    }

    void captureImage(const ug_stereomatcher::CamerasSync::ConstPtr& msg)
    {
        string str1 ("preview");
        string str2 ("full");

		ROS_INFO("Reading Images...");
        try
        {
            if (str1.compare(msg->data.c_str())) // full
            {

            }
            else if (str2.compare(msg->data.c_str())) // preview
            {
                ROS_FATAL("This capture mode is not supported");
            }
            else
            {
                ROS_ERROR("Capture mode not known: %s", msg->data.c_str());
                return;
            }

			// For each call, read images from the list
	        left = s_.nextImage(); // Get left image
	        right = s_.nextImage(); // Get right image

            cv_bridge::CvImage cvi;
            ros::Time time = ros::Time::now();

            // convert OpenCV image to ROS message (Left image)
            cvi.header.stamp = time;
            cvi.header.frame_id = CAM_NAMEL;
            cvi.header.seq = noFrames;
            cvi.encoding = "rgb8";
            cvi.image = left;

            infoL.header.stamp = time;
            infoL.width = left.cols;
            infoL.height = left.rows;
            infoL.header.frame_id = CAM_NAMEL;

            ROS_INFO_STREAM("Left Rows: " << left.rows << " Cols: " << left.cols);

            imageL_pub_.publish(cvi.toImageMsg());
            infoL_pub_.publish(infoL);

            //imageL_pub_.publish(cvi.toImageMsg());

            // convert OpenCV image to ROS message (Right image)
            cvi.header.stamp = time;
            cvi.header.frame_id = CAM_NAMER;
            cvi.header.seq = noFrames;
            cvi.encoding = "rgb8";
            cvi.image = right;

            infoR.header.stamp = time;
            infoR.width = right.cols;
            infoR.height = right.rows;
            infoR.header.frame_id = CAM_NAMER;

            ROS_INFO_STREAM("Right Rows: " << right.rows << " Cols: " << right.cols);

            imageR_pub_.publish(cvi.toImageMsg());
            infoR_pub_.publish(infoR);

            // Display Image
            if(showImage)
            {
                imshow(WINDOW_LEFT, left);
                imshow(WINDOW_RIGHT, right);
                waitKey(3);
            }

            noFrames++;

			ROS_INFO("Messages published");

        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("RHcam_node exception: %s", e.what());
            return;
        }
    }

    void printMatrix(Mat M, bool printType)
    {
        if(printType)
            ROS_INFO_STREAM("Matrix type: " << M.type());
        // dont print empty matrices
        if (M.empty()){
            ROS_INFO("---");
            return;
        }
        // loop through columns and rows of the matrix
        for(int i=0; i < M.rows; i++){
            for(int j=0; j < M.cols ; j++){
                if(M.type() == 6)
                    cout << M.at<double>(i,j) << "\t";
                else
                    cout << M.at<double>(i,j) << "\t";
            }
            cout<<endl;
        }
        cout<<endl;
    }

    sensor_msgs::CameraInfo loadCameraInfo(std::string url)
    {
        sensor_msgs::CameraInfo outInfo;

        try
		{
            ROS_INFO_STREAM("Loading URL: " << url);
			FileStorage fs(url.c_str(), FileStorage::READ);
			if(!fs.isOpened())
		    {
		        ROS_WARN_STREAM("An exception occurred. Using default values for /camera_info");
                fs.release();
                outInfo.K.empty();
                outInfo.D.empty();
                outInfo.P.empty();
                //outInfo.F.push_back(0);
		    }
		    else
		    {

                Mat K, D, P;//, F
                fs["K"] >> K;
                fs["D"] >> D;
                //fs["F"] >> F;
				fs["P"] >> P;

				for(int i = 0; i < D.cols; i++)
				{
				    outInfo.D.push_back(D.at<double>(i));
				    if(i < P.rows)
				    {
				        for(int j = 0 ; j < P.cols; j++)
				        {
                            if(j < K.cols)
				            {
                                outInfo.K[3*i+j] = K.at<double>(i,j);
                                //outInfo.F.push_back(F.at<double>(i,j));
				            }
                            outInfo.P[4*i+j] = P.at<double>(i,j);
				        }
				    }
				}

                ROS_INFO("K:");
                printMatrix(K, true);
                ROS_INFO("P:");
                printMatrix(P, true);
		    }

            ROS_WARN_STREAM_COND((int)outInfo.K.size() == 0, "Size cam_info K: " << outInfo.K.size());
            //ROS_WARN_STREAM_COND((int)outInfo.F.size() == 0, "Size cam_info F: " << outInfo.F.size());

		}
		catch (...)
		{
			ROS_ERROR("Failed to open file %s\n", url.c_str());
	        ros::shutdown();
		}

		return outInfo;

    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "RHcamera_simulation");
    RHcam_node c_;
    ros::spin();
    return 0;
}
