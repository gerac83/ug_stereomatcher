//
//  settings.h (for opencv chessboard, legacy class)
//  Camera Calibration
//
//  Created by Gerardo Aragon on 21/11/2012.
//  Copyright (c) 2012 Gerardo Aragon. All rights reserved.

#ifndef SETTINGS_H
#define SETTINGS_H

class Settings
{
public:
    Settings() : goodInput(false) {}

    void readList()
    {
        goodInput = true;
        atImageList = 0;
        string input2 = input;
        ROS_INFO_STREAM("Image list: " << input2);
        //ROS_INFO("");
        if(!readStringList(input2, imageList))
        {
            ROS_ERROR("Image list is not a readable file");
            goodInput = false;
        }
    }

    Mat nextImage()
    {
        ROS_INFO_STREAM("Size of image list: " << (int)imageList.size());
        int nextToLastInList = (int)imageList.size() - 2;
        Mat result;
        if( atImageList >= (int)imageList.size() )
        {
            atImageList = nextToLastInList;
        }

        ROS_INFO_STREAM("Reading: " << imageList[atImageList]);
        result = imread(imageList[atImageList++], 1);
        imageSize = result.size();

        return result;
    }

    static bool readStringList( const string& filename, vector<string>& l )
    {
        l.clear();
        FileStorage fs(filename, FileStorage::READ);
        if( !fs.isOpened())
        {
            ROS_ERROR_STREAM("Couldn't open the file: " << filename);
            return false;
        }
        FileNode n = fs.getFirstTopLevelNode();
        if( n.type() != FileNode::SEQ )
            return false;
        FileNodeIterator it = n.begin(), it_end = n.end();
        for( ; it != it_end; ++it )
        {
            //cout << (string)*it;
            l.push_back((string)*it);
        }
        return true;
    }

public:
    bool goodInput;
    std::string node_path;

    string input;               	// The input (only for debug purposes)
    vector<string> imageList;
    int atImageList;

    Size imageSize;

};

#endif
