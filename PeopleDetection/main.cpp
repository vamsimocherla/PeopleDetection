//
//  main.cpp
//  PeopleDetection
//
//  Created by Vamsi Mocherla on 5/2/14.
//  Copyright (c) 2014 VamsiMocherla. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

int minHessian = 400;
SurfFeatureDetector detector(minHessian);
SurfDescriptorExtractor extractor;

bool detectLogo(Mat person, Mat desObject, Mat object, vector<KeyPoint> kpObject, vector<Point2f> objCorners)
{
    // scale up the image
    resize(person, person, Size(), 4, 4, CV_INTER_CUBIC);
    
    // sharpen the image
    Mat image;
    GaussianBlur(person, image, cv::Size(0, 0), 3);
    addWeighted(person, 1.75, image, -0.75, 0, image);

    GaussianBlur(person, image, cv::Size(0, 0), 3);
    addWeighted(person, 1.75, image, -0.75, 0, image);

    // detect key points in the input frame
    vector<KeyPoint> kpFrame;
    detector.detect(person, kpFrame);
    
    // extract feature descriptors for the detected key points
    Mat desFrame;
    extractor.compute(person, kpFrame, desFrame);
    if(desFrame.empty() or desObject.empty())
        return false;
    
    // match the key points with object
    FlannBasedMatcher matcher;
    vector< vector <DMatch> > matches;
    matcher.knnMatch(desObject, desFrame, matches, 2);
    
    // compute the good matches among the matched key points
    vector<DMatch> goodMatches;
    for(int i=0; i<desObject.rows; i++)
    {
        if(matches[i][0].distance < 0.6 * matches[i][1].distance)
        {
            goodMatches.push_back(matches[i][0]);
        }
    }
    
    // draw the good matches
//    Mat imageMatches;
//    drawMatches(object, kpObject, person, kpFrame, goodMatches, imageMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(),  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if(goodMatches.size() >= 8)
    {
        vector<Point2f> obj;
        vector<Point2f> scene;
        
        for( int i = 0; i < goodMatches.size(); i++ )
        {
            // get the keypoints from the good matches
            obj.push_back( kpObject[ goodMatches[i].queryIdx ].pt );
            scene.push_back( kpFrame[ goodMatches[i].trainIdx ].pt );
        }
        
        Mat H;
        H = findHomography(obj, scene);
        
        vector<Point2f> sceneCorners(4);
        perspectiveTransform( objCorners, sceneCorners, H);
        
        // draw lines between the corners (the mapped object in the scene image )
/*        line(imageMatches, sceneCorners[0]+Point2f(object.cols, 0), sceneCorners[1]+Point2f(object.cols, 0), Scalar(255, 255, 255), 4);
        line(imageMatches, sceneCorners[1]+Point2f(object.cols, 0), sceneCorners[2]+Point2f(object.cols, 0), Scalar(255, 255, 255), 4);
        line(imageMatches, sceneCorners[2]+Point2f(object.cols, 0), sceneCorners[3]+Point2f(object.cols, 0), Scalar(255, 255, 255), 4);
        line(imageMatches, sceneCorners[3]+Point2f(object.cols, 0), sceneCorners[0]+Point2f(object.cols, 0), Scalar(255, 255, 255), 4);*/
        
        line(person, sceneCorners[0], sceneCorners[1], Scalar(255, 255, 255), 4);
        line(person, sceneCorners[1], sceneCorners[2], Scalar(255, 255, 255), 4);
        line(person, sceneCorners[2], sceneCorners[3], Scalar(255, 255, 255), 4);
        line(person, sceneCorners[3], sceneCorners[0], Scalar(255, 255, 255), 4);
        
        imshow("Person", person);
        cout << "[MESSAGE] LOGO DETECTED" << endl;
        return true;
    }
    return false;
}
int main()
{
    // read the logo image
    Mat object = imread("radio2.png");
    if(!object.data)
    {
        cout << "[ERROR] CANNOT READ IMAGE" << endl;
        return false;
    }
    
    // detect key points in the image using SURF
    vector<KeyPoint> kpObject;
    
    detector.detect(object, kpObject);
    
    // compute feature descriptors
    Mat desObject;
    
    extractor.compute( object, kpObject, desObject );
    
    // get the corners of the object
    vector<Point2f> objCorners(4);
    
    objCorners[0] = cvPoint(0, 0);
    objCorners[1] = cvPoint(object.cols, 0);
    objCorners[2] = cvPoint(object.cols, object.rows);
    objCorners[3] = cvPoint(0, object.rows);
    
    // capture video from webcam
    VideoCapture input(0);
    
    if(!input.isOpened())
	{
		cout << "[ERROR] CANNOT OPEN WEBCAM" << endl;
		return -1;
	}
    
	cout << "[MESSAGE]: CAPTURING VIDEO FROM WEBCAM" << endl;
    
    int key = 0;
    Mat inputFrame;
    Mat pathImage(360, 640, CV_8UC3, Scalar(0,0,0));
    Point newPoint;
    Point oldPoint;
    
    while(key != 27)
    {
		// capture each frame of the video
		input >> inputFrame;
		if(inputFrame.empty())
			break;
        
        // resize the image for faster processing
        resize(inputFrame, inputFrame, Size(), 0.5, 0.5, INTER_LINEAR);

        // run the HOG detector with default parameters
        HOGDescriptor hog;
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        
        vector<Rect> found, found_filtered;
        hog.detectMultiScale(inputFrame, found, 0, Size(8,8), Size(32,32), 1.05, 2);
        
        size_t i, j;
        for (int i=0; i<found.size(); i++)
        {
            Rect r = found[i];
            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }
        
        // draw a bounding box for the detected people
        for (i=0; i<found_filtered.size(); i++)
        {
            Rect r = found_filtered[i];
            
            // the HOG detector returns slightly larger rectangles than the real objects
            // so we slightly shrink the rectangles to get a nicer output
            r.x += cvRound(r.width*0.2);
            r.width = cvRound(r.width*0.55);
            r.y += cvRound(r.height*0.06);
            r.height = cvRound(r.height*0.75);
            
            if(r.x>=0 and r.y>=0 and r.x+r.width<=inputFrame.cols and r.y+r.height<=inputFrame.rows)
            {
                Mat tmp(inputFrame, r);
                // capture the detected person and check for presence of logo
                if(r.height > 0.5*inputFrame.rows)
                {
                    // detect the logo
                    if(detectLogo(tmp, desObject, object, kpObject, objCorners))
                    {
                        // draw a bounding box for the detected people
                        rectangle(inputFrame, r.tl(), r.br(), Scalar(0, 255, 0), 2);
                        
                        // draw the path of the movement of the person
//                        rectangle(pathImage, r.tl(), r.br(), Scalar(0, 255, 0), 2);
//                        circle(pathImage, Point(r.x+r.width/2, r.y+r.height/2), 1, Scalar(0,255,0));
                        
                        // get the new position
                        newPoint.x = r.x+r.width/2;
                        newPoint.y = r.y+r.width/2;
                        line(pathImage, newPoint, oldPoint, Scalar(0,255,0), 2);
                        
                        // save the old position
                        oldPoint.x = newPoint.x;
                        oldPoint.y = newPoint.y;
                    }
                }
            }
        }
        
        // display the detected image
        imshow("People Detection", inputFrame);
        
        // display the path of the person detected
        imshow("Path Traced", pathImage);
        key = waitKey(1);
    }
    
    // display the path of the person detected
//    imshow("Path Traced", pathImage);
//    waitKey(0);

    return 0;
}