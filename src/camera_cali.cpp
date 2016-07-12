#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
    for(;;)
    {
        Mat frame;
        Mat img;
        cap>>frame;
        copyMakeBorder (  frame , img,  abs(frame.cols-frame.rows)/2 , abs(frame.cols-frame.rows)/2, 0 , 0 ,BORDER_CONSTANT , Scalar(0,0,0)  ) ;
        imshow("fuji", img);
        if(waitKey(30) >= 0) break;
    }
    //! [file_read]

}
