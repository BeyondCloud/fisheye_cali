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
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    for(;;)
    {
        Mat frame;
        cap>>frame;
        imshow("fuji", frame);
        if(waitKey(30) >= 0) break;
    }
    //! [file_read]

}
