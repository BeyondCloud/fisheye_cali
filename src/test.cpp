//Press u to update subtract image
//Press e to exit
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdio.h>
#include <conio.h>
//for sound ctrl
//#include <iomanip>
#include <windows.h>

using namespace std;
using namespace cv;
bool isRun = true;
inline void kbmgr()
{
        if (_kbhit() )
		{
			switch (_getch())
			{
				case 'e':
					isRun = false;
					break;
			}
		}
}
int main(int argc, const char** argv)
{


	VideoCapture cap(0); // open the camera 1
    Size patternsize(8,6); //number of centers
     vector<Point2f> centers; //this will be filled by the detected centers
    if (!cap.isOpened()) //return -1 when no camera found
		return -1;
    Mat frame;
    while(isRun){

        cap >> frame;
        bool patternfound = findChessboardCorners(frame,patternsize,centers);
        drawChessboardCorners(frame, patternsize, Mat(centers), patternfound);

        kbmgr();
		imshow("before", frame);
		waitKey(30);
	}

    return 0;
}
