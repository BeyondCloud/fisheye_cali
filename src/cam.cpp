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

int main(int argc, const char** argv)
{
    Mat orig,target;
    Mat *fptr;
    orig = imread("grid2.png");
    cvtColor(orig,orig,CV_RGB2GRAY);
    target = orig.clone();

    imshow("before", orig);
    cout<<"addr of orig before" <<&orig<<"\n";
    cout<<"addr of orig before" <<*orig.ptr<uchar>(0,0)<<"\n";
    uchar *out;
    uchar *in;
    for(int i = 0;i < target.rows;i++)
    {
        in = orig.ptr<uchar>(i);
        out = target.ptr<uchar>(i);
        for(int j = 0;j < target.cols;j++)
        {
            out[j] = in[target.cols -j];
        }
    }
    imshow("target", target);
    Mat orig2 = imread("sample.png");
    waitKey();
    if(!orig2.data)
    {
        cout<<"sample.png open failed";
        return 0;
    }
        for(int i = 0;i < target.rows;i++)
    {

        for(int j = 0;j < target.cols;j++)
        {
            orig.at<uchar>(i,j) = 0;
        }
    }
    cout<<"addr of orig after" <<&orig<<"\n";
    imshow("target after update", target);
    imshow("orig after update", orig);

    waitKey();
    return 0;
}
