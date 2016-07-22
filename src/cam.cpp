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
#define MIN_WIDTH 4
#define KEY 25
int threshold_value = 58;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
int main(int argc, const char** argv)
{
    Mat Img;
    Img = imread("line.jpg");
    cvtColor(Img ,Img, COLOR_RGB2GRAY);
    threshold( Img,Img, threshold_value, max_BINARY_value,threshold_type );


    int key_cnt = 0;
    int pixel_cnt = 0;
    bool bw_switch = 0; //  black/white  true/false
    bool recorded_y[Img.rows] = {false};
    int bcnt  =0;
    int wcnt  =0;

    for (int  y = 0; y <Img.rows ; y++)
    {
        for (int  x = 0; x <Img.cols ; x++)
        {
            if((int)Img.at<uchar>(y,x)==0)
            {
                wcnt++;
                if(bw_switch == true)
                {
                    pixel_cnt = 0;
                    bw_switch = false;
                }
            }
            else
            {
                bcnt++;
                if(bw_switch == false)
                {
                    pixel_cnt = 0;
                    bw_switch = true;
                }
            }
            pixel_cnt++;
            if(pixel_cnt==4)
            {
                key_cnt++;
            }

        }
         //    cout<<key_cnt<<" ";
        if(key_cnt == KEY)
        {
            recorded_y[y] = true;
        }
        key_cnt = 0;
        pixel_cnt = 0;
    }

//cout<<"b"<<bcnt<<"w"<<wcnt;
    for (int  y = 0; y <Img.rows ; y++)
    {
        if( recorded_y[y])
            cout<<y<<"\t";
    }
    imshow("asdf",Img);
    waitKey();
    return 0;
}
