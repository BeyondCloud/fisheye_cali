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
#define CLIP_WIDTH 886
#define BEND_RANGE 12
#define LOWEST_TONE 48

int threshold_value = 120;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
struct note_t
{
    int tone = -1;
    int bend = -1;
};
template<typename T, size_t N>
void record_valid_rows(Mat &Img,T (&valid_y)[N])
{
    int key_cnt = 0;
    int pixel_cnt = 0;
    bool bw_switch = 0; //  black/white  true/false
    for (int  y = 0; y <Img.rows ; y++)
    {
        for (int  x = 0; x <Img.cols ; x++)
        {
            if((int)Img.at<uchar>(y,x)==0)
            {
                if(bw_switch == true)
                {
                    pixel_cnt = 0;
                    bw_switch = false;
                }
            }
            else
            {
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
        if(key_cnt == KEY)
        {
            valid_y[y] = true;
        }
        key_cnt = 0;
        pixel_cnt = 0;
    }
}
int main(int argc, const char** argv)
{
    Mat Img;
    Img = imread("line.jpg");
    cvtColor(Img ,Img, COLOR_RGB2GRAY);
    threshold( Img,Img, threshold_value, max_BINARY_value,threshold_type );
    imshow("thres",Img);

    bool valid_y[CLIP_WIDTH] = {false};
    record_valid_rows(Img,valid_y);
    note_t **note = new note_t*[Img.rows];
    for (int i = 0 ; i < Img.rows ; i++)
        note[i] = new note_t[Img.cols];
//    cvtColor(Img,Img,COLOR_GRAY2BGR);

    const float bpk = 64.0/BEND_RANGE;
    for (int  y = 0; y <Img.rows ; y++)
    {
        if(valid_y[y])
        {
            int current_tone = LOWEST_TONE;
            bool bw_switch = (bool)Img.at<uchar>(y,0);
            int cnt=0;
            float current_bend = 0;
            for (int  x = 1; x <Img.cols ; x++)
            {

                bool current_pixel = (bool)Img.at<uchar>(y,x);
                cnt++;
                if(current_pixel != bw_switch || x == Img.cols-1)
                {
                    bw_switch = current_pixel;
                    double key_width = (double)cnt;
                    while(cnt != -1 )
                   {
                       note[y][x-cnt].tone = current_tone;
                       //48~80
                       note[y][x-cnt].bend =(int)(current_bend+bpk*(1.0-(double)cnt/key_width));
                       cnt--;
                   }
                   current_bend += bpk;
                   current_tone ++;
                }
            }
        }
    }
    for (int  x = 1; x <Img.cols ; x++)
    {
        cout<<note[12][x].bend<<"\t";
    }
    imshow("asdf",Img);
    waitKey();
    return 0;
}
