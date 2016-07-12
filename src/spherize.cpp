// define head function
#ifndef PS_ALGORITHM_H_INCLUDED
#define PS_ALGORITHM_H_INCLUDED
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include "math.h"

using namespace std;
using namespace cv;

void Show_Image(Mat&, const string &);

#endif // PS_ALGORITHM_H_INCLUDED


#include  <time.h>

using namespace std;
using namespace cv;

#define pi 3.1415926
#define CLIP_ORIGIN_X 0
#define CLIP_ORIGIN_Y 0
#define CLIP_WIDTH 1080
#define CLIP_HEIGHT 1080

void MyFilledCircle( Mat img, Point center,double w )
{
 int thickness = -1;
 int lineType = 8;

 circle( img,
         center,
         w/2,
         Scalar( 0, 255, 0 ),
         thickness,
         lineType );
}
struct rmpData_t
{
    int rows;
    int cols;
    Mat map_x;
    Mat map_y;
};
void rmpRead(rmpData_t &r)
{
    FileStorage fs("test.xml", cv::FileStorage::READ);
    fs["cols"] >> r.cols;
    fs["rows"] >> r.rows;
    fs["map_x"] >> r.map_x;
    fs["map_y"] >> r.map_y;
    fs.release();
}
void rmpWrite(rmpData_t &r)
{
        FileStorage fs("test.xml", cv::FileStorage::WRITE);
        cout<<"data below write to test.xml:\n";
        fs << "cols" << r.cols;
        cout<<"cols = "<<r.cols<<"\n";
        fs << "rows" << r.rows;
        cout<<"rows = "<<r.rows<<"\n";
        fs << "map_x" << r.map_x;
        cout<<"map x value\n";
        fs << "map_y" << r.map_y;
        cout<<"map y value"<<"\n";
        fs.release();
}
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();

    int nRows = I.rows;
    int nCols = I.cols * channels;

    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return I;
}
int main()
{
    VideoCapture cap(0);
    Mat Img,frame;
    cap>>frame;
    frame = Mat(frame, Rect(CLIP_ORIGIN_X,CLIP_ORIGIN_Y,CLIP_WIDTH,CLIP_HEIGHT));
    cvtColor(frame, frame, CV_BGR2GRAY);
    Img = frame;

//   Img = imread("image.png",CV_LOAD_IMAGE_COLOR);
//    cvtColor(Img, Img, CV_BGR2GRAY);
    if(!Img.data)
    {
        cout<<"fail to load image";
        return 0;
    }

    Mat Img_out = Img.clone();
    Mat test = imread("image.png",CV_LOAD_IMAGE_COLOR);
    Img_out = Scalar(0,0,255);
    double  w = Img.cols;
    double  h = Img.rows;
    Mat map_x =Mat(w, h, CV_32FC1),
        map_y =Mat(w, h, CV_32FC1);

    MyFilledCircle( Img_out, Point( w/2.0, w/2.0),w );

    for (int  y = 0 ; y < h ; y++)
    {
        // normalize y coordinate to -1 ... 1
        double ny = ((2*y)/h)-1;
        // pre calculate ny*ny
        double ny2 = ny*ny;
        for (int  x = 0 ; x < w ; x++)
        {
            // normalize x coordinate to -1 ... 1
            double nx = ((2*x)/w)-1;
            // pre calculate nx*nx
            double nx2 = nx*nx;
            // calculate distance from center (0,0)
            // this will include circle or ellipse shape portion
            // of the image, depending on image dimensions
            // you can experiment with images with different dimensions
            double r = sqrt(nx2+ny2);
            double theta = atan2(ny,nx);
            // discard pixels outside from circle!
            if (0.0<=r&&r<=1.0)
            {

                double nr;
                //normal spherize
                //nr = (r + (1.0-sqrt(1.0-r*r))) /2.0;
                //inverse spherize
                nr = (2.0*r-1.0+sqrt((-4.0)*r*r + 4.0*r+1))/2.0;
                // discard radius greater than 1.0
                if (nr<=1.0)
                {
                    // calculate the angle for polar coordinates
                    // calculate new x position with new distance in same angle
                    double nxn = nr*cos(theta);
                    // calculate new y position with new distanc,e in same angle
                    double nyn = nr*sin(theta);
                    // map from -1 ... 1 to image coordinates
                    int x2 = (int)(((nxn+1.0)*w)/2.0);
                    // map from -1 ... 1 to image coordinates
                    int y2 = (int)(((nyn+1.0)*h)/2.0);
                    // if(x2<w && y2 < h)
                    Img_out.at<Vec3b>(y,x) = Img.at<Vec3b>(y2,x2);
                    map_x.at<int>(y,x) = x2;
                    map_y.at<int>(y,x) = y2;
                }
            }
         }
    }

//    FileStorage fs("test.xml", cv::FileStorage::WRITE);
//    fs << "cols" << w;
//    fs << "rows" << h;
//    fs << "map_x" << map_x;
//    fs << "map_y" << map_y;
//    fs.release();


    Mat src  = imread("image.png",CV_LOAD_IMAGE_COLOR);
    Mat dst = src.clone();
    rmpData_t rmp_data;
    rmp_data.cols = w;
    rmp_data.rows = h;
    rmp_data.map_x = map_x;
    rmp_data.map_y = map_y;
//    rmpWrite(rmp_data);
//   rmpRead(rmp_data);
//    cout<<rmp_data.cols;
     for(;;)
    {
            cap>>frame;
            frame = Mat(frame, Rect(CLIP_ORIGIN_X,CLIP_ORIGIN_Y,CLIP_WIDTH,CLIP_HEIGHT));
            cvtColor(frame, frame, CV_BGR2GRAY);
            Img = frame;
        double  t = ( double )getTickCount();
    //    ===========on the fly method=========================================
    //    0.018sec 1080 x1080
         for (int  y = 0 ; y < Img.rows ; y++)
        {
                for (int  x = 0 ; x < Img.cols ; x++)
                {
                    Img_out.at<uchar>(y,x)=Img.at<uchar>(
                                                           map_y.at<int>(y,x),
                                                           map_x.at<int>(y,x)
                                                    );
                }
        }
            imshow("out",Img_out);
        t = (( double )getTickCount() - t)/getTickFrequency();
        cout <<  "Times passed in seconds: "  << t << endl;

    }
//    ===========remap method(float): 0.05sec=================================================
//    remap( src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
//    =========LUT===========================================
    imshow("t",dst);

//    imshow("rig",Img);
//    imshow("out",Img_out);
    waitKey(0);

}
