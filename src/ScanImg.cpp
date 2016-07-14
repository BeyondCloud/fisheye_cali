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


#endif // PS_ALGORITHM_H_INCLUDED


#include  <time.h>


#define pi 3.1415926
#define CLIP_ORIGIN_X 0
#define CLIP_ORIGIN_Y 0
#define CLIP_WIDTH 1920
#define CLIP_HEIGHT 1920
struct myPoint
{
    int x;
    int y;
};
struct fisheye_t
{
    Point center;
    int r;//radius
};
void MyFilledCircle( Mat img, Point center,int r )
{

    int thickness = 1;
    int lineType = 8;
    circle( img,
         center,
         r,
         Scalar( 255, 255, 255 ),
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
void fisheye_boarder(Mat &src,Mat &Img,fisheye_t &feye)
{
    copyMakeBorder(src,Img,max(0,feye.r - feye.center.y)
                            ,max(0,feye.r - (Img.rows - feye.center.y) )
                            ,max(0,feye.r - feye.center.x)
                            ,max(0,feye.r - (Img.cols - feye.center.x))
                            ,BORDER_CONSTANT , Scalar(0,0,0));
    cout<<"boarder top,down,left,right";
    cout<<max(0,feye.r - feye.center.y)<<"\n";
    cout<<max(0,feye.r - (Img.rows - feye.center.y) )<<"\n";
    cout<<max(0,feye.r - feye.center.x)<<"\n";
    cout<<max(0,feye.r - (Img.cols - feye.center.x))<<"\n";
    feye.center.x += max(0,feye.r - feye.center.x);
    feye.center.y += max(0,feye.r - feye.center.y);
    MyFilledCircle(Img,feye.center,feye.r);
}
void fisheye_tbl_create(Mat &Img,Mat &map_x,Mat &map_y)
{
    double  w = Img.cols;
    double  h = Img.rows;
    for (int  y = 0; y <Img.rows ; y++)
    {
        // normalize y coordinate to -1 ... 1
        double ny = ((2*y)/h)-1;
        // pre calculate ny*ny
        double ny2 = ny*ny;
        for (int  x = 0 ; x <Img.cols ; x++)
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
                    // calculate new y position with new distance in same angle
                    double nyn = nr*sin(theta);
                    // map from -1 ... 1 to image coordinates
                    int x2 = (int)(((nxn+1.0)*w)/2.0);
                    // map from -1 ... 1 to image coordinates
                    int y2 = (int)(((nyn+1.0)*h)/2.0);
                    // Img_out.at<Vec3b>(y,x) = Img.at<Vec3b>(y2,x2);
                    //data.push_back(Point(x2,y2));
                     map_x.at<int>(y,x) = x2;
                     map_y.at<int>(y,x) = y2;
                }
            }
            else
            {
                map_x.at<int>(y,x) = x;
                map_y.at<int>(y,x) = y;
            }
         }
    }
}
void feye_Space_to_src(fisheye_t &feye,Mat &map_x,Mat &map_y)
{
    const int delx =feye.center.x-feye.r;
    const int dely =feye.center.y-feye.r;
    for (int  y = 0; y <map_x.rows ; y++)
    {
        for (int  x = 0 ; x <map_x.cols ; x++)
        {

             map_x.at<int>(y,x) += delx;
             map_y.at<int>(y,x) += dely;
        }
    }
}

int main()
{
    VideoCapture cap(0);
//    cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
//    cap.set(CV_CAP_PROP_FRAME_WIDTH,480);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

    Mat Img,frame;
    frame = imread("grid2.png");
    if(!frame.data)
    {
        cout<<"fail to open image";
        return 0;
    }
    cvtColor(frame, frame, CV_BGR2GRAY);
    fisheye_t feye;
    feye.center.x = 250;
    feye.center.y = 187;
    feye.r = 250;
    Img = frame.clone();
    imshow("orig",Img);
    fisheye_boarder(frame,Img,feye);
    imshow("after boarder",Img);
    Img = Mat(Img, Rect(0,0,feye.r*2,feye.r*2));
    imshow("after clip",Img);
//    copyMakeBorder (  frame , Img,  abs(frame.cols-frame.rows)/2 ,
//                     abs(frame.cols-frame.rows)/2, 0 , 0 ,BORDER_CONSTANT , Scalar(0,0,0)  ) ;

 //   Img_out = Scalar(0,0,255);

    Mat map_x =Mat(Img.rows,Img.cols,CV_32FC1),
        map_y =Mat(Img.rows,Img.cols,CV_32FC1);
    fisheye_tbl_create(Img,map_x,map_y);

    Point center(Img.rows/2,Img.cols/2);

    feye_Space_to_src(feye,map_x,map_y);

    int delta_x = 200;
    int delta_y = 200;
    Mat Img_out = Mat(delta_x*2,delta_y*2,CV_8UC1);
    int nRows = Img_out.rows;
    int nCols = Img_out.cols;
    uchar *src_ptr;
    uchar *dst_ptr;

    dst_ptr = Img_out.ptr<uchar>(0);
    //scan over Img_out by ptr++ since Img_out is continuous;
    for (int  j =0; j <Img_out.rows; j++)
    {
        for (int  i = 0; i <  Img_out.cols; i++)
        {
            src_ptr = Img.ptr<uchar>(map_y.at<int>(j,i));
            *dst_ptr++ = src_ptr[map_x.at<int>(j,i)];
        }
    }

//    for (int  y = center.y - delta_y; y <  center.y + delta_y; y++)
//    {
//        for (int  x = center.x - delta_x; x <  center.x + delta_x; x++)
//        {
//            Img_out.at<uchar>(y,x) = Img.at<uchar>(map_y.at<int>(y,x),map_x.at<int>(y,x));
//        }
//    }
    imshow("de-fisheye",Img_out);

    waitKey();
//    MyFilledCircle( Img, Point( w/2.0, w/2.0),w/2 );

//    vector<Point> data;



//    Mat src  = imread("image.png",CV_LOAD_IMAGE_COLOR);
//    Mat dst = src.clone();
//    cvtColor(src,src,COLOR_RGB2GRAY);
//    cvtColor(dst,dst,COLOR_RGB2GRAY);
//    rmpData_t rmp_data;
//    rmp_data.cols = w;
//    rmp_data.rows = h;
//    rmp_data.map_x = map_x;
//    rmp_data.map_y = map_y;
//


//    rmpWrite(rmp_data);
//   rmpRead(rmp_data);
//    cout<<rmp_data.cols;
/*
        int nRows = Img.rows;
        int nCols = Img.cols;
        if (Img.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
            cout<<"is continue";
        }

    Mat Img_out(CLIP_HEIGHT,CLIP_WIDTH,CV_8UC1),test;
//     for(;;)
    {
        cap>>frame;
        frame = Mat(frame, Rect(CLIP_ORIGIN_X,CLIP_ORIGIN_Y,CLIP_WIDTH,CLIP_HEIGHT));
        cvtColor(frame, frame, CV_BGR2GRAY);
        test = frame;
//    ===========on the fly method=========================================
//    0.018sec
 double  t = ( double )getTickCount();

    for (int  y = CLIP_ORIGIN_Y ; y < CLIP_HEIGHT+CLIP_ORIGIN_Y; y++)
    {
            for (int  x = CLIP_ORIGIN_X ; x < CLIP_WIDTH + CLIP_ORIGIN_X ; x++)
            {
                Img_out.at<uchar>(x,y)=test.at<uchar>(
                                                       map_x.at<int>(x,y),
                                                       map_y.at<int>(x,y)
                                                );
            }
    }

t = (( double )getTickCount() - t)/getTickFrequency();
//=================================================
//  0.018 sec

//    int j = 0;
//    for( int i = 0; i < rmp_data.rows ; ++i)
//    {
//        out = dst.ptr<uchar>(i);
//        in = src.ptr<uchar>(i);
//        // and both image are in float!
//        for (j = 0; j <rmp_data.cols; ++j)
//        {
//            // Do whatever you want
//            out[ j ] = in[ map_x.at<int>(i,j)];
//            //p[ j ] =src.at<uchar>(map_y.at<int>(i,j),map_x.at<int>(i,j));
//           // p[ j ] = q[rmp_data.cols-j];
//        }
//    }
//=========================================================

//======================
//0.016 1080x1080
//
//    uchar *out;
//    uchar *in;
//
//    int  cnt = 0;
//    vector<Point>::iterator it = data.begin();
//    double  t = ( double )getTickCount();
//
//        for( int i = 0; i < nRows ; ++i)
//        {
//            for (int j = 0; j <nCols; ++j)
//            {
//                in     = Img.ptr<uchar>(data[cnt].y );
//                out    = Img_out.ptr<uchar>(i);
//                out[j] = in[data[cnt].x];
//                cnt++;
//            }
//        }
//================================================
    cout <<  "Times passed in seconds: "  << t << endl;

        imshow("in", Img_out);

         if(waitKey(30) >= 0) break;
//    }

    imshow("out",Img_out);
//    imshow("s",src);

//    imshow("rig",Img);
//    imshow("out",Img_out);
    waitKey(0);
*/

}
