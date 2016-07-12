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
#define CLIP_ORIGIN_X 10
#define CLIP_ORIGIN_Y 100
#define CLIP_WIDTH 640
#define CLIP_HEIGHT 640
struct mapPoint_t
{
    int x;
    int y;
};
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
int main()
{
    VideoCapture cap(0);
//    cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
//    cap.set(CV_CAP_PROP_FRAME_WIDTH,480);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

    Mat frame;
    //cap>>frame;
    frame = imread("image.png",CV_LOAD_IMAGE_COLOR);

    frame = Mat(frame, Rect(CLIP_ORIGIN_X,CLIP_ORIGIN_Y,CLIP_WIDTH,CLIP_HEIGHT));

    if(!frame.data)
    {
        cout<<"fail to load image";
        return 0;
    }
    cvtColor(frame, frame, CV_BGR2GRAY);
    Mat tmp = frame;
    Mat Img = frame;
    Mat Img_out = Img.clone();

 //   Img_out = Scalar(0,0,255);
    double  w = Img.cols;
    double  h = Img.rows;
    Mat map_x =Mat(w, h, CV_32FC1),
        map_y =Mat(w, h, CV_32FC1);
    MyFilledCircle( Img_out, Point( w/2.0, w/2.0),w );

//    mapPoint_t mapPoint;
//    vector<mapPoint_t> mpvec;
    vector<Point> data;
//    mpvec.resize(Img.cols*Img.rows);

    for (int  y = 0 ; y < Img.rows ; y++)
    {
        // normalize y coordinate to -1 ... 1
        double ny = ((2*y)/h)-1;
        // pre calculate ny*ny
        double ny2 = ny*ny;


        for (int  x = 0 ; x < Img.cols ; x++)
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
                    data.push_back(Point(x2,y2));
                    //  map_x.at<int>(y,x) = x2;
                    // map_y.at<int>(y,x) = y2;
                }

            }
            else
                    data.push_back(Point(x,y));

         }
    }

    Mat src  = imread("image.png",CV_LOAD_IMAGE_COLOR);
    Mat dst = src.clone();
    cvtColor(src,src,COLOR_RGB2GRAY);
    cvtColor(dst,dst,COLOR_RGB2GRAY);

   rmpData_t rmp_data;
    rmp_data.cols = w;
    rmp_data.rows = h;
    rmp_data.map_x = map_x;
    rmp_data.map_y = map_y;



//    rmpWrite(rmp_data);
//   rmpRead(rmp_data);
//    cout<<rmp_data.cols;

//    ===========on the fly method=========================================
//    0.026sec
//    for (int  i = 0 ; i < rmp_data.rows ; i++)
//    {
//            for (int  j = 0 ; j < rmp_data.cols ; j++)
//            {
//
//
//                dst.at<uchar>(i,j)=src.at<uchar>(
//                                                       map_y.at<int>(i,j),
//                                                       map_x.at<int>(i,j)
//                                                );
//            }
//    }

//=================================================
//  0.021 sec

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

    for(;;)
    {
            double  t = ( double )getTickCount();

        cap>>frame;
        frame = Mat(frame, Rect(CLIP_ORIGIN_X,CLIP_ORIGIN_Y,CLIP_WIDTH,CLIP_HEIGHT));
        cvtColor(frame, frame, CV_BGR2GRAY);
        uchar *out;
        uchar *in;
        Img = frame;
        int cnt = 0;
        for( int i = 0; i < Img.rows ; ++i)
        {
            for (int j = 0; j <Img.cols; ++j)
            {
                in     = Img.ptr<uchar>(data[cnt].y);
                out    = Img_out.ptr<uchar>(i);
                out[j] = in[data[cnt].x];
                cnt++;
            }
        }
    t = (( double )getTickCount() - t)/getTickFrequency();
    cout <<  "Times passed in seconds: "  << t << endl;

        imshow("int", Img_out);
        if(waitKey(30) >= 0) break;
    }

    imshow("out",Img_out);
//    imshow("s",src);

//    imshow("rig",Img);
//    imshow("out",Img_out);
    waitKey(0);


}
