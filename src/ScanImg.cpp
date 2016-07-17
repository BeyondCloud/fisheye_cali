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
struct fisheye_t
{
    Point orig;
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
//create a square fisheye table
void fisheye_tbl_create(Mat &Img,Mat &map_x,Mat &map_y,fisheye_t &feye)
{
    Point local_orig = Point(feye.center.x-feye.r,feye.center.y-feye.r);
    cout<<"local orig(x,y)=" << local_orig.x<<" "<<local_orig.y<<endl;
    int clip_l = max(local_orig.x,0);
    int clip_r = min(local_orig.x+feye.r*2,Img.cols);
    int clip_t = max(local_orig.y,0);
    int clip_d = min(local_orig.y+feye.r*2,Img.rows);
    bool is_legal = true;
    cout<<"l:"<<clip_l<<"r:"<<clip_r<<"t:"<<clip_t<<"d:"<<clip_d<<endl;
    for (int  y = 0; y <Img.rows ; y++)
    {
        // normalize y coordinate to -1 ... 1
        //double ny = (2*(y-local_orig.y)/w)-1;
        double ny =  (y-feye.center.y)/(double)feye.r;

        double ny2 = ny*ny;
        for (int  x = 0 ; x <Img.cols ; x++)
        {
            // x coordinate to -1 ... 1
          //  double nx = (2*(x-local_orig.x)/w)-1;
            double nx =  (x -feye.center.x)/(double)feye.r;
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
                //nr = sin(r*pi/2.0);
                //nr = (2.0*r-1.0+sqrt((-4.0)*r*r + 4.0*r+1))/2.0;
                //nr = (2.0*r-sqrt(0.3)+sqrt((-4.0)*r*r + 4.0*r+0.3))/2.0;
                double a1 = 0.3,b1 =0.7;
                double tmp = ((-2.0)*a1+sqrt(4.0*a1*a1+4.0*r*(1.0-2.0*a1)))/(2.0-4.0*a1);
                nr = b1*2*(tmp-tmp*tmp)+tmp*tmp;
                // discard radius greater than 1.0 or smaller than 0
                if (0.0 <= nr)
                {
                    // calculate the angle for polar coordinates
                    // calculate new x position with new distance in same angle
                    double nxn = nr*cos(theta);
                    // calculate new y position with new distance in same angle
                    double nyn = nr*sin(theta);
                    // map from -1 ... 1 to image coordinates
                    int x2 = (int)(((nxn+1.0)*(double)feye.r)+local_orig.x);
                    // map from -1 ... 1 to image coordinates
                    int y2 = (int)(((nyn+1.0)*(double)feye.r)+local_orig.y);
                    // Img_out.at<Vec3b>(y,x) = Img.at<Vec3b>(y2,x2);
                    //data.push_back(Point(x2,y2));
                    //if new coordinate is legal ,than record it new coordinate.
                    //otherwise it will map to it self
                    if(0 <= y2 && y2 < Img.rows && 0 <= x2 && x2 < Img.cols)
                    {
                        map_x.at<int>(y,x) = x2;
                        map_y.at<int>(y,x) = y2;
                    }
                    else
                        is_legal = false;
                }
            }
            else
                is_legal = false;
            if(!is_legal)
            {
                map_x.at<int>(y,x) = x;
                map_y.at<int>(y,x) = y;
                is_legal = true;
            }
         }
    }
}

int main()
{
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
    cap.set(CV_CAP_PROP_FRAME_WIDTH,1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,720);
    Mat Img,frame;
    frame = imread("fish3.jpg");
    if(!frame.data)
    {
        cout<<"fail to open image";
        return 0;
    }
    cvtColor(frame, frame, CV_BGR2GRAY);
    fisheye_t feye;
    Img = frame.clone();
    feye.center.x = Img.cols/2;  //640
    feye.center.y = Img.rows/2;  //360
    feye.r = 600;
    Point clip_orig(0,0);
    int clip_w = 1280;
    int clip_h = 720;
    if(clip_w+clip_orig.x > frame.cols)
    {
        cout << "invalid clip window: except clip_w+clip_orig.x < frame.cols:"<<frame.cols<<"\n";
        return 0;
    }
    else if(clip_h+clip_orig.y > frame.rows)
    {
        cout << "invalid clip window: except clip_h+clip_orig.y < frame.rows:"<<frame.rows<<"\n";
        return 0;
    }
    Mat map_x =Mat(Img.rows,Img.cols,CV_32FC1),
        map_y =Mat(Img.rows,Img.cols,CV_32FC1);
    //now Img is a square image
    //square table will be created and mapped back to src image coordinate
    fisheye_tbl_create(Img,map_x,map_y,feye);
    Mat Img_out = Mat(clip_h,clip_w,CV_8UC1);
    uchar *src_ptr;
    uchar *dst_ptr;
// for(;;)
//    {
//        cap>>frame;
//      cvtColor(frame, frame, CV_BGR2GRAY);
//      imshow("src_Img",Img);
//      imshow("cap_Img",frame);
 //     waitKey();
 //       Img = frame;
        //scan over Img_out by ptr++ since Img_out is continuous;
        //src image is in fish eye  space
        dst_ptr = Img_out.ptr<uchar>(0);
        for (int  j =clip_orig.y; j <clip_orig.y + clip_h; j++)
        {
            for (int  i = clip_orig.x; i <clip_orig.x + clip_w; i++)
            {
             //   cout<<"x"<<map_x.at<int>(j,i)<<"y"<<map_x.at<int>(j,i)<<"\n";
                src_ptr = Img.ptr<uchar>(map_y.at<int>(j,i));
                *dst_ptr++ = src_ptr[map_x.at<int>(j,i)];
            }
        }
        imshow("de-fisheye",Img_out);
        waitKey();
//        if(waitKey(30)>=0)
//            break;
//    }
//    for (int  y = center.y - delta_y; y <  center.y + delta_y; y++)
//    {
//        for (int  x = center.x - delta_x; x <  center.x + delta_x; x++)
//        {
//            Img_out.at<uchar>(y,x) = Img.at<uchar>(map_y.at<int>(y,x),map_x.at<int>(y,x));
//        }
//    }


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
