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
int main()
{
    Mat Img;
    Img = imread("image.png",CV_LOAD_IMAGE_COLOR);
//    cvtColor(Img, Img, CV_BGR2GRAY);
    if(!Img.data)
    {
        cout<<"fail to load image";
        return 0;
    }

    Mat Img_out = imread("image.png",CV_LOAD_IMAGE_COLOR);
    Mat test = imread("image.png",CV_LOAD_IMAGE_COLOR);
    Img_out = Scalar(0,0,255);
    double  w = Img.cols;
    double  h = Img.rows;
    Mat mat_x =Mat(w, h, CV_16S),
        mat_y =Mat(w, h, CV_16S);

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
                    // calculate new y position with new distance in same angle
                    double nyn = nr*sin(theta);
                    // map from -1 ... 1 to image coordinates
                    int x2 = (int)(((nxn+1.0)*w)/2.0);
                    // map from -1 ... 1 to image coordinates
                    int y2 = (int)(((nyn+1.0)*h)/2.0);
                    // if(x2<w && y2 < h)
                    Img_out.at<Vec3b>(y,x) = Img.at<Vec3b>(y2,x2);
                    if(!mat_x.at<int>(y,x))
                        mat_x.at<int>(y,x) = x2;
                    if(!mat_y.at<int>(y,x))
                        mat_y.at<int>(y,x) = y2;

                }
            }
         }
    }
//    FileStorage fs("test.xml", cv::FileStorage::WRITE);
//    fs << "cols" << w;
//    fs << "rols" << h;
//    fs << "mat_x" << mat_x;
//    fs << "mat_y" << mat_y;
//    fs.release();
    Mat mat_nx =Mat(w, h, CV_32S),
        mat_ny =Mat(w, h, CV_32S);

    Mat lookup = imread("image.png",CV_LOAD_IMAGE_COLOR);
    Mat nlookup= imread("image.png",CV_LOAD_IMAGE_COLOR);

    FileStorage fs("test.xml", cv::FileStorage::READ);
    int rw;
    int rh;
    fs["cols"] >> rw;
    fs["rols"] >> rh;
    fs["mat_x"] >> mat_nx;
    fs["mat_y"] >> mat_ny;
    for (int  y = 0 ; y < rh ; y++)
    {
            for (int  x = 0 ; x < rw ; x++)
            {
                nlookup.at<Vec3b>(y,x)=lookup.at<Vec3b>(
                                                       mat_ny.at<int>(y,x),
                                                       mat_nx.at<int>(y,x)
                                                       );
            }
    }
    imshow("t",nlookup);
    fs.release();


//    imshow("rig",Img);
//    imshow("out",Img_out);
    waitKey(0);

}


