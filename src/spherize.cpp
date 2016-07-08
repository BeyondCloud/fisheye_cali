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

    Mat Img_out = imread("image.png",CV_LOAD_IMAGE_COLOR);;
    Img_out = Scalar(0,0,0);
    double  w = Img.cols;
    double  h = Img.rows;


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
            // discard pixels outside from circle!
            if (0.0<=r&&r<=1.0)
            {
                double nr = sqrt(1.0-r*r);
                // new distance is between 0 ... 1
                nr = (r + (1.0-nr)) /2.0;
                // discard radius greater than 1.0
                if (nr<=1.0)
                    {
                        // calculate the angle for polar coordinates
                        double theta = atan2(ny,nx);
                        // calculate new x position with new distance in same angle
                        double nxn = nr*cos(theta);
                        // calculate new y position with new distance in same angle
                        double nyn = nr*sin(theta);
                        // map from -1 ... 1 to image coordinates
                        int x2 = (int)(((nxn+1.0)*w)/2.0);
                        // map from -1 ... 1 to image coordinates
                        int y2 = (int)(((nyn+1.0)*h)/2.0);
                    //    if(x2<w && y2 < h)
                            Img_out.at<Vec3b>(y,x) = Img.at<Vec3b>(y2,x2);

                     }
            }

/*
            y0 = y - Center.y;
            x0 = x - Center.x;
            Dis = sqrt(pow(x0,2) + pow(y0,2));
            theta = atan2(y0,x0);
            new_Dis = Dis*(1 + k*pow(Dis,2));
            new_x = (int)(new_Dis * cos(theta)) + Center.x;
            new_y = (int)(new_Dis * sin(theta)) + Center.y;
            if(new_x <width && new_y <height && new_x >=0 && new_y>=0)
            {
                Img_out.at<Vec3b>(new_y,new_x) = Img.at<Vec3b>(y,x);
            }
*/


         }
    }

    imshow("rig",Img);
    imshow("out",Img_out);
    waitKey(0);

}


