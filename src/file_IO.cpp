#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
int main()
{
    //TO WRITE
    Mat myMat;
    int data[10] = { 1, 2, 3, 1, 2, 3, 1, 2,3 };
    int data2[10] = { 1, 2, 1, 1, 2, 1, 1, 2,1 };
    Mat mat_x =Mat(3, 3, CV_32S, data),
        mat_y =Mat(3, 3, CV_32S, data2),
        mat_i =Mat(3, 3, CV_32S),
        mat_j =Mat(3, 3, CV_32S);

    int cnt = 0;
   // myMat = imread("grid.png",CV_LOAD_IMAGE_COLOR);
    for(int i = 0;i<3;i++)
    {
        for(int j = 0;j<3;j++)
        {
            cout<<mat_x.at<int>(i,j)<<","<<mat_y.at<int>(i,j)<<"\t";
        }
        cout<<"\n";
    }
    cv::FileStorage fs("test.xml", cv::FileStorage::READ);
//    fs << "mat_x" << mat_x;
//    fs << "mat_y" << mat_y;
    fs["mat_x"] >> mat_i;
    fs["mat_y"] >> mat_j;
        for(int i = 0;i<3;i++)
    {
        for(int j = 0;j<3;j++)
        {
            cout<<mat_i.at<int>(i,j)<<","<<mat_j.at<int>(i,j)<<"\t";
        }
        cout<<"\n";
    }


    fs.release();
//     remap( src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );

    waitKey();
}
