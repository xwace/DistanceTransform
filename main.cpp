#include "DistanceTransform.h"

int main() {

    Mat src(10, 10, 0, Scalar(0));
    Mat dst(src.size(), CV_32F,Scalar(11));

    src.row(0) = 37;
    src.at<uchar>(0,0) = 0;
    src.at<uchar>(0,3) = 0;
    src.at<uchar>(0,5) = 0;
    src(Range(3, 8), Range(3, 8)).setTo(37);

    //use distanceTransform_L1_8U
    DT::distanceTransform(src,dst,DIST_L1,3,CV_8U);
    //use trueDistTrans
    DT::distanceTransform(src,dst,DIST_USER,DIST_MASK_PRECISE,CV_32F);
    cout<<"dst\n"<<dst<<endl;
}
