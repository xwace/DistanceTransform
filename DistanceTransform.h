//
// Created by yc on 23-3-10.
//

#ifndef CMAKELISTS_TXT_DISTANCETRANSFORM_H
#define CMAKELISTS_TXT_DISTANCETRANSFORM_H
#include <opencv2/opencv.hpp>

namespace DT{
    void distanceTransform( cv::InputArray _src, cv::OutputArray _dst,
                            int distanceType, int maskSize, int dstType);
}

#endif //CMAKELISTS_TXT_DISTANCETRANSFORM_H
