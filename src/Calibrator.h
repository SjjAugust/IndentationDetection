//
// Created by ShiJJ on 2020/8/17.
//

#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <climits>
#include <cmath>
class Calibrator
{
private:
    
public:
    static double calibrateMircoscope(const cv::Mat &calib_pic, const cv::Size &pattern_size, int pixel_each_square, double real_len_per_pixel);
    static void genCalibrator(const cv::Size &resolution, int square_width, const std::string &path);
};

