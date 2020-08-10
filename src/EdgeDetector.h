//
// Created by ShiJJ on 2020/7/27.
//

#ifndef INDENTATIONDETECTION_EDGEDETECTOR_H
#define INDENTATIONDETECTION_EDGEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#define PI acos(-1)


class EdgeDetector {
private:
    cv::Mat ori_pic_;
    cv::Mat dst_pic_;
    static const int MAX_CONTOURS_LENGTH = 650;
    static const int MIN_CONTOURS_LENGTH = 290;
    static const int BINARYZATION_THRESHOLD = 140;
    const double MAX_AREA = 29999;
    const double MIN_AREA = 200;

    void preProcess(const cv::Size& gauss_kernel_size);
    cv::Mat detectEdge();

    double calDistance(const cv::Point& p1, const cv::Point& p2);
    double calAngle(const cv::Point& p1, const cv::Point& p2);
    double calPointToLineDistance(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2);
    static bool compare(std::map<std::string, double> map1, std::map<std::string, double> map2);
    static void imfill(cv::Mat& mat);
    /*------对比度提升相关------*/
    bool getVarianceMean(cv::Mat &scr, cv::Mat &meansDst, cv::Mat &varianceDst, int winSize);
    bool adaptContrastEnhancement(cv::Mat &scr, cv::Mat &dst, int winSize,int maxCg);
    /*------对比度提升相关------*/
public:
    EdgeDetector(const cv::Mat& input_pic);
    cv::Mat findHoleByBinaryzation(const cv::Size &gauss_kernel_size, int hole_num, std::vector<double>& radius, std::vector<cv::Point>& center);
    static double calibrationByCoin(const cv::Mat& coin_pic, double length);
};


#endif //INDENTATIONDETECTION_EDGEDETECTOR_H
