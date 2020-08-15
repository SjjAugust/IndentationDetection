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
#include <climits>
#include <cstdlib>
#define PI acos(-1)


class EdgeDetector {
private:
    cv::Mat ori_pic_;
    cv::Mat dst_pic_;
    std::vector<cv::Mat> input_pics;
    static const int MAX_CONTOURS_LENGTH = 650;
    static const int MIN_CONTOURS_LENGTH = 290;
    int BINARYZATION_THRESHOLD = 138;
    const double MAX_AREA = 29999;
    const double MIN_AREA = 200;
    bool MULTIPLE_INPUT;
    struct Goodness
    {
        double sum_distance = 0;
        double rate = 0;
    };
    double pixel_length = 0;
    

    void preProcess(const cv::Size& gauss_kernel_size);
    cv::Mat detectEdge();

    const double calDistance(const cv::Point& p1, const cv::Point& p2);
    const double calAngle(const cv::Point& p1, const cv::Point& p2);
    double calPointToLineDistance(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2);
    static bool compare(std::map<std::string, double> map1, std::map<std::string, double> map2);
    static void imfill(cv::Mat& mat);
    cv::Mat composePic(const std::vector<cv::Mat> &pics);
    Goodness calGoodnessOfFit(const std::vector<cv::Point> &ori_contour, const std::vector<cv::Point> &aft_contour);
    Goodness calGoodnessOfFit(const std::vector<cv::Point> &ori_contour, const cv::Vec3d &circle_info, const cv::Size &canvas_size);
    /*------对比度提升相关------*/
    bool getVarianceMean(cv::Mat &scr, cv::Mat &meansDst, cv::Mat &varianceDst, int winSize);
    bool adaptContrastEnhancement(cv::Mat &scr, cv::Mat &dst, int winSize,int maxCg);
    /*------对比度提升相关------*/
    /*------根据三点计算圆心和半径------*/
    cv::Vec3d calCircleByThreePoints(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3);
    cv::Vec3d getCircleByRANSAC(const std::vector<cv::Point> &contour, int cycle_num, double threshold, const cv::Size &canvas_size);
    /*------根据三点计算圆心和半径------*/
    double getSubPixelLength(const std::vector<cv::Point>& contour, const cv::Mat& gray_pic);
public:
    EdgeDetector();
    EdgeDetector(const cv::Mat& input_pic, int bin_threshold);
    EdgeDetector(const std::vector<cv::Mat>& input_pics, int bin_threshold);
    cv::Mat findHoleByBinaryzation(const cv::Size &gauss_kernel_size, int hole_num, std::vector<double>& radius, std::vector<cv::Point>& center_vec);
    cv::Mat findHoleSubPixel(const cv::Size &gauss_kernel_size, int hole_num, std::vector<double>& diam, std::vector<cv::Point>& center_vec);
    double calibrationByCoin(const cv::Mat& coin_pic, double length);
    void setPixelLength(double pix);
};


#endif //INDENTATIONDETECTION_EDGEDETECTOR_H
