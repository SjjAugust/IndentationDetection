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
#include <thread>
#include <mutex>
#include <sys/time.h>
#include <sstream>
#include <utility>
#define PI acos(-1)


class EdgeDetector {
private:
    cv::Mat ori_pic_;
    cv::Mat dst_pic_;
    std::vector<cv::Mat> input_pics;
    int MAX_CONTOURS_LENGTH = 99650;
    int MIN_CONTOURS_LENGTH = 2000;
    int BINARYZATION_THRESHOLD = 138;
    int MAX_AREA = 99999999;
    int MIN_AREA = 100000;
    int MIN_RADIUS = 300;
    int MAX_RADIUS = 600;
    double confidence = 0.999999;
    int PIXEL_AROUND_THRESHOLD = 15000;
    int PIXEL_NEIGHBOUR = 5000;
    bool MULTIPLE_INPUT;
    struct Goodness
    {
        double sum_distance = 0;
        double rate = 0;
        int inliners = 0;
        unsigned long long total = 0;
    };
    double PIXEL_LENGTH = 0;
    std::mutex ransac_mutex;
    static std::vector<std::pair<cv::Vec3d, EdgeDetector::Goodness>> compare_info; 

    void preProcess(const cv::Size& gauss_kernel_size);
    cv::Mat detectEdge();
    const double calDistance(const cv::Point& p1, const cv::Point& p2);
    const double calAngle(const cv::Point& p1, const cv::Point& p2);
    double calPointToLineDistance(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2);
    static bool compare(std::map<std::string, double> map1, std::map<std::string, double> map2);
    static bool compareC(std::map<std::string, double> map1, std::map<std::string, double> map2);
    static void imfill(cv::Mat& mat);
    cv::Mat composePic(const std::vector<cv::Mat> &pics);
    /*type == 0:用拟合后坐标计算拟合度，circle_info可以填null
      type == 1:用原始坐标计算拟合度，aft_contour可以填null*/
    Goodness calGoodnessOfFit(const std::vector<cv::Point> &ori_contour, const std::vector<cv::Point> &aft_contour, const cv::Vec3d &circle_info, int type);
    Goodness calGoodnessOfFit(const std::vector<cv::Point> &ori_contour, const cv::Vec3d &circle_info, const cv::Size &canvas_size);
    /*------对比度提升相关------*/
    bool getVarianceMean(cv::Mat &scr, cv::Mat &meansDst, cv::Mat &varianceDst, int winSize);
    bool adaptContrastEnhancement(cv::Mat &scr, cv::Mat &dst, int winSize,int maxCg);
    /*------对比度提升相关------*/
    /*------根据三点计算圆心和半径------*/
    cv::Vec3d calCircleByThreePoints(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3);
    cv::Vec3d getCircleByRANSAC(const std::vector<cv::Point> &contour, int cycle_num, double threshold, const cv::Size &canvas_size);
    void processCalculation(const std::vector<cv::Point> &contour, int &count, const cv::Size &canvas_size);
    int updateNumIters(double p, double ep, int model_points, int max_iters);
    /*------根据三点计算圆心和半径------*/
    double getSubPixelLength(const std::vector<cv::Point>& contour, const cv::Mat& gray_pic);
    /*------获取当前纳秒时间------*/
    long getTimeNs();
    /*------自适应阈值相关------*/
    double calEntropy(const cv::Mat &hist, int begin, int end);
    double calProbability(const cv::Mat &hist, int index);
    int getKswThreshold(const cv::Mat &pic);
    int getAdaptiveThreshold(const cv::Mat &pic, const int around_threshold, const int neigbour);
    /*------设置参数------*/
    void setParameter();

public:
    static struct parameter{
        int min_radius = 0;
        int max_radius = 0;
        int min_contours_length = 0;
        int max_contours_length = 0;
        int min_area = 0;
        int max_area = 0;
        int pixel_around_threshold = 0;
        int pixel_neighbour = 0;
        int binaryzation_threshold = 0;
        double pixel_len = 0;
    } para;
    EdgeDetector();
    EdgeDetector(const cv::Mat& input_pic);
    EdgeDetector(const std::vector<cv::Mat>& input_pics);
    cv::Mat findHoleByBinaryzation(const cv::Size &gauss_kernel_size, int hole_num, std::vector<double>& radius, std::vector<cv::Point>& center_vec);
    cv::Mat findHoleSubPixel(const cv::Size &gauss_kernel_size, int hole_num, std::vector<double>& diam, std::vector<cv::Point>& center_vec);
    double calibrationByCoin(const cv::Mat& coin_pic, double length);
};


#endif //INDENTATIONDETECTION_EDGEDETECTOR_H
