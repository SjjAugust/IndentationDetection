#include <iostream>
#include "EdgeDetector.h"
#include <string>
#include <vector>
int main() {
    cv::Mat input_mat3 = cv::imread("../pic/test/pic_new1.jpg");
    cv::Mat coin_pic = cv::imread("../pic/calib2.jpg");
    std::vector<EdgeDetector> vec;
    vec.emplace_back(EdgeDetector(input_mat3));
    vec.emplace_back(EdgeDetector(input_mat3));
    vec.emplace_back(EdgeDetector(input_mat3));
    double pixel_len = EdgeDetector::calibrationByCoin(coin_pic, 25);
    std::cout << "pixel_len:" << pixel_len << std::endl;
    std::vector<double> radius;
    std::vector<cv::Point> center;
    cv::Mat res = vec[2].findHoleByBinaryzation(cv::Size(15, 15), 2, radius, center);
    std::cout << pixel_len * radius[0] * 2<< " " << pixel_len * radius[1] * 2<< std::endl;
    for(int i = 0; i < 2; i++){
        std::string text = std::to_string( pixel_len * radius[i] * 2);
        cv::putText(res, text, cv::Point(center[i].x, center[i].y+40), cv::FONT_HERSHEY_COMPLEX, 4,cv::Scalar(0, 0, 255), 4);
    }
    cv::imwrite("../pic/x.jpg", res);



}
