#include <iostream>
#include "EdgeDetector.h"
#include <string>
#include <vector>
int main(int argc, char *argv[]) {
    std::string spci_name = argv[1];
    std::string pic_name = "../pic/test/" + spci_name + ".jpg";
    cv::Mat input_mat3 = cv::imread(pic_name);
    cv::Mat coin_pic = cv::imread("../pic/test/calib_new1.jpg");
    EdgeDetector edgedetector(input_mat3, std::stoi(argv[2]));
    // double pixel_len = edgedetector.calibrationByCoin(coin_pic, 36);
    double pixel_len = 0.0257072;
    edgedetector.setPixelLength(pixel_len);
    std::cout << "pixel_len:" << pixel_len << std::endl;
    std::cin.get();
    std::vector<double> radius;
    std::vector<cv::Point> center;
    // cv::Mat res = edgedetector.findHoleByBinaryzation(cv::Size(15, 15), 2, radius, center);
    cv::Mat res = edgedetector.findHoleSubPixel(cv::Size(15, 15), 2, radius, center);
    std::cout << pixel_len * radius[0]<< " " << pixel_len * radius[1]<< std::endl;
    for(int i = 0; i < 2; i++){
        std::string text = std::to_string( pixel_len * radius[i]);
        cv::putText(res, text, cv::Point(center[i].x, center[i].y+40), cv::FONT_HERSHEY_COMPLEX, 4,cv::Scalar(0, 0, 255), 4);
    }
    cv::imwrite("../pic/process/x.jpg", res);
    std::cin.get();


}
