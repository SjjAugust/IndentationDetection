#include <iostream>
#include "EdgeDetector.h"
#include <string>
#include <vector>
int main(int argc, char *argv[]) {
    std::string spci_name = argv[1];
    std::string pic_name = "../pic/test/" + spci_name + ".jpg";
    cv::Mat input_mat3 = cv::imread(pic_name);
    cv::Mat coin_pic = cv::imread("../pic/calib2.jpg");
    EdgeDetector edgedetector(input_mat3, std::stoi(argv[2]));
    double pixel_len = EdgeDetector::calibrationByCoin(coin_pic, 25);
    std::cout << "pixel_len:" << pixel_len << std::endl;
    std::vector<double> radius;
    std::vector<cv::Point> center;
    cv::Mat res = edgedetector.findHoleByBinaryzation(cv::Size(15, 15), 2, radius, center);
    std::cout << pixel_len * radius[0] * 2<< " " << pixel_len * radius[1] * 2<< std::endl;
    for(int i = 0; i < 2; i++){
        std::string text = std::to_string( pixel_len * radius[i] * 2);
        cv::putText(res, text, cv::Point(center[i].x, center[i].y+40), cv::FONT_HERSHEY_COMPLEX, 4,cv::Scalar(0, 0, 255), 4);
    }
    cv::imwrite("../pic/x.jpg", res);
    std::cin.get();


}
