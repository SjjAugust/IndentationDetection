//
// Created by ShiJJ on 2020/8/17.
//
#include "Calibrator.h"
#include <exception>

void Calibrator::genCalibrator(const cv::Size &resolution, int square_width, const std::string &path){
    std::cout << "start generate" << std::endl;
    cv::Mat chess_board = cv::Mat::zeros(resolution, CV_8UC1);
    std::cin.get();
    for(int i = 0; i < chess_board.rows; i++){
        for(int j = 0; j < chess_board.cols; j++){
            if((i / square_width + j / square_width) % 2){
                chess_board.at<uchar>(i, j) = 255;
            }
        }
    }
    std::string file_name = "/calibrator_" + std::to_string(resolution.width) + "_" + std::to_string(resolution.height) 
    + "_" + std::to_string(square_width) + ".jpg"; 
    std::cout << "finish generate" << std::endl;
    cv::imwrite(path + file_name, chess_board); 
}

double Calibrator::calibrateMircoscope(const cv::Mat &calib_pic, const cv::Size &pattern_size, int pixel_each_square, double real_len_per_pixel){
    std::vector<cv::Point> concers;
    int check_status = cv::findChessboardCorners(calib_pic, pattern_size, concers, CV_CALIB_CB_FILTER_QUADS);
    if(!check_status){
        std::cout << "no chessboard" << std::endl;
        std::cin.get();
        return 0;
    }
    cv::Mat canvas = calib_pic.clone();
    for(auto p : concers){
        cv::circle(canvas, p, 2, cv::Scalar(0, 255, 255), 1);
    }
    cv::namedWindow("ccccc", 0);
    cv::imshow("ccccc", canvas);
    cv::waitKey(0);
    double real_square_length = pixel_each_square * real_len_per_pixel;
    std::vector<double> adjacent_distance;
    //行间距
    for(int i = 1; i < concers.size(); i++){
        if(!(i % 8)){
            continue;
        }
        adjacent_distance.push_back(fabs(concers[i].x - concers[i-1].x));
    }
    //列间距
    for(int i = 1; i < pattern_size.height; i++){
        for(int j = 0; j < pattern_size.width; j++){
            adjacent_distance.push_back(fabs(concers[i * pattern_size.width + j].y - concers[(i - 1) * pattern_size.width + j].y));
            cv::Mat temp = calib_pic.clone();
            // cv::circle(temp, concers[i * pattern_size.width + j], 2, cv::Scalar(0, 255, 255), 5);
            // cv::circle(temp, concers[(i - 1) * pattern_size.width + j], 2, cv::Scalar(0, 255, 255), 5);
            // cv::imshow("ccccc", temp);
            // cv::waitKey(0);
        }
    }
    // for(auto d : adjacent_distance){
    //     std::cout << "distance" << d << std::endl;
    // }
    //最小二乘找到最优间距
    std::vector<double>::iterator max_it = std::max_element(adjacent_distance.begin(), adjacent_distance.end());
    std::vector<double>::iterator min_it = std::min_element(adjacent_distance.begin(), adjacent_distance.end());
    double max_val = *max_it, min_val = *min_it;
    std::cout << "max_val:" << max_val << " min_val:" << min_val << std::endl;
    double test_distance = min_val;
    double min_error = INT64_MAX;
    double best_distance = min_val;
    while (test_distance <= max_val){
        double error = 0;
        for(auto d : adjacent_distance){
            error += abs(d - test_distance);
        }
        if(error < min_error){
            min_error = error;
            best_distance = test_distance;
        }
        test_distance += 1;
    }
    std::cout << "best distance:" << best_distance << std::endl;
    return real_square_length / best_distance;
    
}