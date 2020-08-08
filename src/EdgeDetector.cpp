//
// Created by ShiJJ on 2020/7/27.
//

#include "EdgeDetector.h"

EdgeDetector::EdgeDetector(const cv::Mat& input_pic){
    ori_pic_ = input_pic.clone();
}

cv::Mat EdgeDetector::cannyDetect(int min_threshold, int max_threshold, int sobel_size) {
    cv::Canny(ori_pic_, dst_pic_, min_threshold, max_threshold, sobel_size);
    cv::dilate(dst_pic_, dst_pic_, cv::noArray());
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat savedGrayMat = cv::Mat::zeros(dst_pic_.rows, dst_pic_.cols, CV_8UC1);
    cv::findContours(dst_pic_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(savedGrayMat, contours, 0, cv::Scalar(255), cv::FILLED);
    std::cout << "contours size" << contours.size() << std::endl;
    return dst_pic_.clone();
}



void EdgeDetector::preProcess(const cv::Size& gauss_kernel_size) {
    dst_pic_ = ori_pic_.clone();
    adaptContrastEnhancement(dst_pic_, dst_pic_, 15, 10);
    cv::imwrite("../pic/enhance.jpg", dst_pic_);
    cv::cvtColor(dst_pic_, dst_pic_, cv::COLOR_BGR2GRAY);
//    cv::GaussianBlur(dst_pic_, dst_pic_, gauss_kernel_size, 0, 0);
    cv::medianBlur(dst_pic_, dst_pic_, 5);
}

cv::Mat EdgeDetector::detectEdge() {
    cv::adaptiveThreshold(dst_pic_, dst_pic_, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    cv::imwrite("../pic/step1.jpg", dst_pic_);
    cv::medianBlur(dst_pic_, dst_pic_, 9);
    cv::imwrite("../pic/step2.jpg", dst_pic_);
    cv::bitwise_not(dst_pic_, dst_pic_);
    imfill(dst_pic_);
    cv::imwrite("../pic/step3.jpg", dst_pic_);
    cv::namedWindow("x", 0);
    cv::imshow("x", dst_pic_);
    cv::waitKey(0);
    return dst_pic_.clone();
}

cv::Mat EdgeDetector::findHole(const cv::Size &gauss_kernel_size, int min_threshold, int max_threshold, int sobel_size) {
    preProcess(gauss_kernel_size);
    cv::Mat canny_mat = cannyDetect(min_threshold, max_threshold, sobel_size);
    std::vector<std::vector<cv::Point>> all_contours_vec;
    cv::findContours(canny_mat, all_contours_vec, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    std::vector<bool> is_arc = judgeArc(all_contours_vec);


    cv::Mat canvas = cv::Mat::zeros(canny_mat.size(), canny_mat.type());
    for(int i = 0; i < all_contours_vec.size(); i++){
        if(all_contours_vec[i].size() < MIN_CONTOURS_LENGTH || all_contours_vec[i].size() > MAX_CONTOURS_LENGTH){
            is_arc[i] = false;
        }
        if(is_arc[i] == true){
//            std::cout << "length:" << all_contours_vec[i].size() << std::endl;
            cv::drawContours(canvas, all_contours_vec, i, cv::Scalar(255, 0, 0));
            devideCurve(all_contours_vec[i], 6);
        }
    }
    cv::namedWindow("x", 0);
    cv::imshow("x", canvas);
    cv::waitKey(0);


    return canvas;

}

std::vector<bool> EdgeDetector::judgeArc(const std::vector<std::vector<cv::Point>> &all_contour_vec) {
    std::vector<bool> is_arc;
    for(auto vec : all_contour_vec){
        cv::Point start = vec[0], end = vec[vec.size()-1];
        double angle0 = calAngle(start, end);
        std::vector<double> angle_vec;
        for(int j = 1; j < vec.size(); j++){
            double angle1 = calAngle(start, vec[j]);
            angle_vec.push_back(angle1);
        }
        std::vector<bool> judge_vec;
        for(auto angle : angle_vec){
            bool status = angle > angle0;
            judge_vec.push_back(status);
        }
        bool is_cruve = true;
        bool temp = judge_vec[0];
        for(auto status : judge_vec){
            if(status != temp){
                is_cruve = false;
                break;
            }
        }
        is_arc.push_back(is_cruve);
    }
    return is_arc;
}

double EdgeDetector::calDistance(const cv::Point& p1, const cv::Point& p2) {
    return sqrt(pow((p2.y - p1.y), 2) + pow((p2.x - p1.x), 2));
}

double EdgeDetector::calAngle(const cv::Point &p1, const cv::Point &p2) {
    double angle = 0;
    double len = calDistance(p1, p2);
    if(p2.y - p1.y > 0){
        angle = acos((p2.x - p1.x)/len);
    } else {
        angle = PI + PI - acos((p2.x - p1.x)/len);
    }
    return angle;
}

double EdgeDetector::calPointToLineDistance(const cv::Point &p0, const cv::Point &p1, const cv::Point &p2) {
    double dis = 0;
    double a, b, c,  p, S,h;
    a = calDistance(p0, p1);
    b = calDistance(p0, p2);
    c = calDistance(p1, p2);
    p = (a + b + c)/2;
    S = sqrt(p*(p-a)*(p - b)*(p - c) );
    h = S * 2 / a;
    dis = h ;
    return dis;

}

double EdgeDetector::calCurvity(const std::vector<cv::Point> &curve) {
    double curvity = 0;
    if(curve.size() < 4){
        std::cout << "curve length is less than 4!" << std::endl;
        return 0;
    }
    cv::Point start = curve[0], end = curve[curve.size()-1];
    double chord_length = calDistance(start, end);
    double h = 0;
    for(auto p : curve){
        double temp = calPointToLineDistance(p, start, end);
        if(h < temp){
            h = temp;
        }
    }
    double r = 0.5 * (pow(chord_length, 2) / h + h);
    return 1 / r;
}

std::vector<std::vector<cv::Point>> EdgeDetector::devideCurve(const std::vector<cv::Point> &curve, int k) {
    if(curve.size() < 4){
        std::cout << "curve is too short!" << std::endl;
        return std::vector<std::vector<cv::Point>>();
    }
    int length = curve.size();
    std::vector<std::vector<cv::Point>> result;
    //计算曲线上每个点的曲率
    std::vector<double> all_curvity;
    std::cout << "curve length:" << length << std::endl;
    for(int i = 0; i < length; i++){
        std::vector<cv::Point> curve_part;
        for(int j = 0; j < k; j++){
            curve_part.push_back(curve[(i+j)%length]);
        }
        double curvity = calCurvity(curve_part);
        all_curvity.push_back(curvity);
    }
    //计算分割后曲线个数
    bool is_done = false;
    int curve_num = 1;
    for(int i = 0; i < all_curvity.size(); i++){
        double threshold = 0.1;
        if(all_curvity[i] < threshold && all_curvity[i] > 0.03){
            if(!is_done){
                is_done = true;
                curve_num++;
            }
        } else {
            is_done = false;
        }
    }
    std::cout << "devide num:" << curve_num << std::endl;
    return result;
}

cv::Mat EdgeDetector::findHoleByBinaryzation(const cv::Size &gauss_kernel_size, int hole_num, std::vector<double>& radius, std::vector<cv::Point>& center_vec) {
    preProcess(gauss_kernel_size);
    cv::Mat bin_mat = detectEdge();

    std::vector<std::vector<cv::Point>> all_contours_vec;
    cv::findContours(bin_mat, all_contours_vec, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    cv::Mat canvas = cv::Mat::zeros(bin_mat.size(), bin_mat.type());
    cv::Mat ret = ori_pic_.clone();
    cv::Mat bin_canvas = bin_mat.clone();
    cv::cvtColor(bin_canvas, bin_canvas, cv::COLOR_GRAY2BGR);
    std::vector<std::map<std::string, double>> area;
    for(int i = 0; i < all_contours_vec.size(); i++){
        double s = cv::contourArea(all_contours_vec[i]);
        std::map<std::string, double> map;
        map["idx"] = i;
        map["area"] = s;
        area.push_back(map);
    }
    std::sort(area.begin(), area.end(), compare);
    int count = 0;
    for(int i = 0; i < area.size(); i++){
        std::cout << "the " << i << "circumference is:" << all_contours_vec[area[i]["idx"]].size() << std::endl;
        if(count == 2){
            break;
        }
        if(all_contours_vec[area[i]["idx"]].size() > MIN_CONTOURS_LENGTH &&
        all_contours_vec[area[i]["idx"]].size() < MAX_CONTOURS_LENGTH){
            count++;
        } else {
            continue;
        }
        cv::drawContours(bin_canvas, all_contours_vec, area[i]["idx"], cv::Scalar(0 ,0 , 255), 3);
        cv::RotatedRect rrt = cv::fitEllipse(all_contours_vec[area[i]["idx"]]);
        double axis1 = rrt.size.width, axis2 = rrt.size.height;
        double r = (axis1 + axis2) / 4;
        cv::Point center = rrt.center;
        cv::circle(canvas, center, r, cv::Scalar(255, 0, 0), 3);
        cv::circle(ret, center, (int)r, cv::Scalar(0, 0, 255), 3);
//        cv::ellipse(ret, rrt, cv::Scalar(0 ,0 ,255), 1);
        std::cout << "hole" << i << "'s radius: " << r << std::endl;
        radius.push_back(r);
        center_vec.push_back(center);
    }
    cv::imwrite("../pic/step4.jpg", canvas);
    cv::imwrite("../pic/step3_1.jpg", bin_canvas);
    return ret;
}

bool EdgeDetector::compare(std::map<std::string, double> map1, std::map<std::string, double> map2){
    return map1["area"] > map2["area"];
}

double EdgeDetector::calibrationByCoin(const cv::Mat &coin_pic, double length) {
    double pixel_length = 0;
    cv::Mat pic = coin_pic.clone();
    cv::cvtColor(pic, pic, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(pic, pic, cv::Size(9, 9), 0, 0);
    cv::medianBlur(pic, pic, 9);
//    cv::threshold(pic, pic, 100, 255, cv::THRESH_BINARY);
    cv::adaptiveThreshold(pic, pic, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    cv::bitwise_not(pic, pic);
    imfill(pic);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(pic, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    std::vector<std::map<std::string, double>> area;
    for(int i = 0; i < contours.size(); i++){
        double s = cv::contourArea(contours[i]);
        std::map<std::string, double> map;
        map["idx"] = i;
        map["area"] = s;
        area.push_back(map);
    }
    std::sort(area.begin(), area.end(), compare);
    cv::RotatedRect rrt = cv::fitEllipse(contours[area[0]["idx"]]);
    double axis1 = rrt.size.width, axis2 = rrt.size.height;
    double diam = (axis1 + axis2) / 2;
    pixel_length = length / diam;
    cv::Point center = rrt.center;
    cv::circle(coin_pic, center, (int)diam/2, cv::Scalar(0, 0, 255), 3);
    cv::imwrite("../pic/coin.jpg", coin_pic);
    return pixel_length;
}

void EdgeDetector::imfill(cv::Mat &mat) {
    cv::Size ori_size = mat.size();
    cv::imwrite("../pic/fill1.jpg", mat);
    cv::Mat temp = cv::Mat::zeros(ori_size.height+2, ori_size.width+2, mat.type());
    mat.copyTo(temp(cv::Range(1, ori_size.height + 1), cv::Range(1, ori_size.width + 1)));
    cv::floodFill(temp, cv::Point(0, 0), cv::Scalar(255));
    cv::imwrite("../pic/fill2.jpg", temp);
    cv::Mat cut_pic;
    temp(cv::Range(1, ori_size.height + 1), cv::Range(1, ori_size.width + 1)).copyTo(cut_pic);
    mat = mat | (~cut_pic);
    cv::imwrite("../pic/fill3.jpg", mat);
}

bool EdgeDetector::getVarianceMean(cv::Mat &scr, cv::Mat &meansDst, cv::Mat &varianceDst, int winSize) {
    if (!scr.data)  //判断图像是否被正确读取；
    {
        std::cout << "获取方差与均值的函数读入图片有误" << std::endl;
        return false;
    }

    if (winSize % 2 == 0)
    {
        std::cout << "计算局部均值与标准差的窗口大小应该为单数" << std::endl;
        return false;
    }

    cv::Mat copyBorder_yChannels;                        //扩充图像边界；
    int copyBorderSize = (winSize - 1) / 2;
    cv::copyMakeBorder(scr, copyBorder_yChannels, copyBorderSize, copyBorderSize, copyBorderSize, copyBorderSize, cv::BORDER_REFLECT);

    for (int i = (winSize - 1) / 2; i < copyBorder_yChannels.rows - (winSize - 1) / 2; i++)
    {
        for (int j = (winSize - 1) / 2; j < copyBorder_yChannels.cols - (winSize - 1) / 2; j++)
        {

            cv::Mat temp = copyBorder_yChannels(cv::Rect(j - (winSize - 1) / 2, i - (winSize - 1) / 2, winSize, winSize));   //截取扩展后的图像中的一个方块；

            cv::Scalar  mean;
            cv::Scalar  dev;
            meanStdDev(temp, mean, dev);

            varianceDst.at<float>(i - (winSize - 1) / 2, j - (winSize - 1) / 2) = dev.val[0];     ///一一对应赋值；
            meansDst.at<float>(i - (winSize - 1) / 2, j - (winSize - 1) / 2) = mean.val[0];
        }
    }

    return true;

}

bool EdgeDetector::adaptContrastEnhancement(cv::Mat &scr, cv::Mat &dst, int winSize, int maxCg) {
    if (!scr.data)  //判断图像是否被正确读取；
    {
        std::cout << "自适应对比度增强函数读入图片有误" << std::endl;
        return false;
    }

    cv::Mat ycc;                        //转换空间到YCrCb；
    cvtColor(scr, ycc, cv::COLOR_RGB2YCrCb);

    std::vector<cv::Mat> channels(3);        //分离通道；
    split(ycc, channels);


    cv::Mat localMeansMatrix(scr.rows , scr.cols , CV_32FC1);
    cv::Mat localVarianceMatrix(scr.rows , scr.cols , CV_32FC1);

    if (!getVarianceMean(channels[0], localMeansMatrix, localVarianceMatrix, winSize))   //对Y通道进行增强；
    {
        std::cout << "计算图像均值与标准差过程中发生错误" << std::endl;
        return false;
    }

    cv::Mat temp = channels[0].clone();

    cv::Scalar  mean;
    cv::Scalar  dev;
    meanStdDev(temp, mean, dev);

    float meansGlobal = mean.val[0];
    cv::Mat enhanceMatrix(scr.rows, scr.cols, CV_8UC1);

    for (int i = 0; i < scr.rows; i++)            //遍历，对每个点进行自适应调节
    {
        for (int j = 0; j < scr.cols; j++)
        {
            if (localVarianceMatrix.at<float>(i, j) >= 0.01)
            {
                float cg = 0.2*meansGlobal / localVarianceMatrix.at<float>(i, j);
                float cgs = cg > maxCg ? maxCg : cg;
                cgs = cgs < 1 ? 1 : cgs;

                int e = localMeansMatrix.at<float>(i, j) + cgs* (temp.at<uchar>(i, j) - localMeansMatrix.at<float>(i, j));
                if (e > 255){ e = 255; }
                else if (e < 0){ e = 0; }
                enhanceMatrix.at<uchar>(i, j) = e;
            }
            else
            {
                enhanceMatrix.at<uchar>(i, j) = temp.at<uchar>(i, j);
            }
        }

    }

    channels[0] = enhanceMatrix;    //合并通道，转换颜色空间回到RGB
    merge(channels, ycc);

    cvtColor(ycc, dst, cv::COLOR_YCrCb2RGB);
    return true;
}