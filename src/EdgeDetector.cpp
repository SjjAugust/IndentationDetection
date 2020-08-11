//
// Created by ShiJJ on 2020/7/27.
//

#include "EdgeDetector.h"

EdgeDetector::EdgeDetector(const cv::Mat& input_pic, int bin_threshold)
: BINARYZATION_THRESHOLD(bin_threshold){
    ori_pic_ = input_pic.clone();
}



void EdgeDetector::preProcess(const cv::Size& gauss_kernel_size) {
    dst_pic_ = ori_pic_.clone();
//    adaptContrastEnhancement(dst_pic_, dst_pic_, 15, 10);
    cv::imwrite("../pic/enhance.jpg", dst_pic_);
    cv::cvtColor(dst_pic_, dst_pic_, cv::COLOR_BGR2GRAY);
//    cv::GaussianBlur(dst_pic_, dst_pic_, gauss_kernel_size, 0, 0);
    cv::medianBlur(dst_pic_, dst_pic_, 5);
}

cv::Mat EdgeDetector::detectEdge() {
//    cv::adaptiveThreshold(dst_pic_, dst_pic_, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    cv::threshold(dst_pic_, dst_pic_, BINARYZATION_THRESHOLD, 255, cv::THRESH_BINARY);
    cv::imwrite("../pic/step1.jpg", dst_pic_);
    cv::medianBlur(dst_pic_, dst_pic_, 9);
    cv::imwrite("../pic/step2.jpg", dst_pic_);
    cv::bitwise_not(dst_pic_, dst_pic_);
    imfill(dst_pic_);
    cv::imwrite("../pic/step3.jpg", dst_pic_);
    // cv::namedWindow("x", 0);
    // cv::imshow("x", dst_pic_);
    // cv::waitKey(0);
    return dst_pic_.clone();
}

const double EdgeDetector::calDistance(const cv::Point& p1, const cv::Point& p2) {
    return sqrt(pow((p2.y - p1.y), 2) + pow((p2.x - p1.x), 2));
}

const double EdgeDetector::calAngle(const cv::Point &p1, const cv::Point &p2) {
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


cv::Mat EdgeDetector::findHoleByBinaryzation(const cv::Size &gauss_kernel_size, int hole_num, std::vector<double>& radius, std::vector<cv::Point>& center_vec) {
    //预处理，灰度化，中值滤波
    preProcess(gauss_kernel_size);
    //二值化，填充孔洞
    cv::Mat bin_mat = detectEdge();
    //找到所有轮廓
    std::vector<std::vector<cv::Point>> all_contours_vec;
    cv::findContours(bin_mat, all_contours_vec, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    cv::Mat canvas = cv::Mat::zeros(bin_mat.size(), bin_mat.type());
    cv::Mat ret = ori_pic_.clone();
    cv::Mat bin_canvas = bin_mat.clone();
    cv::cvtColor(bin_canvas, bin_canvas, cv::COLOR_GRAY2BGR);
    //对轮廓面积进行滤波，
    std::vector<std::map<std::string, double>> area;
    for(int i = 0; i < all_contours_vec.size(); i++){
        double s = cv::contourArea(all_contours_vec[i]);
        if(s > MIN_AREA && s < MAX_AREA){
            std::map<std::string, double> map;
            map["idx"] = i;
            map["area"] = s;
            area.push_back(map);
        } else {
            cv::drawContours(bin_mat, all_contours_vec, i, cv::Scalar(0), cv::FILLED);
        }

    }
    cv::imwrite("../pic/step4.jpg", bin_mat);
    //按照面积从大到小排序
    std::sort(area.begin(), area.end(), compare);
    //对周长进行滤波
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
        /*
        cv::RotatedRect rrt = cv::fitEllipse(all_contours_vec[area[i]["idx"]]);  
        double axis1 = rrt.size.width, axis2 = rrt.size.height;
        double r = (axis1 + axis2) / 4;
        cv::Point center = rrt.center;*/
        cv::Vec3d circle_info = getCircleByRANSAC(all_contours_vec[area[i]["idx"]], 5000, 0.95, bin_mat.size());
        cv::Point center(circle_info[0], circle_info[1]);
        double r = circle_info[2];

        cv::Mat canvas_temp = cv::Mat::zeros(bin_mat.size(), bin_mat.type());
        std::vector<cv::Point> aft_contour;
        cv::circle(canvas_temp, center, r, cv::Scalar(255), 1);
        for(int i = 0; i < canvas_temp.rows; i++){
            for(int j = 0; j < canvas_temp.cols; j++){
                if(canvas_temp.at<uchar>(i, j) == 255){
                    aft_contour.emplace_back(cv::Point(j, i));
                }
            }
        }
        Goodness goodness = calGoodnessOfFit(all_contours_vec[area[i]["idx"]], aft_contour);
        std::cout << "sum distance:" << goodness.sum_distance << std::endl << "rate:" << goodness.rate << std::endl;

        cv::circle(canvas, center, (int)r, cv::Scalar(255, 0, 0), 3);
        cv::circle(bin_canvas, center, (int)r, cv::Scalar(255, 0, 0), 3);
        cv::circle(ret, center, (int)r, cv::Scalar(0, 0, 255), 3);
//        cv::ellipse(ret, rrt, cv::Scalar(0 ,0 ,255), 1);
        std::cout << "hole" << i << "'s radius: " << r << std::endl;
        radius.push_back(r);
        center_vec.push_back(center);
    }
    cv::imwrite("../pic/step5.jpg", canvas);
    cv::imwrite("../pic/step3_1.jpg", bin_canvas);
    return ret;
}

bool EdgeDetector::compare(std::map<std::string, double> map1, std::map<std::string, double> map2){
    return map1["area"] > map2["area"];
}

double EdgeDetector::calibrationByCoin(const cv::Mat &coin_pic, double length) {
    if(coin_pic.empty()){
        std::cout << "图像读取错误" << std::endl;
        return -1;
    }
    double pixel_length = 0;
    cv::Mat pic = coin_pic.clone();
    cv::cvtColor(pic, pic, cv::COLOR_BGR2GRAY);
    cv::medianBlur(pic, pic, 9);
    cv::threshold(pic, pic, 130, 255, cv::THRESH_BINARY);
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
    // cv::RotatedRect rrt = cv::fitEllipse(contours[area[0]["idx"]]);
    // double axis1 = rrt.size.width, axis2 = rrt.size.height;
    // double diam = (axis1 + axis2) / 2;
    cv::Vec3d circle_info = getCircleByRANSAC(contours[area[0]["idx"]], 500, 0.95, coin_pic.size());
    double diam = circle_info[2] * 2;
    pixel_length = length / diam;
    cv::Point center(circle_info[0], circle_info[1]);
    cv::circle(coin_pic, center, (int)diam/2, cv::Scalar(0, 0, 255), 3);
    cv::imwrite("../pic/coin.jpg", coin_pic);
    this->pixel_length = pixel_length;
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

EdgeDetector::Goodness EdgeDetector::calGoodnessOfFit(const std::vector<cv::Point> &ori_contour, const std::vector<cv::Point> &aft_contour){
    double sum = 0;
    int count = 0;
    for(auto p : aft_contour){
        double min_distance = INT64_MAX;
        for(auto ori_p : ori_contour){
            double d = calDistance(p, ori_p);
            if(d < min_distance){
                min_distance = d;
            } 
        }
        if(min_distance == INT64_MAX){
            std::cout << "未找到最小值??" << std::endl;
            sum += 0;
        } else{
                sum += min_distance;
        }
        if(min_distance < 3) {
            count++;
        }
    }
    double rate;
    if(aft_contour.size() == 0){
        rate = 0;
    } else {
        rate = (double)count / aft_contour.size();
    }
    return {sum, rate};
}

EdgeDetector::Goodness EdgeDetector::calGoodnessOfFit(const std::vector<cv::Point> &ori_contour, const cv::Vec3d &circle_info, const cv::Size &canvas_size){
    cv::Mat canvas_temp = cv::Mat::zeros(canvas_size, CV_8UC1);
        std::vector<cv::Point> aft_contour;
        cv::Point center(circle_info[0], circle_info[1]);
        int r = circle_info[2];
        cv::circle(canvas_temp, center, r, cv::Scalar(255), 1);
        for(int i = 0; i < canvas_temp.rows; i++){
            for(int j = 0; j < canvas_temp.cols; j++){
                if(canvas_temp.at<uchar>(i, j) == 255){
                    aft_contour.emplace_back(cv::Point(j, i));
                }
            }
        }
        return calGoodnessOfFit(ori_contour, aft_contour);
}

cv::Vec3d EdgeDetector::calCircleByThreePoints(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3){
    double A = p1.x * (p2.y - p3.y) - p1.y * (p2.x - p3.x) + p2.x * p3.y - p3.x * p2.y;
    double B = (pow(p1.x, 2) + pow(p1.y, 2)) * (p3.y - p2.y) + (pow(p2.x, 2) + pow(p2.y, 2)) * (p1.y - p3.y) 
    + (pow(p3.x, 2) + pow(p3.y, 2)) * (p2.y - p1.y);
    double C = (pow(p1.x, 2) + pow(p1.y, 2)) * (p2.x - p3.x) + (pow(p2.x, 2) + pow(p2.y, 2)) * (p3.x - p1.x) 
    + (pow(p3.x, 2) + pow(p3.y, 2)) * (p1.x - p2.x);
    double D = (pow(p1.x, 2) + pow(p1.y, 2)) * (p3.x * p2.y - p2.x * p3.y) + (pow(p2.x, 2) + pow(p2.y, 2)) * (p1.x * p3.y - p3.x * p1.y) 
    + (pow(p3.x, 2) + pow(p3.y, 2)) * (p2.x * p1.y - p1.x * p2.y);

    double x = -(B / (2 * A));
    double y = -(C / (2 * A));
    double r;
    if(A != 0){
        r = sqrt((pow(B, 2) + pow(C, 2) - 4 * A * D) / (4 * pow(A, 2)));
    } else {
        std::cout << "A is wrong" << std::endl;
        r = 1000;
    }
    

    return {x, y, r};
}

cv::Vec3d EdgeDetector::getCircleByRANSAC(const std::vector<cv::Point> &contour, int cycle_num, double threshold, const cv::Size &canvas_size){
    int size = contour.size();
    int count = 0;
    struct best
    {
        cv::Vec3d circle;
        double rate = 0;
    };
    
    best best_circle_info;
    while (count++ < cycle_num){ 
        std::vector<int> point_choose;
        for(int i = 0; i < 3; i++){
            int index = rand() % size;
            for(int j = 0; j < point_choose.size(); j++){
                if(index == point_choose[j]){
                    index = rand() % size;
                    j = 0;
                }
            }
            point_choose.push_back(index);
        }
        cv::Vec3d circle_info = calCircleByThreePoints(contour[point_choose[0]], contour[point_choose[1]], contour[point_choose[2]]);
        bool step_to_next = false;
        for(auto p : contour){
            double d = calDistance(cv::Point(circle_info[0], circle_info[1]), p);
            if(d <= 3.0){
                step_to_next = true;
            }
        }
        if(step_to_next){
            continue;
        }
        Goodness goodness = calGoodnessOfFit(contour, circle_info, canvas_size);
        std::cout << "the " << count << " turn, rate:" << goodness.rate << std::endl; 
        if(goodness.rate == 1){
            std::cout << circle_info[0] << " " << circle_info[1] << " " << circle_info[2] << std::endl;
            for(auto p : contour){
                double d = calDistance(cv::Point(circle_info[0], circle_info[1]), p);
                std::cout << d << std::endl;
            }
        }
        if(goodness.rate > best_circle_info.rate){
            best_circle_info.circle = circle_info;
            best_circle_info.rate = goodness.rate;
        }
        if(goodness.rate > threshold){
            break;
        }
    }
    std::cin.get();
    return best_circle_info.circle;
    
}

void EdgeDetector::setPixelLength(double pix){
    this->pixel_length = pix;
}