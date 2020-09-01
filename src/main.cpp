#include <iostream>
#include "EdgeDetector.h"
#include <string>
#include <vector>
#include <io.h>
#include "Calibrator.h"

void getFiles(const std::string & path, std::vector<std::string> & files);
enum operation {
    GEN_CALIBRATOR, 
    CALIBRATE,
    CAL_DIAM_FIT,
    CAL_DIAM_SUBPIXEL,
    TEST_PARAMETERS
};
int main(int argc, char *argv[]) {
    operation op;
    std::string argv1 = argv[1];
    if(argv1 == "gen_calibrator"){
        op = GEN_CALIBRATOR;
    }else if(argv1 == "calibrate"){
        op = CALIBRATE;
    }else if(argv1 == "cal_diam_fit"){
        op = CAL_DIAM_FIT;
    }else if(argv1 == "cal_diam_subpixel"){
        op = CAL_DIAM_SUBPIXEL;
    }else if(argv1 == "test_parameters"){
        op = TEST_PARAMETERS;
    }
    switch (op)
    {
    case GEN_CALIBRATOR:
    {
        int width = std::stoi(argv[2]);
        int height = std::stoi(argv[3]);
        int square_width = std::stoi(argv[4]);
        std::string path = argv[5];
        std::cout << "width:" << width << "\n" 
            << "height:" << height << "\n"
            << "square_width:" << square_width << "\n"
            << "path:" << path << "\n";
        Calibrator::genCalibrator(cv::Size(width, height), square_width, path);
        std::cout << "calibrator is saved in " + path << std::endl;
    }
    break;    
    case CALIBRATE:
    {
        std::string path = argv[2];
        int point_num_row = std::stoi(argv[3]);
        int point_num_col = std::stoi(argv[4]);
        int each_pixel_square = std::stoi(argv[5]);
        double pixel_length = std::stod(argv[6]);
        double result = Calibrator::calibrateMircoscope(cv::imread(path), cv::Size(point_num_row, point_num_col), each_pixel_square, pixel_length);
        std::cout << "calibration result:" << result << "mm/pixel" << std::endl;
    }
        break;
    case CAL_DIAM_FIT:
    {
        std::string read_status = argv[2];
        std::string input_path = argv[3];
        int bin_threshold = std::stoi(argv[4]);
        int pixel_around_threshold = std::stoi(argv[5]);
        int pixel_neighbour = std::stoi(argv[6]);
        double pixel_len = std::stod(argv[7]);
        int min_radius = std::stoi(argv[8]);
        int max_radius = std::stoi(argv[9]);
        EdgeDetector::para = {min_radius, max_radius, 0, 0, 0, 0, pixel_around_threshold, pixel_neighbour, bin_threshold, pixel_len};
        EdgeDetector *edgedetector;
        if(read_status == "dic"){
            std::vector<std::string> file_abs_path;
            getFiles(input_path, file_abs_path);
            std::vector<cv::Mat> input_pics;
            for(auto path : file_abs_path){
                cv::Mat temp = cv::imread(path);
                input_pics.push_back(temp);
            }
            std::cout << "the amount of pictures:" << input_pics.size() << std::endl;

            edgedetector = new EdgeDetector(input_pics);

        } else if(read_status == "file"){
            cv::Mat input_mat3 = cv::imread(input_path);
            edgedetector = new EdgeDetector(input_mat3);
        }
        std::vector<double> radius;
        std::vector<cv::Point> center;
        cv::Mat res = edgedetector->findHoleByBinaryzation(cv::Size(15, 15), 1, radius, center);
        for(int i = 0; i < radius.size(); i++){
            std::cout << "diam is:" << pixel_len * radius[i] * 2 << std::endl;
            std::string text = std::to_string( pixel_len * radius[i] * 2);
            cv::putText(res, text, cv::Point(center[i].x, center[i].y+40), cv::FONT_HERSHEY_COMPLEX, 4,cv::Scalar(0, 0, 255), 4);
        }
        
        cv::imwrite("../pic/process/x.jpg", res);
        std::cin.get();
    }
    break;
    case CAL_DIAM_SUBPIXEL:
    {
        std::string read_status = argv[2];
        std::string input_path = argv[3];
        int bin_threshold = std::stoi(argv[4]);
        int pixel_around_threshold = std::stoi(argv[5]);
        int pixel_neighbour = std::stoi(argv[6]);
        double pixel_len = std::stod(argv[7]);
        int min_radius = std::stoi(argv[8]);
        int max_radius = std::stoi(argv[9]);
        EdgeDetector::para = {min_radius, max_radius, 0, 0, 0, 0, pixel_around_threshold, pixel_neighbour, bin_threshold, pixel_len};
        EdgeDetector *edgedetector;
        if(read_status == "dic"){
            std::vector<std::string> file_abs_path;
            getFiles(input_path, file_abs_path);
            std::vector<cv::Mat> input_pics;
            for(auto path : file_abs_path){
                cv::Mat temp = cv::imread(path);
                input_pics.push_back(temp);
            }
            std::cout << "the amount of pictures:" << input_pics.size() << std::endl;
            edgedetector = new EdgeDetector(input_pics);

        } else if(read_status == "file"){
            cv::Mat input_mat3 = cv::imread(input_path);
            edgedetector = new EdgeDetector(input_mat3);
        }
        std::vector<double> diam;
        std::vector<cv::Point> center;
        cv::Mat res = edgedetector->findHoleSubPixel(cv::Size(15, 15), 1, diam, center);
        for(int i = 0; i < diam.size(); i++){
            std::cout << "diam is:" << pixel_len * diam[i] << std::endl;
            std::string text = std::to_string( pixel_len * diam[i]);
            cv::putText(res, text, cv::Point(center[i].x, center[i].y+40), cv::FONT_HERSHEY_COMPLEX, 4,cv::Scalar(0, 0, 255), 4);
        }
        cv::imwrite("../pic/process/x.jpg", res);
        std::cin.get();
    }
    break;
    case TEST_PARAMETERS:
    {
         
    }
    break;
    default:
        break;
    }
    std::cin.get();
    // cv::Mat calib_pic = cv::imread("../pic/test/calib_camera.JPG");
    // double pixel_len = edgedetector->calibrationByCoin(calib_pic, 36);
    // double pixel_len = 0.028283;
    


}


void getFiles(const std::string & path, std::vector<std::string> & files)
{
	//文件句柄  
	long long hFile = 0;
	//文件信息，_finddata_t需要io.h头文件  
	struct _finddata_t fileinfo;
	std::string p;
	int i = 0;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				//if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					//getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
