#include <iostream>
#include "EdgeDetector.h"
#include <string>
#include <vector>
#include <io.h>

void getFiles(const std::string & path, std::vector<std::string> & files);
int main(int argc, char *argv[]) {
    std::string read_status = argv[3];
    EdgeDetector *edgedetector;
    if(read_status == "dic"){
        std::string dic_name = argv[1];
        std::string dic_path = "../pic/test/" + dic_name;
        std::vector<std::string> file_abs_path;
        getFiles(dic_path, file_abs_path);
        std::vector<cv::Mat> input_pics;
        for(auto path : file_abs_path){
            cv::Mat temp = cv::imread(path);
            input_pics.push_back(temp);
        }
        std::cout << "the amount of pictures:" << input_pics.size() << std::endl;
        edgedetector = new EdgeDetector(input_pics, std::stoi(argv[2]));

    } else if(read_status == "file"){
        std::string spci_name = argv[1];
        std::string pic_name = "../pic/test/" + spci_name + ".jpg";
        cv::Mat input_mat3 = cv::imread(pic_name);
        edgedetector = new EdgeDetector(input_mat3, std::stoi(argv[2]));
    }
    cv::Mat calib_pic = cv::imread("../pic/test/calib_camera.JPG");
    // double pixel_len = edgedetector->calibrationByCoin(calib_pic, 36);
    double pixel_len = 0.028283;
    edgedetector->setPixelLength(pixel_len);
    std::cout << "pixel_len:" << pixel_len << std::endl;
    std::cin.get();
    std::vector<double> radius;
    std::vector<cv::Point> center;
    // cv::Mat res = edgedetector->findHoleByBinaryzation(cv::Size(15, 15), 2, radius, center);
    cv::Mat res = edgedetector->findHoleSubPixel(cv::Size(15, 15), 2, radius, center);
    std::cout << pixel_len * radius[0]<< " " << pixel_len * radius[1]<< std::endl;
    for(int i = 0; i < 2; i++){
        std::string text = std::to_string( pixel_len * radius[i]);
        cv::putText(res, text, cv::Point(center[i].x, center[i].y+40), cv::FONT_HERSHEY_COMPLEX, 4,cv::Scalar(0, 0, 255), 4);
    }
    cv::imwrite("../pic/process/x.jpg", res);
    std::cin.get();


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
