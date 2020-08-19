# import cv2
import numpy as np
import os
def showHistogram(pic):
    pass

def getFilePath(path):
    abs_path = []
    file_list = os.listdir(path)
    for f in file_list:
        abs_path.append(path + "/" + f)
    return abs_path

def main():
    dir_path = "E:/Repositories/IndentationDetection/pic/test/microscope"
    for p in getFilePath(dir_path):
        print(p)


if __name__ == "__main__":
    main()