import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def showHistogram(pic):
    plt.plot([75]*20000, [i for i in range(20000)])
    plt.plot([95]*20000, [i for i in range(20000)])
    plt.hist(pic.ravel(), 256)
    plt.show()

def getFilePath(path):
    abs_path = []
    file_list = os.listdir(path)
    for f in file_list:
        abs_path.append(path + "/" + f)
    return abs_path

def composePicture(pics):
    print(pics[0].shape[0:2])
    ret_mat = np.zeros(pics[0].shape[0:2])
    for pic in pics:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        print(pic.shape)
        ret_mat = cv2.accumulate(pic, ret_mat)
    ret_mat = ret_mat / len(pics)
    return ret_mat

def main():
    pic_list = []
    dir_path = "D:/Repositories/IndentationDetection/py/0821result/6"
    for p in getFilePath(dir_path):
        pic_list.append(cv2.imread(p))
        print(p)
    temp_mat = composePicture(pic_list)
    showHistogram(temp_mat)
    # bin_mat = cv2.threshold(temp_mat, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("x", pic_list[0])
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        return


if __name__ == "__main__":
    main()