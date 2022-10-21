#include<opencv2/opencv.hpp>
#include<string>
#include "debug.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include "filter/kalman_filter.h"
#include "filter/mean_filter.h"

using namespace std;

// 鼠标的坐标位置
cv::Point mousePosition = cv::Point(0, 0);

//mouse event callback
void static mouseEvent(int event, int x, int y, int flags, void *param) {
    if (event == cv::EVENT_MOUSEMOVE) {
        mousePosition = cv::Point(x, y);
    }
}

void test() {
    cv::namedWindow("kalman");
    cv::setMouseCallback("kalman", mouseEvent);
    //卡尔曼滤波
    KalmanFilter filter = KalmanFilter(4, 2);
    // 加权平均滤波
    //MovingMeanFilter filter = MovingMeanFilter(5, 0.6);
    cv::Mat image(800, 800, CV_8UC3, cv::Scalar(0));
    cv::Point cm = cv::Point(-1, -1);
    cv::Point cp = cv::Point(-1, -1);
    cv::Point lm = cv::Point(-1, -1);
    cv::Point lp = cv::Point(-1, -1);
    cv::Point pred;
    while (true) {
        //update measurement
        filter.update(mousePosition);
        //prediction
        pred = filter.predict();   //预测值(x',y')
        cout << "curr_point:" << mousePosition << endl;
        cout << "pred_point:" << pred << endl;

        if (lm.x < 0 && lm.y < 0) {
            lm = mousePosition;
        }
        if (lp.x < 0 && lp.y < 0) {
            lp = pred;
        }
        //draw
        cm = mousePosition;
        cp = pred;
        cv::line(image, lm, cm, cv::Scalar(0, 200, 0));//绘制测量值轨迹（绿色）
        cv::line(image, lp, cp, cv::Scalar(0, 0, 200));//绘制预测值轨迹（红色）
        lm = cm;
        lp = cp;

        cv::imshow("kalman", image);
        int key = cv::waitKey(100);
        if (key == 27) {//esc
            break;
        } else if (key == 'c' || key == 'C') {
            image = cv::Mat(800, 800, CV_8UC3, cv::Scalar(0));
        }
        cout << "========================" << endl;
    }
}

int main(void) {
    test();

}
