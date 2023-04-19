//
// Created by Pan on 2021/1/19.
//

#include "filter/kalman_filter.h"


KalmanFilter::KalmanFilter(int stateNum, int measureNum) {
    mKF = new cv::KalmanFilter(stateNum, measureNum, 0);
    //转移矩阵A
    mKF->transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
    //测量矩阵H
    cv::setIdentity(mKF->measurementMatrix);
    //测量噪声方差矩阵R，取值越小收敛越快
    cv::setIdentity(mKF->measurementNoiseCov, cv::Scalar::all(1e-4));
    //过程(系统)噪声噪声方差矩阵Q
    cv::setIdentity(mKF->processNoiseCov, cv::Scalar::all(1e-6));
    //后验错误估计协方差矩阵P
    cv::setIdentity(mKF->errorCovPost, cv::Scalar::all(1));
    //系统初始状态 x(0)
    //cv::RNG rng;
    //rng.fill(KF->statePost, RNG::UNIFORM, 0, winHeight > winWidth ? winWidth : winHeight);
    //初始测量值x'(0)，因为后面要更新这个值，所以先定义
    mMeasurement = cv::Mat::zeros(measureNum, 1, CV_32F);
}


KalmanFilter::~KalmanFilter(){
    mMeasurement.release();
    delete mKF;
    mKF= nullptr;
}

void KalmanFilter::update(cv::Point pos) {
    if (pos.x > 0 && pos.y > 0) {
        mMeasurement.at<float>(0) = (float) pos.x;
        mMeasurement.at<float>(1) = (float) pos.y;
        //4.update
        mKF->correct(mMeasurement);
    }
}


cv::Point KalmanFilter::predict() {
    //返回的是下一时刻的状态值KF.statePost (k+1)
    cv::Mat prediction = mKF->predict();
    //预测值(x',y')
    cv::Point dst = cv::Point(prediction.at<float>(0), prediction.at<float>(1));
    return dst;
}

