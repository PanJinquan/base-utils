//
// Created by Pan on 2021/1/21.
//

#ifndef DETECTOR_POITFLOW_H
#define DETECTOR_POITFLOW_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "mean_filter.h"
#include "kalman_filter.h"

using namespace std;

class TrackingFlow {
public:
    /***
     *
     * @param num_points
     */
    TrackingFlow(int num_points) {
        mNumPoints = num_points;
        init();
    }

    /***
     *
     */
    ~TrackingFlow() {
        for (int i = 0; i < this->mNumPoints; ++i) {
            delete mFilter.at(i);
            mFilter.at(i) = nullptr;
        }
        vector<MovingMeanFilter *>().swap(mFilter);
        //vector<KalmanFilter *>().swap(mFilter);
    }

private:
    /***
     *
     */
    void init() {
        for (int i = 0; i < this->mNumPoints; ++i) {
            mFilter.push_back(new MovingMeanFilter());
            // mFilter.push_back(KalmanFilter());
        }
    }

    /***
     *
     * @param points
     */
    void update(vector<cv::Point> &points) {
        for (int i = 0; i < this->mNumPoints; ++i) {
            mFilter.at(i)->update(points.at(i));
        }
    }

    /***
     *
     * @param points
     */
    void predict(vector<cv::Point> &points) {
        for (int i = 0; i < this->mNumPoints; ++i) {
            points.at(i) = mFilter.at(i)->predict();
        }
    }

private:
    int mNumPoints;
    std::vector<MovingMeanFilter *> mFilter;
    //std::vector<KalmanFilter*> mFilter;
};

#endif //DETECTOR_POITFLOW_H
