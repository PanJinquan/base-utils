//
// Created by Pan on 2021/1/21.
//

#include <filter/mean_filter.h>


MovingMeanFilter::MovingMeanFilter(int win_size, float decay) {
    this->mWinSize = win_size;
    vector<float> WeightDecay = get_weight_decay(win_size, decay);
    // fix a bug: 必须clone()，否则会被释放掉
    this->mWeightDecay = cv::Mat(WeightDecay).reshape(1, 1).clone();
}

MovingMeanFilter::~MovingMeanFilter() {
    mQueue.clear();
    vector<cv::Point>().swap(mQueue);
    mWeightDecay.release();
    //mWeightDecay.clear();
    //vector<float>().swap(mWeightDecay);
}


void MovingMeanFilter::update(cv::Point pos) {
    if (mQueue.size() >= this->mWinSize) {
        mQueue.erase(mQueue.begin());
    }
    if (pos.x > 0 && pos.y > 0) {
        mQueue.push_back(pos);
    }
    // 进行填充
    int size = this->mWinSize - mQueue.size();
    //for (int i = 0; i < (this->mWinSize - mQueue.size()); ++i) {// fix  a Bug
    if (size > 0 && pos.x > 0 && pos.y > 0) {
        for (int i = 0; i < size; ++i) {
            mQueue.push_back(pos);
        }
    }
}


cv::Point MovingMeanFilter::predict() {
    cv::Point dst;
    if (mQueue.size() > 0) {
        dst = filter();
    } else {
        dst = cv::Point(0, 0);
    }
    return dst;
}


cv::Point MovingMeanFilter::filter() {
    cv::Mat data = cv::Mat(mQueue).reshape(1, mQueue.size());
    data.convertTo(data, CV_32FC1, 1.0);
    cv::Mat out = this->mWeightDecay * data; //矩阵乘法:w(1,num)*data(num,2)
    cv::Point dst = cv::Point(out.at<float>(0), out.at<float>(1));
    return dst;
}


vector<float> MovingMeanFilter::get_weight_decay(int n, float decay) {
    float r = decay / (1.0 - decay);
    float sum = 0.f;
    //计算衰减权重
    vector<float> w = {1.0};
    for (int i = 1; i < n; ++i) {
        // fix bug: accumulate的init必须设置为输入同一类型，否则计算精度损失
        sum = accumulate(w.begin(), w.end(), 0.f);
        w.push_back(sum * r);
    }
    // 进行归一化
    sum = accumulate(w.begin(), w.end(), 0.f);
    for (int i = 0; i < w.size(); ++i) {
        w.at(i) = w.at(i) / sum;
    }
    return w;
}