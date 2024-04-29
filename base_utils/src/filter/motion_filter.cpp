//
// Created by Pan on 2021/1/21.
//

#include <filter/motion_filter.h>


MotionFilter::MotionFilter(int win_size, float decay) {
    this->mWinSize = win_size;
    this->mDecay = decay;
    this->curr = {-1, -1};
    this->last = {-1, -1};
}

MotionFilter::~MotionFilter() {

}


void MotionFilter::update(cv::Point pos) {
    if (pos.x >= 0 && pos.y >= 0) {
        this->curr = pos;
        if (this->last.x < 0 or this->last.y < 0) {
            this->last = pos;
        }
    }
}


cv::Point MotionFilter::predict() {
    cv::Point dst;
    if (this->curr.x >= 0 and this->curr.x >= 0) {
        dst = filter();
    } else {
        dst = cv::Point(0, 0);
    }
    return dst;
}


cv::Point MotionFilter::filter() {
    cv::Point dst = this->curr * this->mDecay + (1.0 - this->mDecay) * this->last;
    this->last = dst;
    return dst;
}

