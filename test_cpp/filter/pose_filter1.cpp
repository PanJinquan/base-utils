//
// Created by 390737991@qq.com on 2020/6/3.
//


#include "pose_filter1.h"

PoseFilter::PoseFilter(vector<int> filter_id, int win_size, float decay) {
    this->filter_id = filter_id;
    for (int i = 0; i < filter_id.size(); ++i) {
        FILTER *mFilter = new FILTER(win_size, decay);
        //FILTER *mFilter = new FILTER();
        mFilters.push_back(mFilter);
    }

}


PoseFilter::~PoseFilter() {
    for (int i = 0; i < this->mFilters.size(); ++i) {
        if (this->mFilters[i] != nullptr) {
            delete this->mFilters[i];
            this->mFilters[i] = nullptr;
        }
    }
}

void PoseFilter::filter(ObjectInfo &obj) {
    for (int i = 0; i < this->mFilters.size(); ++i) {
        int id = this->filter_id[i];
        cv::Point pos(obj.keypoints.at(id).point.x, obj.keypoints.at(id).point.y);
        if (pos.x > 0 && pos.y > 0) {
            //update
            this->mFilters[i]->update(pos);
            //prediction
            obj.keypoints.at(id).point = this->mFilters[i]->predict();   //预测值(x',y')
        }
    }
}
