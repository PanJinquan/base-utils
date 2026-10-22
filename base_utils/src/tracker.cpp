//
// Created by PKing on 2026/9/23.
//

#include "tracker.h"
#include <algorithm>

Tracker::Tracker(int maxLost, int maxSzie, float iouThresh, AlgType algo)
    : maxLost(maxLost), maxSzie(maxSzie), iouThresh(iouThresh), algo(algo),
      byteTracker(30, maxLost, 0.5f, 0.6f, 0.8f, 0.1f, maxSzie) {
}

Tracker::~Tracker() = default;

void Tracker::reset() {
    // 重置跟踪器：清空所有目标、跟踪器与卡尔曼滤波器，重新分配ID
    trackers.clear();
    kalmanFilters.clear();
    obj_tracks.clear();
    nextID = 0;
    byteTracker.reset();
}

float Tracker::iou(const cv::Rect2f &a, const cv::Rect2f &b) {
    // 计算两个矩形框的交并比
    cv::Rect2f inter = a & b;
    float interArea = inter.area();
    if (interArea <= 0.f) return 0.f;
    float unionArea = a.area() + b.area() - interArea;
    return unionArea > 0.f ? interArea / unionArea : 0.f;
}

vector<int> Tracker::greedyMatch(const vector<cv::Rect2f> &boxes, const vector<cv::Rect2f> &trackBoxes, float thresh) {
    // 贪心匹配：每个 box 找 IoU 最大且超过阈值的 track，且每个 track 最多被匹配一次
    vector<int> result(boxes.size(), -1);
    vector<bool> used(trackBoxes.size(), false);
    for (size_t b = 0; b < boxes.size(); b++) {
        float bestIoU = thresh;
        int bestIdx = -1;
        for (size_t t = 0; t < trackBoxes.size(); t++) {
            if (used[t]) continue;
            float v = iou(boxes[b], trackBoxes[t]);
            if (v > bestIoU) {
                bestIoU = v;
                bestIdx = (int) t;
            }
        }
        if (bestIdx >= 0) {
            result[b] = bestIdx;
            used[bestIdx] = true;
        }
    }
    return result;
}

void Tracker::appendTrajectory(Track &track, const cv::Rect2f &box) {
    // 记录目标框中心点作为轨迹点
    cv::Point2f center(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
    track.trajectory.push_back(center);
    // 限制轨迹长度，避免长时间运行内存无限增长
    if ((int) track.trajectory.size() > maxSzie) {
        track.trajectory.erase(track.trajectory.begin());
    }
}

cv::Ptr<cv::legacy::Tracker> Tracker::makeTracker() const {
    // 按 algo 创建对应的单目标跟踪器
    switch (algo) {
        case AlgType::CSRT:
            return cv::legacy::TrackerCSRT::create();
        case AlgType::MOSSE:
            return cv::legacy::TrackerMOSSE::create();
        case AlgType::KCF:
        default:
            return cv::legacy::TrackerKCF::create();
    }
}

cv::KalmanFilter Tracker::makeKalmanFilter(const cv::Rect2f &box) const {
    // 卡尔曼滤波器：8 维状态 [x, y, w, h, vx, vy, vw, vh]，4 维测量 [x, y, w, h]
    cv::KalmanFilter kf(8, 4, 0, CV_32F);
    // 状态转移矩阵：匀速运动模型
    kf.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
    for (int i = 0; i < 4; i++) {
        kf.transitionMatrix.at<float>(i, i + 4) = 1.f;
    }
    // 测量矩阵：只观测位置与尺寸
    kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; i++) {
        kf.measurementMatrix.at<float>(i, i) = 1.f;
    }
    kf.processNoiseCov = cv::Mat::eye(8, 8, CV_32F) * 1e-2f;
    kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-1f;
    kf.errorCovPost = cv::Mat::eye(8, 8, CV_32F);
    // 初始状态：以检测框位置为初始值，速度置 0
    kf.statePost.at<float>(0) = box.x;
    kf.statePost.at<float>(1) = box.y;
    kf.statePost.at<float>(2) = box.width;
    kf.statePost.at<float>(3) = box.height;
    return kf;
}

void Tracker::update(const cv::Mat &image, vector<ImageInfo> &dets) {

    // ByteTrack 模式：直接使用检测框与分数做两次关联(高分/低分)，与 legacy 跟踪器互斥
    if (algo == AlgType::BYTE) {
        vector<cv::Rect2f> detBoxes;
        vector<float> detScores;
        detBoxes.reserve(dets.size());
        detScores.reserve(dets.size());
        for (size_t i = 0; i < dets.size(); i++) {
            detBoxes.push_back(cv::Rect2f(dets[i].x1, dets[i].y1,
                                          dets[i].x2 - dets[i].x1,
                                          dets[i].y2 - dets[i].y1));
            detScores.push_back(dets[i].score);
        }
        vector<Track> tracks;
        byteTracker.update(detBoxes, detScores, tracks);
        for (size_t i = 0; i < dets.size() && i < tracks.size(); i++) {
            dets[i].track = tracks[i];
        }
        return;
    }

    // 空帧：清空所有检测目标的 track 信息后直接返回
    if (image.empty()) {
        for (size_t i = 0; i < dets.size(); i++) {
            dets[i].track = Track();
        }
        return;
    }

    // 辅助函数：从跟踪目标中提取目标框(用于预测和 IoU 匹配)
    auto boxesOf = [](const vector<ImageInfo> &tracks) {
        vector<cv::Rect2f> boxes;
        boxes.reserve(tracks.size());
        for (size_t i = 0; i < tracks.size(); i++) {
            boxes.push_back(cv::Rect2f(tracks[i].x1, tracks[i].y1,
                                       tracks[i].x2 - tracks[i].x1,
                                       tracks[i].y2 - tracks[i].y1));
        }
        return boxes;
    };

    const bool useSort = (algo == AlgType::SORT);

    // 第一步：预测目标在当前帧的位置，作为后续 IoU 匹配的依据。
    if (useSort) {
        // SORT 模式：所有目标均用卡尔曼滤波器预测(开销可忽略)
        for (size_t t = 0; t < obj_tracks.size(); t++) {
            cv::Mat pred = kalmanFilters[t].predict();
            float x = pred.at<float>(0), y = pred.at<float>(1);
            float w = pred.at<float>(2), h = pred.at<float>(3);
            if (w > 0 && h > 0) {
                obj_tracks[t].x1 = x;
                obj_tracks[t].y1 = y;
                obj_tracks[t].x2 = x + w;
                obj_tracks[t].y2 = y + h;
            }
        }
    } else {
        // legacy 模式：仅对丢失中的目标(已有跟踪器)预测位置，可见目标不运行跟踪器
        for (size_t t = 0; t < obj_tracks.size(); t++) {
            if (trackers[t].empty()) continue;
            cv::Rect2d pred;
            if (trackers[t]->update(image, pred) && pred.width > 0 && pred.height > 0) {
                obj_tracks[t].x1 = (float) pred.x;
                obj_tracks[t].y1 = (float) pred.y;
                obj_tracks[t].x2 = (float) (pred.x + pred.width);
                obj_tracks[t].y2 = (float) (pred.y + pred.height);
            }
            // 跟踪器更新失败时保持原框，后续由丢失计数自然删除
        }
    }

    // 第二步：构建当前帧检测框，并与已有目标做 IoU 匹配
    vector<cv::Rect2f> detBoxes;
    detBoxes.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); i++) {
        detBoxes.push_back(cv::Rect2f(dets[i].x1, dets[i].y1,
                                      dets[i].x2 - dets[i].x1,
                                      dets[i].y2 - dets[i].y1));
    }
    vector<int> detToTrack = greedyMatch(detBoxes, boxesOf(obj_tracks), iouThresh);

    // 第三步：匹配到的检测框修正已有目标
    vector<bool> matched(obj_tracks.size(), false);
    for (size_t d = 0; d < dets.size(); d++) {
        int t = detToTrack[d];
        if (t < 0) continue;
        matched[t] = true;
        obj_tracks[t].x1 = detBoxes[d].x;
        obj_tracks[t].y1 = detBoxes[d].y;
        obj_tracks[t].x2 = detBoxes[d].x + detBoxes[d].width;
        obj_tracks[t].y2 = detBoxes[d].y + detBoxes[d].height;
        obj_tracks[t].track.index = (int) d;   // 记录当前帧匹配的检测索引
        obj_tracks[t].track.lost = 0;
        appendTrajectory(obj_tracks[t].track, detBoxes[d]);
        if (useSort) {
            // 用检测框校正卡尔曼滤波器
            cv::Mat meas = (cv::Mat_<float>(4, 1) << detBoxes[d].x, detBoxes[d].y,
                            detBoxes[d].width, detBoxes[d].height);
            kalmanFilters[t].correct(meas);
        } else {
            trackers[t].reset();               // 目标重新可见，不再需要跟踪器
        }
    }

    // 第四步：未被匹配的检测框作为新目标加入跟踪
    for (size_t d = 0; d < dets.size(); d++) {
        if (detToTrack[d] >= 0) continue;
        ImageInfo nt;
        nt.x1 = detBoxes[d].x;
        nt.y1 = detBoxes[d].y;
        nt.x2 = detBoxes[d].x + detBoxes[d].width;
        nt.y2 = detBoxes[d].y + detBoxes[d].height;
        nt.label = dets[d].label;
        nt.score = dets[d].score;
        nt.track.idx = nextID++;
        nt.track.index = (int) d;           // 记录当前帧匹配的检测索引
        appendTrajectory(nt.track, detBoxes[d]);
        obj_tracks.push_back(nt);
        if (useSort) {
            kalmanFilters.push_back(makeKalmanFilter(detBoxes[d]));  // 初始化卡尔曼滤波器
        } else {
            trackers.push_back(cv::Ptr<cv::legacy::Tracker>());      // 可见目标，暂无跟踪器
        }
    }

    // 第五步：未被检测匹配的已有目标视为丢失
    for (size_t t = 0; t < matched.size(); t++) {
        if (matched[t]) continue;
        obj_tracks[t].track.lost++;
        obj_tracks[t].track.index = -1;        // 丢失时清空检测索引
        if (!useSort && trackers[t].empty()) {
            // legacy 模式：首次丢失时用其当前框初始化一个跟踪器
            cv::Ptr<cv::legacy::Tracker> tracker = makeTracker();
            cv::Rect2d box(obj_tracks[t].x1, obj_tracks[t].y1,
                           obj_tracks[t].x2 - obj_tracks[t].x1,
                           obj_tracks[t].y2 - obj_tracks[t].y1);
            tracker->init(image, box);
            trackers[t] = tracker;
        }
        // SORT 模式：无需额外处理，下一帧由卡尔曼继续外推预测
    }

    // 第六步：删除长时间丢失的目标(同步删除其跟踪器/卡尔曼滤波器)
    for (size_t t = 0; t < obj_tracks.size();) {
        if (obj_tracks[t].track.lost > maxLost) {
            obj_tracks.erase(obj_tracks.begin() + t);
            if (useSort) {
                kalmanFilters.erase(kalmanFilters.begin() + t);
            } else {
                trackers.erase(trackers.begin() + t);
            }
        } else {
            t++;
        }
    }

    // 第七步：将跟踪信息写回 ImageInfo(每个检测目标填充其 track 字段)
    for (size_t i = 0; i < dets.size(); i++) {
        dets[i].track = Track();
    }
    for (size_t t = 0; t < obj_tracks.size(); t++) {
        int index = obj_tracks[t].track.index;
        if (index >= 0 && index < (int) dets.size()) {
            dets[index].track = obj_tracks[t].track;
        }
    }
}
