//
// Created by PKing on 2026/10/12.
//

#include "bytetrack.h"
#include <algorithm>
#include <cmath>

// ============================================================================
// KalmanFilterXYAH
// ============================================================================

KalmanFilterXYAH::KalmanFilterXYAH() {
    stdWeightPosition = 1.0f / 20.0f;
    stdWeightVelocity = 1.0f / 160.0f;
    // 状态转移矩阵：匀速运动模型
    motionMat = cv::Mat::eye(8, 8, CV_32F);
    for (int i = 0; i < 4; i++) {
        motionMat.at<float>(i, i + 4) = 1.f;
    }
    // 观测矩阵：只观测 [cx, cy, a, h]
    updateMat = cv::Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; i++) {
        updateMat.at<float>(i, i) = 1.f;
    }
}

cv::Mat KalmanFilterXYAH::boxToXyah(const cv::Rect2f &box) {
    float h = std::max(box.height, 1e-6f);
    float w = std::max(box.width, 1e-6f);
    float cx = box.x + w * 0.5f;
    float cy = box.y + h * 0.5f;
    float a = w / h;
    return (cv::Mat_<float>(4, 1) << cx, cy, a, h);
}

cv::Rect2f KalmanFilterXYAH::xyahToBox(const cv::Mat &xyah) {
    float cx = xyah.at<float>(0);
    float cy = xyah.at<float>(1);
    float a = xyah.at<float>(2);
    float h = xyah.at<float>(3);
    float w = a * h;
    return cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
}

cv::Rect2f KalmanFilterXYAH::toBox(const cv::Mat &mean) {
    return xyahToBox(mean);
}

void KalmanFilterXYAH::initiate(const cv::Rect2f &box, cv::Mat &mean, cv::Mat &covariance) const {
    cv::Mat measurement = boxToXyah(box);  // 4x1
    float h = measurement.at<float>(3);
    // 初始状态：[观测值, 速度0]
    mean = cv::Mat::zeros(8, 1, CV_32F);
    for (int i = 0; i < 4; i++) {
        mean.at<float>(i) = measurement.at<float>(i);
    }
    // 初始协方差：位置噪声与速度噪声按目标高度缩放
    float stds[8] = {
        2.f * stdWeightPosition * h,
        2.f * stdWeightPosition * h,
        1e-2f,
        2.f * stdWeightPosition * h,
        10.f * stdWeightVelocity * h,
        10.f * stdWeightVelocity * h,
        1e-5f,
        10.f * stdWeightVelocity * h
    };
    covariance = cv::Mat::zeros(8, 8, CV_32F);
    for (int i = 0; i < 8; i++) {
        covariance.at<float>(i, i) = stds[i] * stds[i];
    }
}

void KalmanFilterXYAH::predict(cv::Mat &mean, cv::Mat &covariance) const {
    float h = mean.at<float>(3);
    float stdPos[4] = {stdWeightPosition * h, stdWeightPosition * h, 1e-2f, stdWeightPosition * h};
    float stdVel[4] = {stdWeightVelocity * h, stdWeightVelocity * h, 1e-5f, stdWeightVelocity * h};
    cv::Mat motionCov = cv::Mat::zeros(8, 8, CV_32F);
    for (int i = 0; i < 4; i++) {
        motionCov.at<float>(i, i) = stdPos[i] * stdPos[i];
        motionCov.at<float>(i + 4, i + 4) = stdVel[i] * stdVel[i];
    }
    mean = motionMat * mean;
    covariance = motionMat * covariance * motionMat.t() + motionCov;
}

void KalmanFilterXYAH::update(cv::Mat &mean, cv::Mat &covariance, const cv::Rect2f &box) const {
    cv::Mat measurement = boxToXyah(box);  // 4x1
    float h = mean.at<float>(3);
    // 观测噪声：宽高比项噪声更大
    float stdPos[4] = {stdWeightPosition * h, stdWeightPosition * h, 1e-1f, stdWeightPosition * h};
    cv::Mat innovationCov = cv::Mat::zeros(4, 4, CV_32F);
    for (int i = 0; i < 4; i++) {
        innovationCov.at<float>(i, i) = stdPos[i] * stdPos[i];
    }
    // 投影到观测空间
    cv::Mat projectedMean = updateMat * mean;                                  // 4x1
    cv::Mat projectedCov = updateMat * covariance * updateMat.t() + innovationCov;  // 4x4
    // 卡尔曼增益 K = P H^T S^-1
    cv::Mat pHt = covariance * updateMat.t();  // 8x4
    cv::Mat sInv;
    cv::invert(projectedCov, sInv, cv::DECOMP_SVD);
    cv::Mat gain = pHt * sInv;                 // 8x4
    // 状态校正
    cv::Mat innovation = measurement - projectedMean;  // 4x1
    mean = mean + gain * innovation;
    covariance = covariance - gain * projectedCov * gain.t();
}

// ============================================================================
// STrack
// ============================================================================

void STrack::appendTrajectory(const cv::Rect2f &det, int maxTrajLen) {
    cv::Point2f center(det.x + det.width * 0.5f, det.y + det.height * 0.5f);
    trajectory.push_back(center);
    if ((int) trajectory.size() > maxTrajLen) {
        trajectory.erase(trajectory.begin());
    }
}

void STrack::activate(const KalmanFilterXYAH &kf, const cv::Rect2f &det, float detScore,
                      int frame, int id, int maxTrajLen) {
    box = det;
    score = detScore;
    trackId = id;
    state = Tracked;
    isActivated = true;
    frameId = frame;
    startFrame = frame;
    trackletLen = 0;
    detIndex = -1;
    trajectory.clear();
    kf.initiate(det, mean, covariance);
    appendTrajectory(det, maxTrajLen);
}

void STrack::reActivate(const KalmanFilterXYAH &kf, const cv::Rect2f &det, float detScore,
                        int frame, int maxTrajLen) {
    kf.update(mean, covariance, det);
    box = det;
    score = detScore;
    trackletLen = 0;
    state = Tracked;
    isActivated = true;
    frameId = frame;
    appendTrajectory(det, maxTrajLen);
}

void STrack::updateState(const KalmanFilterXYAH &kf, const cv::Rect2f &det, float detScore,
                         int frame, int maxTrajLen) {
    frameId = frame;
    trackletLen++;
    kf.update(mean, covariance, det);
    box = det;
    score = detScore;
    state = Tracked;
    isActivated = true;
    appendTrajectory(det, maxTrajLen);
}

void STrack::predict(const KalmanFilterXYAH &kf) {
    cv::Mat meanState = mean.clone();
    // 丢失轨迹不再外推高度方向速度，避免过度漂移
    if (state != Tracked) {
        meanState.at<float>(7) = 0.f;
    }
    kf.predict(meanState, covariance);
    mean = meanState;
    box = KalmanFilterXYAH::toBox(mean);
    detIndex = -1;  // 每帧开始时清空上一帧的检测索引
}

void STrack::markLost() {
    state = Lost;
}

void STrack::markRemoved() {
    state = Removed;
}

// ============================================================================
// 匈牙利算法(最小代价指派，输入为方阵)
// ============================================================================
static void hungarian(const vector<vector<float>> &a, vector<int> &assignment) {
    int n = (int) a.size();
    assignment.assign(n, -1);
    if (n == 0) return;
    const float INF = 1e18f;
    vector<float> u(n + 1, 0.f), v(n + 1, 0.f);
    vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<float> minv(n + 1, INF);
        vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0], j1 = 0;
            float delta = INF;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    float cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    for (int j = 1; j <= n; ++j) {
        if (p[j] > 0) {
            assignment[p[j] - 1] = j - 1;
        }
    }
}

// ============================================================================
// BYTETracker
// ============================================================================

BYTETracker::BYTETracker(int frameRate, int trackBuffer, float trackThresh, float highThresh,
                         float matchThresh, float lowThresh, int maxTrajLen)
    : maxTrajLen(maxTrajLen), trackThresh(trackThresh), highThresh(highThresh),
      matchThresh(matchThresh), lowThresh(lowThresh) {
    // 按30fps折算轨迹最大丢失帧数
    int rate = std::max(frameRate, 1);
    maxTimeLost = std::max(1, (int) (rate / 30.0 * trackBuffer));
}

BYTETracker::~BYTETracker() = default;

void BYTETracker::reset() {
    trackedStracks.clear();
    lostStracks.clear();
    removedStracks.clear();
    frameId = 0;
    nextId = 0;
}

float BYTETracker::iou(const cv::Rect2f &a, const cv::Rect2f &b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    float uni = a.area() + b.area() - inter;
    return uni > 1e-6f ? inter / uni : 0.f;
}

void BYTETracker::iouDistance(const vector<STrack *> &tracks, const vector<cv::Rect2f> &dets,
                              vector<vector<float>> &cost) {
    cost.assign(tracks.size(), vector<float>(dets.size(), 0.f));
    for (size_t i = 0; i < tracks.size(); i++) {
        for (size_t j = 0; j < dets.size(); j++) {
            cost[i][j] = 1.f - iou(tracks[i]->box, dets[j]);
        }
    }
}

void BYTETracker::fuseScore(vector<vector<float>> &cost, const vector<float> &scores) {
    for (size_t i = 0; i < cost.size(); i++) {
        for (size_t j = 0; j < cost[i].size() && j < scores.size(); j++) {
            // 相似度加权后再转回代价
            float sim = (1.f - cost[i][j]) * scores[j];
            cost[i][j] = 1.f - sim;
        }
    }
}

void BYTETracker::linearAssignment(const vector<vector<float>> &cost, float thresh,
                                   vector<pair<int, int>> &matches,
                                   vector<int> &uTracks, vector<int> &uDets) {
    matches.clear();
    uTracks.clear();
    uDets.clear();
    int nTracks = (int) cost.size();
    int nDets = nTracks > 0 ? (int) cost[0].size() : 0;
    if (nTracks == 0 || nDets == 0) {
        for (int i = 0; i < nTracks; i++) uTracks.push_back(i);
        for (int j = 0; j < nDets; j++) uDets.push_back(j);
        return;
    }
    // 通过补齐到方阵来支持矩形代价矩阵，补位代价取大值以保证优先使用真实匹配
    const float LARGE = 1e6f;
    int n = std::max(nTracks, nDets);
    vector<vector<float>> square(n, vector<float>(n, LARGE));
    for (int i = 0; i < nTracks; i++) {
        for (int j = 0; j < nDets; j++) {
            square[i][j] = cost[i][j];
        }
    }
    vector<int> assign;
    hungarian(square, assign);

    vector<bool> trackUsed(nTracks, false), detUsed(nDets, false);
    for (int i = 0; i < nTracks; i++) {
        int j = assign[i];
        if (j >= 0 && j < nDets && cost[i][j] <= thresh) {
            matches.push_back({i, j});
            trackUsed[i] = true;
            detUsed[j] = true;
        }
    }
    for (int i = 0; i < nTracks; i++) {
        if (!trackUsed[i]) uTracks.push_back(i);
    }
    for (int j = 0; j < nDets; j++) {
        if (!detUsed[j]) uDets.push_back(j);
    }
}

void BYTETracker::removeDuplicates(vector<STrack> &tracked, vector<STrack> &lost) {
    if (tracked.empty() || lost.empty()) return;
    vector<bool> removeTracked(tracked.size(), false), removeLost(lost.size(), false);
    for (size_t i = 0; i < tracked.size(); i++) {
        for (size_t j = 0; j < lost.size(); j++) {
            if (iou(tracked[i].box, lost[j].box) > 0.85f) {
                // 重叠度过高视为同一目标，保留存在时间更长的轨迹
                int tDur = tracked[i].frameId - tracked[i].startFrame;
                int lDur = lost[j].frameId - lost[j].startFrame;
                if (tDur > lDur) {
                    removeLost[j] = true;
                } else {
                    removeTracked[i] = true;
                }
            }
        }
    }
    vector<STrack> newTracked, newLost;
    for (size_t i = 0; i < tracked.size(); i++) {
        if (!removeTracked[i]) newTracked.push_back(tracked[i]);
    }
    for (size_t j = 0; j < lost.size(); j++) {
        if (!removeLost[j]) newLost.push_back(lost[j]);
    }
    tracked.swap(newTracked);
    lost.swap(newLost);
}

// 将 src 中未在 dst 出现过的轨迹(按 trackId 去重)追加到 dst
static void jointStracks(vector<STrack> &dst, const vector<STrack> &src) {
    for (const STrack &t: src) {
        bool found = false;
        for (const STrack &d: dst) {
            if (d.trackId == t.trackId) {
                found = true;
                break;
            }
        }
        if (!found) dst.push_back(t);
    }
}

void BYTETracker::update(const vector<cv::Rect2f> &detBoxes, const vector<float> &detScores,
                         vector<Track> &outTracks) {
    outTracks.assign(detBoxes.size(), Track());
    frameId++;

    // 第一步：按分数把检测分为高分/低分两组(记录其在原始检测中的下标)
    vector<cv::Rect2f> highDets, lowDets;
    vector<float> highScores, lowScores;
    vector<int> highIdx, lowIdx;
    for (size_t i = 0; i < detBoxes.size(); i++) {
        float s = i < detScores.size() ? detScores[i] : 0.f;
        if (s > trackThresh) {
            highDets.push_back(detBoxes[i]);
            highScores.push_back(s);
            highIdx.push_back((int) i);
        } else if (s > lowThresh) {
            lowDets.push_back(detBoxes[i]);
            lowScores.push_back(s);
            lowIdx.push_back((int) i);
        }
    }

    // 第二步：预测所有轨迹在当前帧的位置
    for (size_t i = 0; i < trackedStracks.size(); i++) trackedStracks[i].predict(kalmanFilter);
    for (size_t i = 0; i < lostStracks.size(); i++) lostStracks[i].predict(kalmanFilter);

    // 关联池：正在跟踪 + 短暂丢失(它们的指针在本次更新内保持有效)
    vector<STrack *> pool;
    for (size_t i = 0; i < trackedStracks.size(); i++) pool.push_back(&trackedStracks[i]);
    for (size_t i = 0; i < lostStracks.size(); i++) pool.push_back(&lostStracks[i]);

    vector<STrack> activated;  // 本帧成功关联的轨迹
    vector<STrack> refind;     // 丢失后重新找回的轨迹
    vector<STrack> lost;       // 本帧新丢失的轨迹
    vector<STrack> removed;    // 本帧被删除的轨迹

    // 第三步：第一次关联——高分检测 vs 关联池
    vector<vector<float>> cost;
    iouDistance(pool, highDets, cost);
    fuseScore(cost, highScores);
    vector<pair<int, int>> matches;
    vector<int> uTrack, uDet;
    linearAssignment(cost, matchThresh, matches, uTrack, uDet);

    vector<bool> highMatched(highDets.size(), false);
    for (const auto &m: matches) {
        STrack *track = pool[m.first];
        int d = m.second;
        highMatched[d] = true;
        if (track->state == STrack::Tracked) {
            track->updateState(kalmanFilter, highDets[d], highScores[d], frameId, maxTrajLen);
            track->detIndex = highIdx[d];
            activated.push_back(*track);
        } else {
            track->reActivate(kalmanFilter, highDets[d], highScores[d], frameId, maxTrajLen);
            track->detIndex = highIdx[d];
            refind.push_back(*track);
        }
    }

    // 第四步：第二次关联——低分检测 vs 第一次未匹配的“正在跟踪”轨迹
    vector<STrack *> rTracked;
    for (int idx: uTrack) {
        if (pool[idx]->state == STrack::Tracked) rTracked.push_back(pool[idx]);
    }
    const float kSecondMatchThresh = 0.5f;
    vector<vector<float>> cost2;
    iouDistance(rTracked, lowDets, cost2);
    vector<pair<int, int>> matches2;
    vector<int> uTrack2, uDet2;
    linearAssignment(cost2, kSecondMatchThresh, matches2, uTrack2, uDet2);
    for (const auto &m: matches2) {
        STrack *track = rTracked[m.first];
        int d = m.second;
        if (track->state == STrack::Tracked) {
            track->updateState(kalmanFilter, lowDets[d], lowScores[d], frameId, maxTrajLen);
            track->detIndex = lowIdx[d];
            activated.push_back(*track);
        } else {
            track->reActivate(kalmanFilter, lowDets[d], lowScores[d], frameId, maxTrajLen);
            track->detIndex = lowIdx[d];
            refind.push_back(*track);
        }
    }
    // 二次关联后仍未匹配的“正在跟踪”轨迹标记为丢失
    for (int idx: uTrack2) {
        STrack *track = rTracked[idx];
        if (track->state != STrack::Lost) {
            track->markLost();
            lost.push_back(*track);
        }
    }

    // 第五步：未被匹配的高分检测，分数足够高则新建轨迹
    for (size_t d = 0; d < highDets.size(); d++) {
        if (highMatched[d]) continue;
        if (highScores[d] < highThresh) continue;
        STrack track;
        track.activate(kalmanFilter, highDets[d], highScores[d], frameId, nextId++, maxTrajLen);
        track.detIndex = highIdx[d];
        activated.push_back(track);
    }

    // 第六步：丢失时间过长的轨迹删除
    for (size_t i = 0; i < lostStracks.size(); i++) {
        if (frameId - lostStracks[i].frameId > maxTimeLost) {
            lostStracks[i].markRemoved();
            removed.push_back(lostStracks[i]);
        }
    }

    // 第七步：重组轨迹列表
    vector<STrack> newTracked;
    for (const STrack &t: trackedStracks) {
        if (t.state == STrack::Tracked) newTracked.push_back(t);
    }
    jointStracks(newTracked, activated);
    jointStracks(newTracked, refind);

    vector<STrack> newLost;
    for (const STrack &t: lostStracks) {
        if (t.state == STrack::Lost) newLost.push_back(t);
    }
    jointStracks(newLost, lost);
    // 已被跟踪或已删除的轨迹不应留在丢失列表中
    for (auto it = newLost.begin(); it != newLost.end();) {
        bool tracked = false;
        for (const STrack &t: newTracked) {
            if (t.trackId == it->trackId) {
                tracked = true;
                break;
            }
        }
        if (tracked) {
            it = newLost.erase(it);
        } else {
            ++it;
        }
    }

    removeDuplicates(newTracked, newLost);

    trackedStracks.swap(newTracked);
    lostStracks.swap(newLost);
    for (const STrack &t: removed) {
        removedStracks.push_back(t);
    }

    // 第八步：将跟踪信息写回检测结果
    for (const STrack &t: trackedStracks) {
        if (!t.isActivated) continue;
        if (t.detIndex >= 0 && t.detIndex < (int) outTracks.size()) {
            Track &info = outTracks[t.detIndex];
            info.idx = t.trackId;
            info.lost = 0;
            info.index = t.detIndex;
            info.trajectory = t.trajectory;
        }
    }
}
