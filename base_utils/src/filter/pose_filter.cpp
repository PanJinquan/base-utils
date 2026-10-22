//
// Created by Pking on 2020/6/3.
//

#include <algorithm>
#include <cmath>
#include "filter/pose_filter.h"


PoseFilter::PoseFilter(const std::vector<int> &kid, int wsize, float decay, float vTh,
                       float max_step, int median_size, int recover_frames, int lost_frames)
        : mKid(kid),
          mWsize(std::max(wsize, 1)),
          mDecay(std::min(std::max(decay, 0.0f), 0.999f)),
          mVTh(vTh),
          mMaxStep(max_step),
          mMedianSize(std::max(median_size, 1)),
          mRecoverFrames(std::max(recover_frames, 1)),
          mLostFrames(std::max(lost_frames, 1)) {
}


PoseFilter::~PoseFilter() {
    reset();
}


void PoseFilter::reset() {
    mState.clear();
}


std::vector<cv::Point2f> &PoseFilter::update(std::vector<cv::Point2f> &points, float dt) {
    if (points.empty()) {
        return points;
    }
    // 帧间隔非法（<=0 或 NaN）时按 1 帧处理
    if (!(dt > 0.0f)) {
        dt = 1.0f;
    }
    int nums = (int) points.size();
    if (mKid.empty()) {
        // kid 为空时平滑全部关键点
        for (int k = 0; k < nums; ++k) {
            updatePoint(k, points.at(k), dt);
        }
    } else {
        for (size_t i = 0; i < mKid.size(); ++i) {
            int k = mKid.at(i);
            if (k < 0 || k >= nums) {
                continue;
            }
            updatePoint(k, points.at(k), dt);
        }
    }
    return points;
}


void PoseFilter::updatePoint(int k, cv::Point2f &point, float dt) {
    if (!isValid(point)) {
        // 无效点：短期保持上一次有效结果，长时间丢失则清空历史，避免输出陈旧坐标
        std::map<int, State>::iterator it = mState.find(k);
        if (it == mState.end() || !it->second.has_last) {
            return;
        }
        State &state = it->second;
        state.miss += 1;
        if (state.miss > mLostFrames) {
            mState.erase(it);
        } else {
            point = state.last;
        }
        return;
    }
    State &state = getState(k);
    state.miss = 0;
    point = smooth(state, point, dt);
}


PoseFilter::State &PoseFilter::getState(int k) {
    std::map<int, State>::iterator it = mState.find(k);
    if (it == mState.end()) {
        it = mState.insert(std::make_pair(k, State())).first;
    }
    return it->second;
}


cv::Point2f PoseFilter::smooth(State &state, const cv::Point2f &point, float dt) {
    // 1. 中值预筛：剔除单帧孤立跳点，避免误检污染平滑历史
    pushBounded(state.raw, point, (size_t) mMedianSize);
    cv::Point2f obs;
    if (mMedianSize > 1) {
        obs.x = median(state.raw, false);
        obs.y = median(state.raw, true);
    } else {
        obs = point;
    }
    if (!state.has_last) {
        pushBounded(state.hist, obs, (size_t) mWsize);
        state.last = obs;
        state.has_last = true;
        return obs;
    }
    // 2. 跳变检测：超限时先保持上一次输出，连续同方向超限才认为是真实快速运动
    cv::Point2f disp = obs - state.last;
    float dist = std::sqrt(disp.x * disp.x + disp.y * disp.y);
    if (mMaxStep > 0 && dist > mMaxStep * dt) {
        cv::Point2f direction(disp.x / (dist + 1e-6f), disp.y / (dist + 1e-6f));
        if (state.has_jump_dir && (state.jump_dir.x * direction.x + state.jump_dir.y * direction.y) > 0.8f) {
            state.jump_cnt += 1;
        } else {
            state.jump_cnt = 1;
        }
        state.jump_dir = direction;
        state.has_jump_dir = true;
        if (state.jump_cnt < mRecoverFrames) {
            return state.last; // 视为偶发跳变，保持上一次输出
        }
        // 确认为真实运动：清空历史，直接重新捕获当前观测
        state.raw.clear();
        pushBounded(state.raw, obs, (size_t) mMedianSize);
        state.hist.clear();
        pushBounded(state.hist, obs, (size_t) mWsize);
        state.speed = 0.0f;
        state.jump_cnt = 0;
        state.jump_dir = cv::Point2f(0.0f, 0.0f);
        state.has_jump_dir = false;
        state.last = obs;
        return obs;
    }
    state.jump_cnt = 0;
    state.jump_dir = cv::Point2f(0.0f, 0.0f);
    state.has_jump_dir = false;
    // 3. 速度估计：按 dt 归一化，并做 EMA 平滑，避免单帧噪声导致平滑力度剧烈变化
    float speed = 0.5f * (dist / dt) + 0.5f * state.speed;
    state.speed = speed;
    // 4. 速度自适应：运动越快，decay 越接近1，历史权重越小（平滑越弱，延迟越小）
    float decay = mDecay;
    if (mVTh > 0) {
        decay = decay + (1.0f - decay) * std::min(speed / mVTh, 1.0f);
    }
    // 帧间隔越大，历史帧越"过期"，按 dt 次方加速衰减
    decay = 1.0f - std::pow(1.0f - decay, dt);
    pushBounded(state.hist, obs, (size_t) mWsize);
    // 5. 指数加权滑动平均：越新的观测权重越大，age 为距当前帧的帧数
    int n = (int) state.hist.size();
    float sum_w = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int i = 0; i < n; ++i) {
        int age = n - 1 - i;
        float w = std::pow(1.0f - decay, (float) age);
        sum_w += w;
        sum_x += w * state.hist.at(i).x;
        sum_y += w * state.hist.at(i).y;
    }
    cv::Point2f smooth_pt = obs;
    if (sum_w > 0.0f) {
        smooth_pt = cv::Point2f(sum_x / sum_w, sum_y / sum_w);
    }
    state.last = smooth_pt;
    return smooth_pt;
}


bool PoseFilter::isValid(const cv::Point2f &point) {
    // 排除 NaN/inf，以及检测失败时的占位点：Python 版本使用(0,0)，C++后处理使用(-1,-1)
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        return false;
    }
    if (point.x == 0.0f && point.y == 0.0f) {
        return false;
    }
    if (point.x < 0.0f || point.y < 0.0f) {
        return false;
    }
    return true;
}


float PoseFilter::median(const std::deque<cv::Point2f> &buf, bool use_y) {
    if (buf.empty()) {
        return 0.0f;
    }
    std::vector<float> values;
    values.reserve(buf.size());
    for (size_t i = 0; i < buf.size(); ++i) {
        values.push_back(use_y ? buf.at(i).y : buf.at(i).x);
    }
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 1) {
        return values.at(n / 2);
    }
    // 偶数个元素取中间两个的均值（与 numpy.median 一致）
    return 0.5f * (values.at(n / 2 - 1) + values.at(n / 2));
}


void PoseFilter::pushBounded(std::deque<cv::Point2f> &buf, const cv::Point2f &point, size_t maxlen) {
    if (maxlen == 0) {
        return;
    }
    while (buf.size() >= maxlen) {
        buf.pop_front();
    }
    buf.push_back(point);
}
