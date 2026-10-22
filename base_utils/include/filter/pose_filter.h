//
// Created by PKing on 2020/6/3.
//


#ifndef DETECTOR_POSE_FILTER_H
#define DETECTOR_POSE_FILTER_H

#include <vector>
#include <deque>
#include <map>
#include <opencv2/opencv.hpp>

/***
 * 关键点去抖动滤波器（视频时序后处理），对应 pose_filter.py 的 C++ 实现
 * max_step 建议设为略大于目标真实最大单帧位移。设得过小（小于真实速度）会出现“保持→跳一步”的阶梯跟踪；不确定时保持 None。
 * 静止场景抖动仍明显：减小 decay（如 0.05）或增大 wsize。
 * 快速运动拖影：增大 decay、降低 vTh。
 * 手部等有快速局部运动的场景，可把 median_size 设为 1 关闭中值预筛以减小 1 帧延迟。
 * 对每个关键点独立维护历史轨迹，主要处理：
 * 1. 中值预筛：先用中值滤波剔除单帧孤立跳点，再做平滑，避免误检点污染历史；
 * 2. 失效保护：检测失败的关键点（(0,0)/(-1,-1)/NaN/inf）不参与平滑，短期输出上一次的有效结果，
 *    连续丢失超过 lost_frames 帧后清空该点历史，避免长期输出陈旧坐标、恢复时跳变；
 * 3. 跳变重捕获：位移超过 max_step 时先暂停更新（保持上一次输出），连续 recover_frames 帧
 *    同方向超限才判定为真实快速运动并重新捕获，兼顾抗漂移与快速跟踪；
 * 4. 指数加权平滑：对最近 wsize 帧的观测按 decay 指数衰减加权平均，越久远的帧权重越小；
 * 5. 速度自适应：用平滑后的速度调整平滑力度，快速运动时减弱平滑（减小拖影/延迟），
 *    静止或微抖时加强平滑；速度按帧间隔 dt 归一化，跳帧时不会误判。
 */
class PoseFilter {
public:
    /***
     * @param kid:   需要跟踪的关键点ID，可指定多个,如{0,1,2,3}；为空时处理全部关键点
     * @param wsize: 平滑窗口，即参与平滑的历史帧数；越大越平滑，但延迟也越大
     * @param decay: 权重衰减系数，值越大，历史影响衰减的越快，平滑力度越小，取值范围[0,1)
     * @param vTh:   速度阈值（像素/帧），位移超过该值时减弱平滑以降低延迟；<=0 表示关闭速度自适应
     * @param max_step: 单帧最大允许位移（像素/帧），用于区分"偶发跳变"和"真实快速运动"；
     *                  建议设为略大于目标真实最大单帧位移；<=0 表示不启用跳变检测
     * @param median_size: 中值预筛窗口大小，用于剔除孤立跳点；1 表示关闭预筛
     * @param recover_frames: 连续 N 帧同方向超过 max_step 时，判定为真实运动并重新捕获
     * @param lost_frames: 关键点连续丢失帧数超过该值时清空其历史状态
     */
    PoseFilter(const std::vector<int> &kid = std::vector<int>(), int wsize = 10, float decay = 0.1f,
               float vTh = 15.0f, float max_step = -1.0f, int median_size = 3,
               int recover_frames = 3, int lost_frames = 15);

    ~PoseFilter();

    /***
     * 清空滤波状态（切换视频、目标ID变化或场景切换时调用）
     */
    void reset();

    /***
     * 对关键点数组进行时序平滑（原地修改并返回）
     * @param points: (N,2)关键点，单位：像素
     * @param dt: 帧间隔（以帧为单位，默认1.0；跳帧/变帧率时传入实际间隔，用于速度归一化）
     * @return: 平滑后的关键点（即入参 points 本身）
     */
    std::vector<cv::Point2f> &update(std::vector<cv::Point2f> &points, float dt = 1.0f);

private:
    // 单个关键点的滤波状态
    struct State {
        std::deque<cv::Point2f> raw;    // 中值预筛缓存
        std::deque<cv::Point2f> hist;   // 参与加权平均的观测
        cv::Point2f last;               // 上一次输出的平滑坐标
        bool has_last = false;          // last 是否有效（对应 Python 的 last is None）
        float speed = 0.0f;             // 平滑后的运动速度（像素/帧）
        int jump_cnt = 0;               // 连续超限帧数
        cv::Point2f jump_dir;           // 上一次超限位移的方向
        bool has_jump_dir = false;      // jump_dir 是否有效（对应 Python 的 jump_dir is None）
        int miss = 0;                   // 连续丢失帧数
    };

    // 获取（或初始化）关键点 k 的滤波状态
    State &getState(int k);

    // 对单个关键点做时序平滑
    cv::Point2f smooth(State &state, const cv::Point2f &point, float dt);

    // 处理单个关键点的一帧观测（原地修改 point）
    void updatePoint(int k, cv::Point2f &point, float dt);

    // 判断关键点是否有效：排除 NaN/inf 以及检测失败时的占位点(0,0)/(-1,-1)
    static bool isValid(const cv::Point2f &point);

    // 取缓存的单坐标中值（use_y 为 true 时取 y）
    static float median(const std::deque<cv::Point2f> &buf, bool use_y);

    // 有界队列写入，超过 maxlen 时丢弃最旧的数据
    static void pushBounded(std::deque<cv::Point2f> &buf, const cv::Point2f &point, size_t maxlen);

private:
    std::vector<int> mKid;
    int mWsize;
    float mDecay;
    float mVTh;
    float mMaxStep;
    int mMedianSize;
    int mRecoverFrames;
    int mLostFrames;
    std::map<int, State> mState; // 关键点ID -> 滤波状态
};


#endif //DETECTOR_POSE_FILTER_H
