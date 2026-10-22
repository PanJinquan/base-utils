//
// Created by PKing on 2026/10/12.
//

#ifndef DETECTOR_BYTETRACKER_H
#define DETECTOR_BYTETRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "base.h"

using namespace std;

/***
 * 卡尔曼滤波器(ByteTrack 采用的 XYAH 状态模型)
 *
 * 状态向量：[cx, cy, a, h, vx, vy, va, vh]，
 * 其中 cx/cy 为目标框中心，a 为宽高比(w/h)，h 为高；
 * 观测向量：[cx, cy, a, h]。
 */
class KalmanFilterXYAH {
public:
    KalmanFilterXYAH();

    /*** 依据观测框初始化状态均值与协方差 */
    void initiate(const cv::Rect2f &box, cv::Mat &mean, cv::Mat &covariance) const;

    /*** 状态预测(匀速模型) */
    void predict(cv::Mat &mean, cv::Mat &covariance) const;

    /*** 依据观测框校正状态均值与协方差 */
    void update(cv::Mat &mean, cv::Mat &covariance, const cv::Rect2f &box) const;

    /*** 由状态均值恢复目标框(tlwh 形式) */
    static cv::Rect2f toBox(const cv::Mat &mean);

private:
    /*** 将 tlwh 框转换为 XYAH 观测向量(4x1) */
    static cv::Mat boxToXyah(const cv::Rect2f &box);

    /*** 由 XYAH 状态均值恢复 tlwh 框 */
    static cv::Rect2f xyahToBox(const cv::Mat &xyah);

    cv::Mat motionMat;        // 8x8 状态转移矩阵(匀速模型)
    cv::Mat updateMat;        // 4x8 观测矩阵
    float stdWeightPosition;  // 位置噪声权重
    float stdWeightVelocity;  // 速度噪声权重
};


/***
 * 单条轨迹(ByteTrack 中的 STrack)
 */
struct STrack {
    enum State {
        Tracked = 1,  // 正在跟踪
        Lost = 2,     // 短暂丢失
        Removed = 3,  // 已删除
    };

    cv::Rect2f box;                  // 目标框(tlwh：左上角 x/y + 宽/高)
    cv::Mat mean;                    // 卡尔曼状态均值(8x1)
    cv::Mat covariance;              // 卡尔曼状态协方差(8x8)
    int state = Tracked;             // 轨迹状态
    bool isActivated = false;        // 是否已激活(可输出)
    int trackId = 0;                 // 轨迹ID
    int frameId = 0;                 // 最近一次更新的帧序号
    int startFrame = 0;              // 起始帧序号
    int trackletLen = 0;             // 连续匹配的帧数
    float score = 0.f;               // 最近一次匹配的检测分数
    int detIndex = -1;               // 当前帧匹配的检测索引(未匹配为-1)
    vector<cv::Point2f> trajectory;  // 目标中心点的运动轨迹

    /*** 以检测框新建轨迹 */
    void activate(const KalmanFilterXYAH &kf, const cv::Rect2f &det, float detScore,
                  int frame, int id, int maxTrajLen);

    /*** 丢失轨迹重新匹配到检测后重新激活 */
    void reActivate(const KalmanFilterXYAH &kf, const cv::Rect2f &det, float detScore,
                    int frame, int maxTrajLen);

    /*** 已跟踪轨迹匹配到检测后更新 */
    void updateState(const KalmanFilterXYAH &kf, const cv::Rect2f &det, float detScore,
                     int frame, int maxTrajLen);

    /*** 卡尔曼状态预测，并同步更新目标框位置 */
    void predict(const KalmanFilterXYAH &kf);

    /*** 标记为丢失 */
    void markLost();

    /*** 标记为删除 */
    void markRemoved();

private:
    /*** 记录目标框中心点到轨迹(限制最大长度) */
    void appendTrajectory(const cv::Rect2f &det, int maxTrajLen);
};


/***
 * ByteTrack 多目标跟踪器
 *
 * 核心思想：充分利用检测框的分数，将检测分为高分/低分两组：
 * 1. 先用高分检测与“正在跟踪 + 短暂丢失”的轨迹做关联；
 * 2. 再用低分检测对剩余轨迹做二次关联，找回被遮挡/模糊的目标，
 *    从而减少漏跟与ID跳变。
 */
class BYTETracker {
public:
    /***
     * 构造函数
     * @param frameRate    视频帧率
     * @param trackBuffer  轨迹保留帧数(按30fps折算为最大丢失帧数)
     * @param trackThresh  高分/低分检测的分界分数
     * @param highThresh   新建轨迹所需的最低检测分数
     * @param matchThresh  首次关联的匹配阈值(1 - IoU)
     * @param lowThresh    低分检测的下限分数
     * @param maxTrajLen   每条轨迹最多保留的轨迹点数
     */
    BYTETracker(int frameRate = 30, int trackBuffer = 30, float trackThresh = 0.5f,
                float highThresh = 0.6f, float matchThresh = 0.8f, float lowThresh = 0.1f,
                int maxTrajLen = 30);

    ~BYTETracker();

    /*** 清空所有轨迹，重新开始跟踪 */
    void reset();

    /***
     * 更新跟踪结果
     * @param detBoxes  当前帧检测框(tlwh 形式的 cv::Rect2f：x/y 为左上角)
     * @param detScores 当前帧检测分数(与 detBoxes 一一对应)
     * @param outTracks 输出：与 detBoxes 一一对应的跟踪信息(未匹配的检测 idx=-1)
     */
    void update(const vector<cv::Rect2f> &detBoxes, const vector<float> &detScores,
                vector<Track> &outTracks);

private:
    /*** 计算两个矩形框的 IoU(交并比) */
    static float iou(const cv::Rect2f &a, const cv::Rect2f &b);

    /*** 计算轨迹与检测框的 IoU 代价矩阵(1 - IoU，行=轨迹，列=检测) */
    static void iouDistance(const vector<STrack *> &tracks, const vector<cv::Rect2f> &dets,
                            vector<vector<float>> &cost);

    /*** 用检测分数融合代价矩阵(det_score fuse) */
    static void fuseScore(vector<vector<float>> &cost, const vector<float> &scores);

    /***
     * 匈牙利算法求解指派问题，并按阈值筛除低质量匹配
     * @param cost    代价矩阵(行=轨迹，列=检测)
     * @param thresh  匹配阈值，代价大于该值的匹配将被丢弃
     * @param matches 输出：匹配对(轨迹下标, 检测下标)
     * @param uTracks 输出：未匹配的轨迹下标
     * @param uDets   输出：未匹配的检测下标
     */
    static void linearAssignment(const vector<vector<float>> &cost, float thresh,
                                 vector<pair<int, int>> &matches,
                                 vector<int> &uTracks, vector<int> &uDets);

    /*** 剔除 tracked 与 lost 之间高度重叠的重复轨迹 */
    static void removeDuplicates(vector<STrack> &tracked, vector<STrack> &lost);

private:
    KalmanFilterXYAH kalmanFilter;  // 共享卡尔曼滤波器
    vector<STrack> trackedStracks;  // 正在跟踪的轨迹
    vector<STrack> lostStracks;     // 短暂丢失的轨迹
    vector<STrack> removedStracks;  // 已删除的轨迹
    int frameId = 0;                // 当前帧序号
    int nextId = 0;                 // 下一个轨迹ID
    int maxTimeLost;                // 轨迹最多丢失多少帧
    int maxTrajLen;                 // 轨迹最多保留的轨迹点数
    float trackThresh;              // 高分/低分检测分界
    float highThresh;               // 新建轨迹最低分数
    float matchThresh;              // 首次关联阈值
    float lowThresh;                // 低分检测下限
};


#endif //DETECTOR_BYTETRACKER_H
