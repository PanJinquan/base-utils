//
// Created by PKing on 2026/9/23.
//

#ifndef DETECTOR_TRACKER_H
#define DETECTOR_TRACKER_H

#include <opencv2/opencv.hpp>
// 是否启用 OpenCV tracking 模块(CSRT/MOSSE/KCF)。OpenCV 较老版本不包含
// <opencv2/tracking.hpp> 与 <opencv2/tracking/tracking_legacy.hpp>，此时在 CMake 中取消定义即可关闭；
// SORT 与 BYTE 不依赖该头文件，不受影响。
// 注意：该宏会影响 Tracker 类的成员布局，必须保证所有编译单元(含 base_utils)定义一致。
// CMake 中已在 add_subdirectory(base_utils) 之前统一添加，勿在此处单独定义。
// #define OPENCV_TRACK_ON

#ifdef OPENCV_TRACK_ON
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#endif
#include <vector>
#include "base.h"
#include "bytetrack.h"

using namespace std;

/***
 * 多目标跟踪器
 *
 * 实现思路：
 * 1. 以检测结果(来自YOLO)为主：可见目标直接用检测框通过 IoU 关联保持目标ID稳定；
 * 2. 目标短暂丢失时，才为该目标增量创建一个单目标跟踪器(按 algo)预测其位置，
 *    待重新被检测到或超过 maxLost 后销毁，避免每帧全量重建带来的开销；
 * 3. 记录每个目标中心点的历史位置形成运动轨迹，并将跟踪信息写入 frameInfo.info[i].track。
 */
class Tracker {
public:
    /*** 可选的目标跟踪算法(CSRT、KCF、MOSSE 需定义 OPENCV_TRACK_ON，否则自动回退到 SORT)*/
    enum class AlgType {
        SORT,  // 极快，SORT(卡尔曼滤波算法)，适合实时
        KCF,   // 一般，速度与精度平衡，KFC(核相关滤波)
        CSRT,  // 较慢，精度高、尺度变化鲁棒
        MOSSE, // 较快，精度一般，MOSSE(最小输出平方误差和)跟踪器
        BYTE,  // 较快，ByteTrack(高低分检测两次关联 + 卡尔曼滤波)，抗遮挡、ID稳定
    };

    /***
     * 构造函数
     * @param maxLost 目标丢失多少帧后删除
     * @param maxSzie 轨迹最多保留的点数
     * @param iouThresh 关联匹配的 IoU 阈值
     * @param algo 目标跟踪算法，支持 SORT(默认)、KCF、CSRT、MOSSE等算法
     */
    Tracker(int maxLost = 30, int maxSzie = 10, float iouThresh = 0.3f, AlgType algo = AlgType::SORT);

    /***
     *析构函数
     */
    ~Tracker();

    /***
     * 更新多目标跟踪结果
     * @param image     当前帧图像(BGR)
     * @param frameInfo 输入：当前帧检测结果；输出：填充每个检测目标的 track 信息(id、轨迹等)
     */
    void update(const cv::Mat &image, vector<ImageInfo> &dets);

    /***
     * 清空所有跟踪目标，重新开始跟踪
     */
    void reset();

private:
    /*** 计算两个矩形框的 IoU(交并比) */
    float iou(const cv::Rect2f &a, const cv::Rect2f &b);

    /***
     * 贪心 IoU 匹配：将 boxes 依次匹配到 trackBoxes 上
     * @param boxes      待匹配的框
     * @param trackBoxes 已有跟踪目标框
     * @param thresh     匹配阈值，IoU 低于该值视为不匹配
     * @return 返回与每个 box 匹配的 trackBoxes 下标，未匹配为 -1
     */
    vector<int> greedyMatch(const vector<cv::Rect2f> &boxes, const vector<cv::Rect2f> &trackBoxes, float thresh);

    /*** 记录目标中心点到轨迹中(同时限制轨迹最大长度) */
    void appendTrajectory(Track &track, const cv::Rect2f &box);

    /*** 按 algo 创建一个单目标跟踪器(仅 CSRT、MOSSE、KCF 算法可用) */
#ifdef OPENCV_TRACK_ON
    cv::Ptr<cv::legacy::Tracker> makeTracker() const;
#endif

    /*** 创建一个卡尔曼滤波器并以 box 作为初始状态(SORT 算法) */
    cv::KalmanFilter makeKalmanFilter(const cv::Rect2f &box) const;

private:
#ifdef OPENCV_TRACK_ON
    vector<cv::Ptr<cv::legacy::Tracker>> trackers;  // 与 obj_tracks 一一对应的单目标跟踪器(仅丢失中的目标非空)
#endif
    vector<cv::KalmanFilter> kalmanFilters;         // 与 obj_tracks 一一对应的卡尔曼滤波器(仅 SORT 模式使用)
    vector<ImageInfo> obj_tracks;                  // 当前活跃的跟踪目标(目标框 + track 信息)
    int nextID = 0;                                 // 下一个可分配的目标ID
    int maxLost;                                    // 目标丢失多少帧后删除
    int maxSzie;                                    // 轨迹最多保留的点数
    float iouThresh;                                // 关联匹配的 IoU 阈值
    AlgType algo;                                   // 目标跟踪算法
    BYTETracker byteTracker;                        // ByteTrack 跟踪器(仅 BYTE 模式使用)
};

#endif //DETECTOR_TRACKER_H
