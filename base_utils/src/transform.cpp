//
// Created by Pan on 2024/4/24.
//

#include <debug.h>
#include "transform.h"


void get_order_points(vector<cv::Point2f> inp, vector<cv::Point2f> &dst) {
    vector<float> x_add_y, x_dif_y;
    for (int i = 0; i < inp.size(); ++i) {
        x_add_y.push_back(inp.at(i).y + inp.at(i).x);
        x_dif_y.push_back(inp.at(i).y - inp.at(i).x);
    }
    /***
     *     0(top-left)----(w10)----1(top-right)
     *        |                       |
     *      (h30)                    (h21)
     *        |                       |
     *    3(bottom-left)--(w23)---2(bottom-right)
     */
    // [top-left, top-right, bottom-right, bottom-left]
    int tl, br, tr, bl;
    find_min_max_indices<float>(x_add_y, tl, br);
    find_min_max_indices<float>(x_dif_y, tr, bl);
    dst.clear();
    dst.push_back(inp.at(tl));
    dst.push_back(inp.at(tr));
    dst.push_back(inp.at(br));
    dst.push_back(inp.at(bl));
}

void get_transform(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts, cv::Mat &M,
                   int method) {
    if (method == 0) {
        M = solve_lstsq(src_pts, dst_pts);
    } else {
        M = cv::estimateAffine2D(src_pts, dst_pts);
    }
}

void get_transform(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts, cv::Mat &M, cv::Mat &Minv,
                   int method) {
    get_transform(src_pts, dst_pts, M);
    cv::invertAffineTransform(M, Minv);
}

cv::Mat solve_lstsq(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts) {
    int size = src_pts.size();
    cv::Mat src_pts_ = cv::Mat_<float>(size, 3);
    cv::Mat dst_pts_ = cv::Mat_<float>(size, 3);
    for (int i = 0; i < src_pts.size(); ++i) {
        src_pts_.at<cv::Vec3f>(i) = cv::Point3f(src_pts[i].x, src_pts[i].y, 1.);
        dst_pts_.at<cv::Vec3f>(i) = cv::Point3f(dst_pts[i].x, dst_pts[i].y, 1.);
    }
    cv::Mat H = cv::Mat_<float>(3, 3);
    cv::Mat M = cv::Mat_<float>(2, 3);
    // 通过最小二乘法计算变换矩阵
    bool res = cv::solve(src_pts_, dst_pts_, H, cv::DECOMP_QR);
    M.at<cv::Vec3f>(0) = cv::Point3f(
            H.at<float>(0, 0),
            H.at<float>(1, 0),
            H.at<float>(2, 0));
    M.at<cv::Vec3f>(1) = cv::Point3f(
            H.at<float>(0, 1),
            H.at<float>(1, 1),
            H.at<float>(2, 1));
    return M;
}


cv::Mat image_alignment(cv::Mat &image,
                        vector<cv::Point2f> src_pts,
                        vector<cv::Point2f> &dst_pts,
                        cv::Size dsize,
                        cv::Size2f scale,
                        int flags,
                        int borderMode,
                        cv::Scalar color) {
    // TODO dst_pts是src_pts变换后目前区域的位置点
    // TODO 若dst_pts为空，则利用src_pts计算dst_pts和tsize，相当于只映射src_pts轮廓的最小外接矩形框
    cv::Size2f tsize;//tsize是变换后目前区域矩形框
    if (dst_pts.empty()) {
        float w10 = cal_distance(src_pts.at(1), src_pts.at(0));
        float w23 = cal_distance(src_pts.at(2), src_pts.at(3));
        float h21 = cal_distance(src_pts.at(2), src_pts.at(1));
        float h30 = cal_distance(src_pts.at(3), src_pts.at(0));
        tsize = cv::Size2f((w10 + w23) / 2.f, (h21 + h30) / 2.f);
        dst_pts = {cv::Point2f{0, 0}, cv::Point2f{tsize.width, 0},
                   cv::Point2f{tsize.width, tsize.height}, cv::Point2f{0, tsize.height}};
    } else {
        tsize = cv::Size2f(image.cols, image.rows);
    }
    if (dsize.width < 0 || dsize.height < 0) {
        dsize = cv::Size(tsize.width, tsize.height);
    }
    // 映射居中
    cv::Size2f diff(dsize.width - tsize.width, dsize.height - tsize.height);
    for (int i = 0; i < dst_pts.size(); i++) {
        dst_pts[i].x = (dst_pts[i].x + diff.width / 2.f) * dsize.width / dsize.width;
        dst_pts[i].y = (dst_pts[i].y + diff.height / 2.f) * dsize.height / dsize.height;
    }

    // 缩放大小
    diff.width = dsize.width * scale.width - dsize.width;
    diff.height = dsize.height * scale.height - dsize.height;
    dsize = cv::Size((int) (dsize.width * scale.width), (int) (dsize.height * scale.height));
    for (int i = 0; i < dst_pts.size(); i++) {
        dst_pts[i].x = dst_pts[i].x + diff.width / 2.f;
        dst_pts[i].y = dst_pts[i].y + diff.height / 2.f;
    }
    // get transformer matrix
    cv::Mat M;
    cv::Mat I;
    get_transform(src_pts, dst_pts, M, 0);
    cv::warpAffine(image, I, M, dsize, flags, borderMode, color);
    return I;
}