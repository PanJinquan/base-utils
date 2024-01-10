//
// Created by 390737991@qq.com on 2018/6/3.
//

#include <debug.h>
#include "face_alignment.h"

FaceAlignment::FaceAlignment(int faceWidth, int faceHeight, float ex, float ey, bool square) {
    face_width = faceWidth;
    face_height = faceHeight;
    // Get reference facial points
    // std::vector<std::vector<float>> tmp_ref_landmarks(default_facial_points);
    dst_pts = new cv::Point2f[num_landmarks];
    memcpy(dst_pts, kpts_ref, num_landmarks * sizeof(cv::Point2f));
    // std::vector<int> tmp_crop_size(default_crop_size);
    if (square || size_ref[0] != faceWidth || size_ref[1] != faceHeight) {
        float maxL = size_ref[0] > size_ref[1] ? size_ref[0] : size_ref[1];
        float w_diff = maxL - size_ref[0];
        float h_diff = maxL - size_ref[1];
        for (int i = 0; i < num_landmarks; i++) {
            dst_pts[i].x = (dst_pts[i].x + w_diff / 2.) * faceWidth / maxL;
            dst_pts[i].y = (dst_pts[i].y + h_diff / 2.) * faceHeight / maxL;
        }
    }
    // 缩放大小
    float w_diff = face_width * ex - face_width;
    float h_diff = face_height * ey - face_height;
    face_width = (int) (face_width * ex);
    face_height = (int) (face_height * ey);
    for (int i = 0; i < num_landmarks; i++) {
        dst_pts[i].x = dst_pts[i].x + w_diff / 2.;
        dst_pts[i].y = dst_pts[i].y + h_diff / 2.;
    }
    for (int i = 0; i < num_landmarks; i++) {
        printf("kpts_ref x:%f,y:%f\n", dst_pts[i].x, dst_pts[i].y);
    }
    printf("init face_alignment successful.\n");
}

FaceAlignment::~FaceAlignment() {
    // ref_landmarks.reset()
    delete[] dst_pts;
}


void FaceAlignment::crop_faces_alignment(cv::Mat &image, vector<cv::Point2f> &landmarks, cv::Mat &out_face) {
    // get transformer matrix
    std::vector<cv::Point2f> dst_pts_(dst_pts, dst_pts + 5);
    cv::Mat M;
    get_transform(landmarks, dst_pts_, M, 0);
    cv::warpAffine(image, out_face, M, cv::Size(face_width, face_height));
}


void FaceAlignment::crop_faces_alignment(cv::Mat &image, vector<cv::Point2f> &landmarks, cv::Mat &out_face, cv::Mat &M,
                                         cv::Mat &Minv) {
    // get transformer matrix
    std::vector<cv::Point2f> dst_pts_(dst_pts, dst_pts + 5);
    get_transform(landmarks, dst_pts_, M, Minv, 0);
    cv::warpAffine(image, out_face, M, cv::Size(face_width, face_height));
}


void FaceAlignment::get_transform(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts, cv::Mat &M,
                                  int method) {
    if (method == 0) {
        M = solve_lstsq(src_pts, dst_pts);
    } else {
        M = cv::estimateAffine2D(src_pts, dst_pts);
    }
}

void FaceAlignment::get_transform(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts, cv::Mat &M, cv::Mat &Minv,
                                  int method) {
    get_transform(src_pts, dst_pts, M);
    cv::invertAffineTransform(M, Minv);
}

cv::Mat FaceAlignment::solve_lstsq(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts) {
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


void FaceAlignment::crop_faces(cv::Mat &image, cv::Rect &rect, cv::Mat &out_face) {
    out_face = image_crop(image, rect);
    cv::resize(out_face, out_face, cv::Size(face_width, face_height));
}
