//
// Created by 390737991@qq.com on 2018/6/3.
//

#include <debug.h>
#include "face_alignment.h"
#include "transform.h"

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
    get_transform(landmarks, dst_pts_, M, 1);
    cv::warpAffine(image, out_face, M, cv::Size(face_width, face_height));
}


void FaceAlignment::crop_faces_alignment(cv::Mat &image, vector<cv::Point2f> &landmarks, cv::Mat &out_face, cv::Mat &M,
                                         cv::Mat &Minv) {
    // get transformer matrix
    std::vector<cv::Point2f> dst_pts_(dst_pts, dst_pts + 5);
    get_transform(landmarks, dst_pts_, M, Minv, 1);
    cv::warpAffine(image, out_face, M, cv::Size(face_width, face_height));
}


void FaceAlignment::crop_faces(cv::Mat &image, cv::Rect &rect, cv::Mat &out_face) {
    out_face = image_crop(image, rect);
    cv::resize(out_face, out_face, cv::Size(face_width, face_height));
}
