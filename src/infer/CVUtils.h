#pragma once
#include <cmath>
#include <numbers>

#include "ncnn/simpleocv.h"

namespace breeze_ocr {
struct RotatedRect {
  cv::Point2f center;
  cv::Size2f size;
  float angle;

  RotatedRect(const cv::Point2f &c, const cv::Size2f &s, float a)
      : center(c), size(s), angle(a) {}

  inline std::vector<cv::Point2f> getVertices() const {
    return {
        cv::Point2f(center.x -
                        size.width / 2 * cos(angle * std::numbers::pi / 180) -
                        size.height / 2 * sin(angle * std::numbers::pi / 180),
                    center.y -
                        size.width / 2 * sin(angle * std::numbers::pi / 180) +
                        size.height / 2 * cos(angle * std::numbers::pi / 180)),
        cv::Point2f(center.x +
                        size.width / 2 * cos(angle * std::numbers::pi / 180) -
                        size.height / 2 * sin(angle * std::numbers::pi / 180),
                    center.y +
                        size.width / 2 * sin(angle * std::numbers::pi / 180) +
                        size.height / 2 * cos(angle * std::numbers::pi / 180)),
        cv::Point2f(center.x +
                        size.width / 2 * cos(angle * std::numbers::pi / 180) +
                        size.height / 2 * sin(angle * std::numbers::pi / 180),
                    center.y +
                        size.width / 2 * sin(angle * std::numbers::pi / 180) -
                        size.height / 2 * cos(angle * std::numbers::pi / 180)),
        cv::Point2f(center.x -
                        size.width / 2 * cos(angle * std::numbers::pi / 180) +
                        size.height / 2 * sin(angle * std::numbers::pi / 180),
                    center.y -
                        size.width / 2 * sin(angle * std::numbers::pi / 180) -
                        size.height / 2 * cos(angle * std::numbers::pi / 180))};
  }
};

} // namespace breeze_ocr