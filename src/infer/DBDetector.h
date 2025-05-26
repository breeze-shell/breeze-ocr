#pragma once
#include "ncnn/net.h"
#include "ncnn/simpleocv.h"
#include <memory>
#include <string_view>

#include "CVUtils.h"

namespace breeze_ocr {

struct TextDetectionResult {
  std::vector<RotatedRect> boxes;
  int origin_width, origin_height;

  void visualizeAndSave(const cv::Mat &image, const std::string &output_path);

  struct TextDetectionBox {
    cv::Mat image;
    RotatedRect box;
  };

  std::vector<TextDetectionBox> getBoxes(const cv::Mat &image) const;
};

class DBDetector {
public:
  DBDetector() = default;
  ~DBDetector() = default;

  void loadModel(std::string_view model_path);

  TextDetectionResult detect(const cv::Mat &image);

private:
  std::unique_ptr<ncnn::Net> net;
  float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
  float norm_vals[3] = {1.0f / 0.229f / 255.f, 1.0f / 0.224f / 255.f,
                        1.0f / 0.225f / 255.f};
  int resize_long = 960;
};
} // namespace breeze_ocr