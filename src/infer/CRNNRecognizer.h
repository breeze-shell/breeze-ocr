#pragma once
#include "ncnn/net.h"
#include "ncnn/simpleocv.h"
#include <expected>
#include <memory>
#include <string_view>
#include <vector>
#include <string>

#include "CVUtils.h"

namespace breeze_ocr {
struct RecognitionResult {
  std::string text;
  float confidence;
  std::vector<int> labels;
};

class CRNNRecognizer {
public:
  CRNNRecognizer() = default;
  ~CRNNRecognizer() = default;

  void loadModel(std::string_view model_path);

  RecognitionResult recognize(const cv::Mat &image);

  std::expected<void, std::string> loadCharset(std::string_view charset_path);

  std::vector<std::string> charset;
private:
  std::unique_ptr<ncnn::Net> net;
  int resize_long = 320;
};
} // namespace breeze_ocr