#include "./infer/CRNNRecognizer.h"
#include "./infer/DBDetector.h"
#include "ncnn/simpleocv.h"
#include "ylt/easylog.hpp"
#include "ylt/easylog/record.hpp"
#include <filesystem>

int main() {
  easylog::init_log(easylog::Severity::INFO);

  breeze_ocr::DBDetector detector;
  breeze_ocr::CRNNRecognizer recognizer;

  auto cwd = std::filesystem::current_path();

  detector.loadModel(cwd.string() + "./models/PP-OCRv5_mobile_det_infer");
  recognizer.loadModel(cwd.string() + "./models/PP_OCRv4_mobile_rec_infer");
  if (auto res = recognizer.loadCharset(cwd.string() + "./b.txt"); !res) {
    ELOGFMT(ERROR, "Failed to load charset: {}", res.error());
    return -1;
  }

  cv::Mat image = cv::imread(cwd.string() + "/ocr.png", cv::IMREAD_COLOR);
  if (image.empty()) {
    ELOGFMT(ERROR, "Failed to load image");
    return -1;
  }

  auto detection_result = detector.detect(image);
  if (detection_result.boxes.empty()) {
    ELOGFMT(WARNING, "No text detected in the image");
    return 0;
  }

  ELOGFMT(INFO, "Detected {} text boxes", detection_result.boxes.size());
  detection_result.visualizeAndSave(image, "detected_boxes.png");

  auto imageVisualized = image.clone();

  for (auto &&box : detection_result.getBoxes(image)) {
    auto recognition_result = recognizer.recognize(box.image);
    auto points = box.box.getVertices();

    for (size_t i = 0; i < points.size(); ++i) {
      cv::Point p1(points[i].x / 2, points[i].y / 2);
      cv::Point p2(points[(i + 1) % points.size()].x / 2,
                   points[(i + 1) % points.size()].y / 2);
      cv::line(imageVisualized, p1, p2, cv::Scalar(0, 255, 0), 2);
    }

    cv::putText(imageVisualized, recognition_result.text,
                cv::Point(points[0].x / 2, points[0].y / 2 - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
  }

  cv::imwrite("recognized_text.png", imageVisualized);

  return 0;
}