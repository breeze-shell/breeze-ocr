#include "DBDetector.h"
#include "ModelLoader.h"
#include "ncnn/simpleocv.h"
#include "ylt/easylog.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace breeze_ocr {

uchar &matAt(const cv::Mat &mat, int x, int y) {
  if (x < 0 || x >= mat.cols || y < 0 || y >= mat.rows) {
    return mat.data[0];
  }
  return mat.data[y * mat.cols + x];
}

std::vector<std::vector<cv::Point>> findContours(const cv::Mat &binary) {
  std::vector<std::vector<cv::Point>> contours;
  int height = binary.rows;
  int width = binary.cols;
  cv::Mat visited(height, width, CV_8UC1);

  visited = cv::Scalar(0);

  int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {

      if (binary.data[y * width + x] == 255 &&
          visited.data[y * width + x] == 0) {

        std::vector<cv::Point> contour;

        std::vector<cv::Point> stack;
        stack.push_back(cv::Point(x, y));
        visited.data[y * width + x] = 1;

        while (!stack.empty()) {
          cv::Point p = stack.back();
          stack.pop_back();
          contour.push_back(p);

          for (int i = 0; i < 8; ++i) {
            int nx = p.x + dx[i];
            int ny = p.y + dy[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {

              if (binary.data[ny * width + nx] == 255 &&
                  visited.data[ny * width + nx] == 0) {
                stack.push_back(cv::Point(nx, ny));
                visited.data[ny * width + nx] = 1;
              }
            }
          }
        }

        if (contour.size() > 10) {
          contours.push_back(contour);
        }
      }
    }
  }
  return contours;
}

std::optional<RotatedRect>
findRotatedRect(const std::vector<cv::Point> &contour) {
  if (contour.size() < 5) {
    return {};
  }

  RotatedRect rect{cv::Point2f(0, 0), cv::Size2f(0, 0), 0.0f};

  double m00 = 0.0, m10 = 0.0, m01 = 0.0, m20 = 0.0, m11 = 0.0, m02 = 0.0;

  for (const auto &point : contour) {
    m00 += 1.0;
    m10 += point.x;
    m01 += point.y;
  }

  double centerX = m10 / m00;
  double centerY = m01 / m00;
  rect.center = cv::Point2f(centerX, centerY);

  for (const auto &point : contour) {
    double dx = point.x - centerX;
    double dy = point.y - centerY;
    m20 += dx * dx;
    m11 += dx * dy;
    m02 += dy * dy;
  }

  double mu20 = m20 / m00;
  double mu11 = m11 / m00;
  double mu02 = m02 / m00;

  double theta;
  if (mu11 == 0 && mu20 == mu02) {
    theta = 0;
  } else {
    theta = 0.5 * std::atan2(2 * mu11, mu20 - mu02);
  }

  rect.angle = theta * 180.0 / std::numbers::pi;
  double sinTheta = std::sin(theta);
  double cosTheta = std::cos(theta);
  double maxX = 0, maxY = 0;

  for (const auto &point : contour) {
    double x = (point.x - centerX) * cosTheta + (point.y - centerY) * sinTheta;
    double y = -(point.x - centerX) * sinTheta + (point.y - centerY) * cosTheta;

    maxX = std::max(maxX, std::abs(x));
    maxY = std::max(maxY, std::abs(y));
  }

  rect.size = cv::Size2f(2 * maxX, 2 * maxY);
  rect.angle = std::fmod(rect.angle + 180.0f, 180.0f);
  if (rect.angle > 90.0f) {
    rect.angle -= 180.0f;
  }
  return rect;
}

void DBDetector::loadModel(std::string_view model_path) {
  net = breeze_ocr::loadModel(model_path);

  net->opt.use_vulkan_compute = true;
}

TextDetectionResult DBDetector::detect(const cv::Mat &image) {
  std::vector<RotatedRect> boxes;

  if (image.empty() || !net) {
    return {boxes, 0, 0};
  }

  cv::Mat resized;
  float scale = 1.0f;

  int max_side = std::max(image.cols, image.rows);
  if (max_side > resize_long) {
    scale = static_cast<float>(resize_long) / max_side;
    cv::resize(image, resized,
               cv::Size(int(image.cols * scale), int(image.rows * scale)));
  } else {
    resized = image.clone();
  }
  ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR,
                                        resized.cols, resized.rows);

  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = net->create_extractor();
  ex.input("in0", in);

  ncnn::Mat out;
  ex.extract("out0", out);

  float *prob_data = (float *)out.data;
  int height = out.h;
  int width = out.w;

  cv::Mat binary = cv::Mat(height, width, CV_8UC1);
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      float prob = prob_data[h * width + w];
      if (prob > 0.3f) {
        binary.data[h * width + w] = 255;
      } else {
        binary.data[h * width + w] = 0;
      }
    }
  }

  std::vector<std::vector<cv::Point>> contours = findContours(binary);

  ELOGFMT(DEBUG, "Found {} contours", contours.size());

  for (const auto &contour : contours) {
    if (contour.size() < 4) {
      continue;
    }

    if (auto rect = findRotatedRect(contour)) {
      RotatedRect box = *rect;
      box.center.x = box.center.x / scale;
      box.center.y = box.center.y / scale;
      box.size.width = box.size.width / scale;
      box.size.height = box.size.height / scale;

      box.size.width *= 1.05;
      box.size.height *= 1.9;

      boxes.push_back(box);
    }
  }

  return {boxes, resized.cols, resized.rows};
}

void TextDetectionResult::visualizeAndSave(const cv::Mat &image,
                                           const std::string &output_path) {
  cv::Mat resized = image.clone();
  if (resized.empty()) {
    ELOGFMT(ERROR, "Failed to clone image for visualization");
    return;
  }
  cv::resize(resized, resized, cv::Size(resized.cols / 2, resized.rows / 2));

  for (const auto &box : boxes) {
    auto vertices = box.getVertices();

    for (size_t i = 0; i < vertices.size(); ++i) {
      cv::Point p1(vertices[i].x / 2, vertices[i].y / 2);
      cv::Point p2(vertices[(i + 1) % vertices.size()].x / 2,
                   vertices[(i + 1) % vertices.size()].y / 2);
      cv::line(resized, p1, p2, cv::Scalar(0, 255, 0), 2);
    }
  }

  cv::imwrite(output_path, resized);
}

cv::Point operator*(const cv::Point &p, float scale) {
  return cv::Point(static_cast<int>(p.x * scale),
                   static_cast<int>(p.y * scale));
}
cv::Point operator+(const cv::Point &a, const cv::Point &b) {
  return cv::Point(a.x + b.x, a.y + b.y);
}

std::vector<TextDetectionResult::TextDetectionBox>
TextDetectionResult::getBoxes(const cv::Mat &image) const {
  std::vector<TextDetectionBox> result;

  for (const auto &box : boxes) {
    auto vertices = box.getVertices();
    if (vertices.size() != 4)
      continue;

    float minX = vertices[0].x, maxX = vertices[0].x;
    float minY = vertices[0].y, maxY = vertices[0].y;

    for (const auto &vertex : vertices) {
      minX = std::min(minX, vertex.x);
      maxX = std::max(maxX, vertex.x);
      minY = std::min(minY, vertex.y);
      maxY = std::max(maxY, vertex.y);
    }

    int width = static_cast<int>(maxX - minX);
    int height = static_cast<int>(maxY - minY);

    if (width <= 0 || height <= 0)
      continue;

    cv::Mat rectified(height, width, CV_8UC3);

    // Manual perspective transformation
    // Order vertices: top-left, top-right, bottom-right, bottom-left
    std::vector<cv::Point2f> srcPts = vertices;
    std::vector<cv::Point2f> dstPts = {
        cv::Point2f(0, 0), cv::Point2f(width - 1, 0),
        cv::Point2f(width - 1, height - 1), cv::Point2f(0, height - 1)};

    // Sort vertices to correct order
    std::sort(
        srcPts.begin(), srcPts.end(),
        [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });

    // Top two points
    if (srcPts[0].x > srcPts[1].x)
      std::swap(srcPts[0], srcPts[1]);
    // Bottom two points
    if (srcPts[2].x > srcPts[3].x)
      std::swap(srcPts[2], srcPts[3]);

    std::vector<cv::Point2f> orderedSrc = {srcPts[0], srcPts[1], srcPts[3],
                                           srcPts[2]};

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float u = static_cast<float>(x) / (width - 1);
        float v = static_cast<float>(y) / (height - 1);

        cv::Point2f p1 = orderedSrc[0] * (1 - u) + orderedSrc[1] * u;
        cv::Point2f p2 = orderedSrc[3] * (1 - u) + orderedSrc[2] * u;
        cv::Point2f srcPoint = p1 * (1 - v) + p2 * v;

        int srcX = static_cast<int>(std::round(srcPoint.x));
        int srcY = static_cast<int>(std::round(srcPoint.y));

        if (srcX >= 0 && srcX < image.cols && srcY >= 0 && srcY < image.rows) {
          for (int i = 0; i < rectified.channels(); i++)
            rectified.data[rectified.cols * y * rectified.channels() +
                           x * rectified.channels() + i] =
                image.data[srcY * image.cols * image.channels() +
                           srcX * image.channels() + i];
        }
      }
    }

    result.emplace_back(TextDetectionBox{.image = rectified, .box = box});
  }

  return result;
}
} // namespace breeze_ocr
