#include "CRNNRecognizer.h"
#include "ModelLoader.h"
#include "ylt/easylog.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace breeze_ocr {

std::vector<std::string> load_charset(const std::string &model_dir) {
  std::vector<std::string> charset;
  charset.push_back("");

  std::filesystem::path charset_path =
      std::filesystem::path(model_dir) / "charset.txt";
  std::filesystem::path yml_path =
      std::filesystem::path(model_dir) / "inference.yml";

  if (std::filesystem::exists(charset_path)) {
    std::ifstream file(charset_path);
    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty()) {
        charset.push_back(line);
      }
    }
    ELOGFMT(INFO, "Loaded {} characters from charset file", charset.size() - 1);
    return charset;
  }

  if (std::filesystem::exists(yml_path)) {
    std::ifstream file(yml_path);
    std::string line;
    bool in_charset_section = false;

    while (std::getline(file, line)) {
      if (line.find("character_dict:") != std::string::npos) {
        in_charset_section = true;
        continue;
      }

      if (in_charset_section) {

        size_t pos = line.find("- ");
        if (pos != std::string::npos) {
          std::string ch = line.substr(pos + 2);
          charset.push_back(ch);
        } else if (line.find("- ") == std::string::npos && !line.empty() &&
                   line[0] != ' ') {

          break;
        }
      }
    }

    ELOGFMT(INFO, "Loaded {} characters from yml file", charset.size() - 1);
    return charset;
  }

  ELOGFMT(WARNING, "No charset file found, using default charset");
  for (char c = '0'; c <= '9'; ++c) {
    charset.push_back(std::string(1, c));
  }
  for (char c = 'A'; c <= 'Z'; ++c) {
    charset.push_back(std::string(1, c));
  }
  for (char c = 'a'; c <= 'z'; ++c) {
    charset.push_back(std::string(1, c));
  }

  return charset;
}

void CRNNRecognizer::loadModel(std::string_view model_path) {
  net = breeze_ocr::loadModel(model_path);
  net->opt.use_vulkan_compute = true;

  std::filesystem::path path(model_path);
  std::string model_dir = path.parent_path().string();

  charset = load_charset(model_dir);
  ELOGFMT(INFO, "CRNN recognizer loaded with {} characters",
          charset.size() - 1);
}

uchar &matAt(const cv::Mat &mat, int x, int y, int z) {
  static uchar dummy = 0;
  if (x < 0 || x >= mat.cols || y < 0 || y >= mat.rows || z < 0 ||
      z >= mat.channels()) {
    return dummy;
  }
  return mat.data[y * mat.cols * mat.channels() + x * mat.channels() + z];
}

RecognitionResult CRNNRecognizer::recognize(const cv::Mat &image) {
  RecognitionResult result{
      .text = "", .confidence = 0.0f, .labels = std::vector<int>()};

  if (image.empty() || !net) {
    return result;
  }

  cv::Mat resized;

  // max: width 320 height 48

  int target_w = resize_long;
  int target_h = 48;
  if (image.cols > image.rows) {
    target_w = std::min(resize_long, image.cols);
    target_h = static_cast<int>(static_cast<float>(image.rows) / image.cols *
                                resize_long);
  } else {
    target_h = std::min(resize_long, image.rows);
    target_w = static_cast<int>(static_cast<float>(image.cols) / image.rows *
                                resize_long);
  }

  float scale = std::min(static_cast<float>(target_w) / image.cols,
                         static_cast<float>(target_h) / image.rows);

  resized.create(target_h, target_w, CV_8UC3);

  for (int y = 0; y < target_h; ++y) {
    for (int x = 0; x < target_w; ++x) {
      int src_x = static_cast<int>(x / scale);
      int src_y =
          static_cast<int>(y / scale - (target_h - image.rows) * scale / 2.0f);
      if (src_x < image.cols && src_y < image.rows && src_x >= 0 &&
          src_y >= 0) {
        matAt(resized, x, y, 0) = matAt(image, src_x, src_y, 0);
        matAt(resized, x, y, 1) = matAt(image, src_x, src_y, 1);
        matAt(resized, x, y, 2) = matAt(image, src_x, src_y, 2);
      } else {
        matAt(resized, x, y, 0) = 255;
        matAt(resized, x, y, 1) = 255;
        matAt(resized, x, y, 2) = 255;
      }
    }
  }

  ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR,
                                        resized.cols, resized.rows);

  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = net->create_extractor();

  ex.input("in0", in);

  ncnn::Mat out;
  ncnn::Mat out_blob;
  ex.extract("out0", out);

  std::string text;
  float confidence = 0.0f;
  int prev_index = -1;

  if (charset.empty()) {
    ELOGFMT(ERROR, "Charset is empty");
    return result; // No charset loaded, cannot recognize
  }

  float total_conf = 0.0f;
  int valid_char_count = 0;

  ELOGFMT(INFO, "Output size: {}x{}", out.w, out.h);

  for (int t = 0; t < out.h; t++) {

    int max_idx = -1;
    float max_prob = -1;

    for (int i = 1; i < out.w - 1; i++) {
      float prob = out.row(t)[i];
      if (prob > max_prob) {
        max_prob = prob;
        max_idx = i;
      }
    }

    if (max_prob > 0.0001) {
      if (max_idx != 0 && max_idx != prev_index) {
        if (max_idx < charset.size()) {
          result.labels.push_back(max_idx);
          text += charset[max_idx - 1];
          total_conf += max_prob;
          valid_char_count++;
        }
      }
    }

    prev_index = max_idx;
  }

  if (valid_char_count > 0) {
    confidence = total_conf / valid_char_count;
  }

  result.text = text;
  result.confidence = confidence;

  return result;
}
std::expected<void, std::string>
CRNNRecognizer::loadCharset(std::string_view charset_path) {
  std::filesystem::path path(charset_path);
  if (!std::filesystem::exists(path)) {
    return std::unexpected("Charset file not found: " + path.string());
  }
  std::ifstream file(path);
  if (!file.is_open()) {
    return std::unexpected("Failed to open charset file: " + path.string());
  }

  charset.clear();

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue; // Skip empty lines and comments
    }
    // Remove leading and trailing whitespace
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);
    if (!line.empty()) {
      charset.push_back(line);
    }
  }

  file.close();
  if (charset.empty()) {
    return std::unexpected("Charset file is empty: " + path.string());
  }

  ELOGFMT(INFO, "Loaded {} characters from charset file", charset.size());

  return {};
}
} // namespace breeze_ocr