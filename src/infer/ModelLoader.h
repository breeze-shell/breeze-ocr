#pragma once
#include "ncnn/net.h"
#include <filesystem>
#include <memory>
#include <string_view>


namespace breeze_ocr {
static std::unique_ptr<ncnn::Net> loadModel(std::string_view model_path) {
  std::filesystem::path param_path(model_path);
  std::filesystem::path bin_path(model_path);

  // Try xx.param and xx.bin
  if (param_path.extension() != ".param") {
    param_path.replace_extension(".param");
  }
  if (bin_path.extension() != ".bin") {
    bin_path.replace_extension(".bin");
  }

  auto net = std::make_unique<ncnn::Net>();
  net->opt.use_vulkan_compute = true;

  auto param_paths = {param_path, param_path.replace_extension(".param"),
                      param_path.replace_extension(".ncnn.param")};

  auto bin_paths = {bin_path, bin_path.replace_extension(".bin"),
                    bin_path.replace_extension(".ncnn.bin")};

  for (const auto &p : param_paths) {
    if (std::filesystem::exists(p)) {
      net->load_param(p.string().c_str());
      goto load_bin;
    }
  }

  throw std::runtime_error("Parameter file not found: " + param_path.string());
load_bin:
  for (const auto &b : bin_paths) {
    if (std::filesystem::exists(b)) {
      net->load_model(b.string().c_str());
      return net;
    }
  }

  throw std::runtime_error("Model file not found: " + bin_path.string());
}
} // namespace breeze_ocr