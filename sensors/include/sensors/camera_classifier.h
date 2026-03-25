#pragma once

#include "autograd/autograd.h"

#include <memory>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace sensors {

struct CameraClassification {
    std::string label = "unknown";
    float confidence = 0.0f;
    bool  valid = false;
};

class CameraObjectClassifier {
public:
    static constexpr uint32_t CHECKPOINT_MAGIC = 0x434E4E41; // "ANNC"
    static constexpr uint32_t CHECKPOINT_VERSION = 1;
    static constexpr int INPUT_SIZE = 64;

    explicit CameraObjectClassifier(std::vector<std::string> class_names = {});

    CameraClassification classify(const cv::Mat& bgr_or_gray) const;

    bool train_on_directory(const std::string& dataset_root,
                            int epochs,
                            float learning_rate,
                            float& final_loss_out);

    bool load_weights(const std::string& path);
    void save_weights(const std::string& path) const;

    bool is_ready() const { return ready_; }
    const std::vector<std::string>& class_names() const { return class_names_; }

private:
    struct Sample {
        std::vector<float> pixels;
        int label = 0;
    };

    void build_model();
    autograd::TensorPtr preprocess(const cv::Mat& image) const;
    std::vector<Sample> load_dataset(const std::string& dataset_root) const;

    std::vector<std::string>         class_names_;
    autograd::Sequential             model_;
    std::unique_ptr<autograd::Adam>  optimizer_;
    bool                             ready_ = false;
};

} // namespace sensors
