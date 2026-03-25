#include "sensors/camera_classifier.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace sensors {

namespace {
std::vector<std::string> default_class_names()
{
    return {"person", "chair", "box", "unknown"};
}
}

CameraObjectClassifier::CameraObjectClassifier(std::vector<std::string> class_names)
    : class_names_(class_names.empty() ? default_class_names() : std::move(class_names))
{
    build_model();
}

void CameraObjectClassifier::build_model()
{
    model_ = autograd::Sequential{};
    model_.add<autograd::Conv2d>(1, 8, 5, 2, 42)
          .add<autograd::ReLU>()
          .add<autograd::Conv2d>(8, 16, 3, 2, 43)
          .add<autograd::ReLU>()
          .add<autograd::Flatten>()
          .add<autograd::Linear>(16 * 14 * 14, 64, 44)
          .add<autograd::ReLU>()
          .add<autograd::Linear>(64, class_names_.size(), 45);
    optimizer_ = std::make_unique<autograd::Adam>(model_.parameters(), 1e-3f);
}

autograd::TensorPtr CameraObjectClassifier::preprocess(const cv::Mat& image) const
{
#ifndef HAVE_OPENCV
    (void)image;
    throw std::runtime_error("CameraObjectClassifier requires OpenCV");
#else
    if (image.empty()) {
        throw std::runtime_error("CameraObjectClassifier::preprocess received empty image");
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(INPUT_SIZE, INPUT_SIZE), 0.0, 0.0, cv::INTER_AREA);

    std::vector<float> pixels(INPUT_SIZE * INPUT_SIZE, 0.0f);
    for (int y = 0; y < INPUT_SIZE; ++y) {
        for (int x = 0; x < INPUT_SIZE; ++x) {
            pixels[static_cast<size_t>(y * INPUT_SIZE + x)] =
                static_cast<float>(resized.at<unsigned char>(y, x)) / 255.0f;
        }
    }

    return autograd::make_tensor(std::move(pixels),
                                 std::vector<size_t>{1, 1, INPUT_SIZE, INPUT_SIZE},
                                 /*requires_grad=*/false);
#endif
}

CameraClassification CameraObjectClassifier::classify(const cv::Mat& bgr_or_gray) const
{
    CameraClassification out;
#ifndef HAVE_OPENCV
    (void)bgr_or_gray;
    return out;
#else
    if (!ready_) return out;

    autograd::NoGradGuard no_grad;
    auto x = preprocess(bgr_or_gray);
    auto logits = model_.forward(x);
    auto probs = logits->softmax();

    size_t best_idx = 0;
    float best_prob = probs->data[0];
    for (size_t i = 1; i < probs->data.size(); ++i) {
        if (probs->data[i] > best_prob) {
            best_prob = probs->data[i];
            best_idx = i;
        }
    }

    out.label = best_idx < class_names_.size() ? class_names_[best_idx] : "unknown";
    out.confidence = best_prob;
    out.valid = true;
    return out;
#endif
}

std::vector<CameraObjectClassifier::Sample>
CameraObjectClassifier::load_dataset(const std::string& dataset_root) const
{
    std::vector<Sample> samples;
#ifndef HAVE_OPENCV
    (void)dataset_root;
    return samples;
#else
    namespace fs = std::filesystem;
    for (size_t class_idx = 0; class_idx < class_names_.size(); ++class_idx) {
        fs::path class_dir = fs::path(dataset_root) / class_names_[class_idx];
        if (!fs::exists(class_dir) || !fs::is_directory(class_dir)) continue;

        for (const auto& entry : fs::directory_iterator(class_dir)) {
            if (!entry.is_regular_file()) continue;
            const auto ext = entry.path().extension().string();
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (img.empty()) continue;
            auto t = preprocess(img);
            samples.push_back({t->data, static_cast<int>(class_idx)});
        }
    }
    return samples;
#endif
}

bool CameraObjectClassifier::train_on_directory(const std::string& dataset_root,
                                                int epochs,
                                                float learning_rate,
                                                float& final_loss_out)
{
    final_loss_out = 0.0f;
#ifndef HAVE_OPENCV
    (void)dataset_root;
    (void)epochs;
    (void)learning_rate;
    return false;
#else
    auto samples = load_dataset(dataset_root);
    if (samples.empty()) return false;

    build_model();
    optimizer_->set_lr(learning_rate);

    std::mt19937 rng(42);
    for (int epoch = 0; epoch < std::max(1, epochs); ++epoch) {
        std::shuffle(samples.begin(), samples.end(), rng);
        float epoch_loss = 0.0f;

        for (const auto& sample : samples) {
            auto x = autograd::make_tensor(sample.pixels,
                                           std::vector<size_t>{1, 1, INPUT_SIZE, INPUT_SIZE},
                                           /*requires_grad=*/false);
            optimizer_->zero_grad();
            auto logits = model_.forward(x);
            auto loss = autograd::Tensor::cross_entropy(logits, {sample.label});
            epoch_loss += loss->data[0];
            loss->backward();
            optimizer_->step(1.0f);
        }

        final_loss_out = epoch_loss / static_cast<float>(samples.size());
    }

    ready_ = true;
    return true;
#endif
}

void CameraObjectClassifier::save_weights(const std::string& path) const
{
    if (path.empty()) return;

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        throw std::runtime_error("CameraObjectClassifier::save_weights: cannot open '" + path + "'");
    }

    auto write_u32 = [&](uint32_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };

    write_u32(CHECKPOINT_MAGIC);
    write_u32(CHECKPOINT_VERSION);
    write_u32(static_cast<uint32_t>(class_names_.size()));
    write_u32(static_cast<uint32_t>(INPUT_SIZE));

    for (const auto& name : class_names_) {
        write_u32(static_cast<uint32_t>(name.size()));
        f.write(name.data(), static_cast<std::streamsize>(name.size()));
    }

    auto params = const_cast<autograd::Sequential&>(model_).parameters();
    write_u32(static_cast<uint32_t>(params.size()));
    for (const auto& p : params) {
        write_u32(static_cast<uint32_t>(p->shape.size()));
        for (size_t dim : p->shape) {
            write_u32(static_cast<uint32_t>(dim));
        }
        write_u32(static_cast<uint32_t>(p->numel()));
        f.write(reinterpret_cast<const char*>(p->data.data()),
                static_cast<std::streamsize>(p->numel() * sizeof(float)));
    }

    if (!f) {
        throw std::runtime_error("CameraObjectClassifier::save_weights: write error on '" + path + "'");
    }
}

bool CameraObjectClassifier::load_weights(const std::string& path)
{
    if (path.empty()) return false;

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };

    const uint32_t magic = read_u32();
    const uint32_t version = read_u32();
    if (magic != CHECKPOINT_MAGIC || version != CHECKPOINT_VERSION) {
        throw std::runtime_error("CameraObjectClassifier::load_weights: bad checkpoint '" + path + "'");
    }

    const uint32_t num_classes = read_u32();
    const uint32_t input_size = read_u32();
    if (input_size != static_cast<uint32_t>(INPUT_SIZE)) {
        throw std::runtime_error("CameraObjectClassifier::load_weights: INPUT_SIZE mismatch");
    }

    std::vector<std::string> loaded_names;
    loaded_names.reserve(num_classes);
    for (uint32_t i = 0; i < num_classes; ++i) {
        const uint32_t len = read_u32();
        std::string name(len, '\0');
        f.read(name.data(), static_cast<std::streamsize>(len));
        loaded_names.push_back(std::move(name));
    }

    class_names_ = std::move(loaded_names);
    build_model();

    const uint32_t num_params = read_u32();
    auto params = model_.parameters();
    if (num_params != params.size()) {
        throw std::runtime_error("CameraObjectClassifier::load_weights: param count mismatch");
    }

    for (uint32_t i = 0; i < num_params; ++i) {
        const uint32_t ndim = read_u32();
        std::vector<size_t> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            shape[d] = read_u32();
        }
        const uint32_t numel = read_u32();

        if (shape != params[i]->shape || numel != params[i]->numel()) {
            throw std::runtime_error("CameraObjectClassifier::load_weights: parameter shape mismatch");
        }

        f.read(reinterpret_cast<char*>(params[i]->data.data()),
               static_cast<std::streamsize>(numel * sizeof(float)));
        params[i]->grad.clear();
    }

    ready_ = true;
    return true;
}

} // namespace sensors
