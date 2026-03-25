#include "sensors/camera_classifier.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
[[noreturn]] void usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " --data DATASET_DIR [options]\n"
        << "  --data DIR       dataset root with subdirs per class\n"
        << "  --epochs N       training epochs (default: 8)\n"
        << "  --lr FLOAT       learning rate (default: 0.001)\n"
        << "  --out PATH       checkpoint path (default: aaronnet_camera_cls.bin)\n";
    std::exit(1);
}
}

int main(int argc, char* argv[])
{
    std::string data_dir;
    std::string out_path = "aaronnet_camera_cls.bin";
    int epochs = 8;
    float lr = 1e-3f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) usage(flag);
            return std::string(argv[++i]);
        };

        if (arg == "--data") {
            data_dir = next("--data");
        } else if (arg == "--epochs") {
            epochs = std::stoi(next("--epochs"));
        } else if (arg == "--lr") {
            lr = std::stof(next("--lr"));
        } else if (arg == "--out") {
            out_path = next("--out");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage(argv[0]);
        }
    }

    if (data_dir.empty()) usage(argv[0]);

    sensors::CameraObjectClassifier classifier;
    float final_loss = 0.0f;
    if (!classifier.train_on_directory(data_dir, epochs, lr, final_loss)) {
        std::cerr << "Training failed. Ensure OpenCV is installed and the dataset has images under class subdirectories.\n";
        return 1;
    }

    classifier.save_weights(out_path);

    std::cout << "Saved CNN checkpoint to " << out_path
              << " (final_loss=" << final_loss << ")\n";
    std::cout << "Classes:";
    for (const auto& name : classifier.class_names()) {
        std::cout << " " << name;
    }
    std::cout << "\n";
    return 0;
}
