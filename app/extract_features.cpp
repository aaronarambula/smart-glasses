// ─── extract_features.cpp ────────────────────────────────────────────────────
// Utility to extract and inspect feature vectors from TTC frames.
// Useful for:
//   - Debugging sector binning
//   - Exporting features for external ML training
//   - Validating normalization strategies
//
// Usage:
//   ./extract_features --verbose
//   ./extract_features --test-frame 5
//   ./extract_features --export features.csv --num-frames 100
//

#include "prediction/prediction.h"
#include "sensors/sensors.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <string>

namespace {

[[noreturn]] void usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\nOptions:\n"
        << "  --help               Show this message\n"
        << "  --verbose            Print detailed feature breakdown\n"
        << "  --test-frame N       Extract features from synthetic frame N\n"
        << "  --export FILE        Export features to CSV file\n"
        << "  --num-frames N       Number of frames to generate (default: 50)\n"
        << "\nExample:\n"
        << "  " << prog << " --verbose --test-frame 10\n"
        << "  " << prog << " --export features.csv --num-frames 100\n";
    std::exit(1);
}

void print_feature_vector(const prediction::FeatureVector& feat, size_t frame_id)
{
    std::cout << "\n[Frame " << frame_id << " Features]\n";
    std::cout << "─────────────────────────────────────────────────────\n";

    // Group A: Sector distances
    std::cout << "Sector Distances (Group A):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "  Sector " << i << " (angle=" << (i*45) << "°–" << ((i+1)*45) << "°): ";
        std::cout << std::fixed << std::setprecision(3) << feat.sector_dist(i) << "\n";
    }

    // Group B: Sector TTCs
    std::cout << "\nSector TTCs (Group B):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "  Sector " << i << ": ";
        std::cout << std::fixed << std::setprecision(3) << feat.sector_ttc(i) << "\n";
    }

    // Group C: Named-sector density
    std::cout << "\nNamed-Sector Density (Group C):\n";
    std::cout << "  Forward (0°±30°):    " << std::fixed << std::setprecision(3)
              << feat.density_forward() << "\n";
    std::cout << "  Right (30°–150°):    " << std::fixed << std::setprecision(3)
              << feat.density_right() << "\n";
    std::cout << "  Rear (150°–210°):    " << std::fixed << std::setprecision(3)
              << feat.density_rear() << "\n";
    std::cout << "  Left (210°–330°):    " << std::fixed << std::setprecision(3)
              << feat.density_left() << "\n";

    // Group D: Global statistics
    std::cout << "\nGlobal Statistics (Group D):\n";
    std::cout << "  Max closing speed:   " << std::fixed << std::setprecision(3)
              << feat.max_closing_speed() << "\n";
    std::cout << "  Confirmed tracks:    " << std::fixed << std::setprecision(3)
              << feat.num_confirmed_tracks() << "\n";
    std::cout << "  Min TTC (global):    " << std::fixed << std::setprecision(3)
              << feat.global_min_ttc() << "\n";
    std::cout << "  Occupancy density:   " << std::fixed << std::setprecision(3)
              << feat.local_occ_density() << "\n";
}

void export_features_csv(const std::string& filename,
                         const std::vector<prediction::FeatureVector>& features,
                         const std::vector<prediction::RiskLevel>& labels)
{
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "error: could not open " << filename << " for writing\n";
        std::exit(1);
    }

    // Header: all 24 features + label
    out << "frame_id";
    for (int i = 0; i < 8; ++i) out << ",dist_" << i;
    for (int i = 0; i < 8; ++i) out << ",ttc_" << i;
    out << ",dens_fwd,dens_right,dens_rear,dens_left";
    out << ",max_cs,num_tracks,min_ttc,occ_density";
    out << ",label\n";

    // Data rows
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feat = features[i];
        const auto& label = labels[i];

        out << i;

        // Sector distances
        for (int s = 0; s < 8; ++s) {
            out << "," << std::fixed << std::setprecision(5) << feat.sector_dist(s);
        }

        // Sector TTCs
        for (int s = 0; s < 8; ++s) {
            out << "," << std::fixed << std::setprecision(5) << feat.sector_ttc(s);
        }

        // Named densities
        out << "," << std::fixed << std::setprecision(5) << feat.density_forward()
            << "," << feat.density_right()
            << "," << feat.density_rear()
            << "," << feat.density_left();

        // Global stats
        out << "," << std::fixed << std::setprecision(5) << feat.max_closing_speed()
            << "," << feat.num_confirmed_tracks()
            << "," << feat.global_min_ttc()
            << "," << feat.local_occ_density();

        // Label (0=CLEAR, 1=CAUTION, 2=WARNING, 3=DANGER)
        out << "," << static_cast<int>(label) << "\n";
    }

    out.close();
    std::cout << "✓ Exported " << features.size() << " feature vectors to " << filename << "\n";
}

} // namespace

int main(int argc, char* argv[])
{
    bool verbose = false;
    int test_frame = -1;
    std::string export_file;
    int num_frames = 50;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--test-frame") {
            if (i + 1 >= argc) usage(argv[0]);
            test_frame = std::stoi(argv[++i]);
        } else if (arg == "--export") {
            if (i + 1 >= argc) usage(argv[0]);
            export_file = argv[++i];
        } else if (arg == "--num-frames") {
            if (i + 1 >= argc) usage(argv[0]);
            num_frames = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage(argv[0]);
        }
    }

    std::cout << "═════════════════════════════════════════════════════════\n";
    std::cout << "Feature Extraction Utility\n";
    std::cout << "═════════════════════════════════════════════════════════\n\n";

    // Create a synthetic frame generator (same as simulator uses)
    // For now, we create dummy frames to demonstrate the extraction logic

    std::vector<prediction::FeatureVector> all_features;
    std::vector<prediction::RiskLevel> all_labels;

    std::cout << "Generating " << num_frames << " synthetic frames...\n\n";

    // In a real deployment, these would come from the actual sensor pipeline.
    // For demonstration, we're showing the feature structure.

    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        // Create a dummy feature vector for demonstration
        prediction::FeatureVector feat;

        // Synthetic data: vary sector distances and TTCs
        for (int s = 0; s < 8; ++s) {
            // Distance: simulate obstacles at varying ranges
            float dist = 0.5f + (s % 3) * 0.2f;  // 0.5–0.9 range
            feat.sector_dist(s) = std::min(dist, 1.0f);

            // TTC: derive from distance (closer = lower TTC)
            float ttc = std::max(0.0f, 1.0f - dist * 0.5f);
            feat.sector_ttc(s) = ttc;
        }

        // Density: alternate between occupied and empty
        feat.density_forward() = (frame_idx % 2 == 0) ? 0.5f : 0.2f;
        feat.density_right()   = (frame_idx % 3 == 0) ? 0.3f : 0.1f;
        feat.density_rear()    = (frame_idx % 4 == 0) ? 0.4f : 0.0f;
        feat.density_left()    = (frame_idx % 5 == 0) ? 0.2f : 0.1f;

        // Global stats
        feat.max_closing_speed() = (frame_idx % 7) * 0.1f;
        feat.num_confirmed_tracks() = (frame_idx % 5) / 5.0f;
        feat.global_min_ttc() = (frame_idx % 10) * 0.05f;
        feat.local_occ_density() = (frame_idx % 8) * 0.08f;

        // Assign label based on overall risk (synthetic pseudo-label)
        prediction::RiskLevel label = prediction::RiskLevel::CLEAR;
        float avg_dist = (feat.density_forward() + feat.density_right() +
                          feat.density_rear() + feat.density_left()) / 4.0f;
        if (avg_dist > 0.4f) {
            label = prediction::RiskLevel::DANGER;
        } else if (avg_dist > 0.3f) {
            label = prediction::RiskLevel::WARNING;
        } else if (avg_dist > 0.1f) {
            label = prediction::RiskLevel::CAUTION;
        }

        all_features.push_back(feat);
        all_labels.push_back(label);

        if (verbose || frame_idx == test_frame) {
            print_feature_vector(feat, frame_idx);
        }
    }

    // Export to CSV if requested
    if (!export_file.empty()) {
        std::cout << "\nExporting to CSV...\n";
        export_features_csv(export_file, all_features, all_labels);
    }

    std::cout << "\n═════════════════════════════════════════════════════════\n";
    std::cout << "Feature structure (24 total):\n"
              << "  [0..7]   Sector distances (8 × 45° sectors)\n"
              << "  [8..15]  Sector TTC values\n"
              << "  [16..19] Named-sector densities (forward, right, rear, left)\n"
              << "  [20..23] Global stats (max_cs, num_tracks, min_ttc, occ_density)\n\n"
              << "Use --export to save features to CSV for external ML training.\n";

    return 0;
}
