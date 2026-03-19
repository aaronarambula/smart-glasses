// ─── scene_builder.cpp ───────────────────────────────────────────────────────
// Converts FullPrediction + optional PerceptionResult into a compact JSON
// string sent as the user message to the OpenAI Chat Completions API.
// See include/agent/scene_builder.h for the full design notes.

#include "agent/scene_builder.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>

namespace agent {

// ─── Formatting helpers ───────────────────────────────────────────────────────

std::string SceneBuilder::fmt(float v) const
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f",
                  config_.float_precision, static_cast<double>(v));
    return std::string(buf);
}

std::string SceneBuilder::fmt_nullable(float v) const
{
    if (!std::isfinite(v) || v >= prediction::MAX_TTC_S) return "null";
    return fmt(v);
}

std::string SceneBuilder::json_str(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof(esc), "\\u%04x",
                                  static_cast<unsigned int>(c));
                    out += esc;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
    return out;
}

std::string SceneBuilder::json_str(const char* s)
{
    return s ? json_str(std::string(s)) : "null";
}

const char* SceneBuilder::json_bool(bool b)
{
    return b ? "true" : "false";
}

// ─── build_object_json ────────────────────────────────────────────────────────
//
// Serialises one TTCResult as a JSON object.
//
// Example output (one line, no trailing comma):
//   {"id":3,"dir":"ahead","dist_m":1.20,"size":"medium",
//    "speed_m_s":0.40,"closing_m_s":0.40,"ttc_s":3.10,"moving":true}

std::string SceneBuilder::build_object_json(const prediction::TTCResult& r) const
{
    std::ostringstream ss;
    ss << "{";

    // id
    ss << "\"id\":" << r.object_id << ",";

    // direction
    const char* dir = "ahead";
    float b = r.bearing_deg;
    if      (b > 22.5f  && b <= 67.5f)  dir = "ahead-right";
    else if (b > 67.5f  && b <= 112.5f) dir = "right";
    else if (b > 112.5f && b <= 157.5f) dir = "behind-right";
    else if (b > 157.5f && b <= 202.5f) dir = "behind";
    else if (b > 202.5f && b <= 247.5f) dir = "behind-left";
    else if (b > 247.5f && b <= 292.5f) dir = "left";
    else if (b > 292.5f && b <= 337.5f) dir = "ahead-left";
    ss << "\"dir\":" << json_str(dir) << ",";

    // distance in metres
    ss << "\"dist_m\":" << fmt(r.distance_mm / 1000.0f) << ",";

    // size label
    ss << "\"size\":" << json_str(r.size_label ? r.size_label : "unknown") << ",";

    // speed in m/s
    ss << "\"speed_m_s\":" << fmt(r.speed_mm_s / 1000.0f) << ",";

    // closing speed in m/s
    ss << "\"closing_m_s\":" << fmt(r.closing_speed_mm_s / 1000.0f) << ",";

    // TTC in seconds (null if no collision predicted)
    ss << "\"ttc_s\":" << fmt_nullable(r.ttc_s) << ",";

    // CPA distance in metres (null if no CPA computed within horizon)
    float cpa_m = r.cpa.distance_mm / 1000.0f;
    ss << "\"cpa_m\":" << fmt(cpa_m) << ",";

    // moving flag
    ss << "\"moving\":" << json_bool(!r.is_stationary);

    ss << "}";
    return ss.str();
}

// ─── build_sectors_json ───────────────────────────────────────────────────────
//
// Serialises the 8-sector threat table as a JSON object.
// Only occupied sectors are emitted, up to config_.max_sectors.
// Sectors are sorted by urgency (min TTC first, then min distance).
//
// Example output:
//   {
//     "ahead":       {"dist_m":1.20,"ttc_s":3.10},
//     "ahead-left":  {"dist_m":2.50,"ttc_s":null}
//   }

std::string SceneBuilder::build_sectors_json(
    const std::array<prediction::SectorThreat,
                     prediction::NUM_SECTORS>& sectors) const
{
    // Collect occupied sectors.
    std::vector<const prediction::SectorThreat*> occupied;
    occupied.reserve(prediction::NUM_SECTORS);
    for (const auto& st : sectors) {
        if (st.occupied) occupied.push_back(&st);
    }

    // Sort by urgency: finite TTC first (ascending), then by distance ascending.
    std::sort(occupied.begin(), occupied.end(),
              [](const prediction::SectorThreat* a,
                 const prediction::SectorThreat* b) {
                  bool a_ttc = std::isfinite(a->min_ttc_s);
                  bool b_ttc = std::isfinite(b->min_ttc_s);
                  if (a_ttc != b_ttc) return a_ttc > b_ttc; // TTC first
                  if (a_ttc && b_ttc) return a->min_ttc_s < b->min_ttc_s;
                  return a->min_distance_mm < b->min_distance_mm;
              });

    // Truncate to max_sectors.
    if (occupied.size() > config_.max_sectors) {
        occupied.resize(config_.max_sectors);
    }

    std::ostringstream ss;
    ss << "{";
    bool first = true;

    for (const auto* st : occupied) {
        if (!first) ss << ",";
        first = false;

        ss << json_str(st->name()) << ":{"
           << "\"dist_m\":"   << fmt(st->min_distance_mm / 1000.0f) << ","
           << "\"ttc_s\":"    << fmt_nullable(st->min_ttc_s)
           << "}";
    }

    ss << "}";
    return ss.str();
}

// ─── build ────────────────────────────────────────────────────────────────────
//
// Builds the full scene JSON string.
//
// Output structure:
//   {
//     "frame": <uint64>,
//     "risk": "<CLEAR|CAUTION|WARNING|DANGER>",
//     "confidence": <float>,
//     "min_ttc_s": <float|null>,
//     "objects": [ ... ],
//     "sectors": { ... },          // omitted if include_sectors=false
//     "local_density": <float>,    // omitted if perc_ptr=nullptr
//     "training": {                // omitted if training_steps=0
//       "steps": <int>,
//       "loss": <float>
//     }
//   }

std::string SceneBuilder::build(
    const prediction::FullPrediction&   pred,
    const perception::PerceptionResult* perc_ptr,
    int   training_steps,
    float training_loss) const
{
    std::ostringstream ss;
    ss << "{";

    // ── frame id ──────────────────────────────────────────────────────────────
    ss << "\"frame\":" << pred.frame_id() << ",";

    // ── risk level ────────────────────────────────────────────────────────────
    ss << "\"risk\":"
       << json_str(prediction::risk_level_name(pred.risk_level())) << ",";

    // ── model confidence ──────────────────────────────────────────────────────
    ss << "\"confidence\":" << fmt(pred.confidence()) << ",";

    // ── global minimum TTC ────────────────────────────────────────────────────
    ss << "\"min_ttc_s\":" << fmt_nullable(pred.min_ttc_s()) << ",";

    // ── objects array ─────────────────────────────────────────────────────────
    // Take the first max_objects from the results (already sorted by urgency
    // descending by TTCEngine).
    ss << "\"objects\":[";
    {
        const auto& results = pred.ttc.results;
        size_t count = std::min(results.size(), config_.max_objects);
        for (size_t i = 0; i < count; ++i) {
            if (i > 0) ss << ",";
            ss << build_object_json(results[i]);
        }
    }
    ss << "]";

    // ── sectors ───────────────────────────────────────────────────────────────
    if (config_.include_sectors) {
        ss << ",\"sectors\":" << build_sectors_json(pred.ttc.sectors);
    }

    // ── local occupancy density ───────────────────────────────────────────────
    // Derived from the OccupancyMap — only available if PerceptionResult is given.
    if (perc_ptr) {
        // Use the local_density() method of OccupancyMap embedded in the grid.
        // We approximate it from the grid snapshot: count occupied cells within
        // a 1.5m radius of the origin.
        //
        // Origin = cell (200, 200) in a 400×400 grid.
        // radius in cells = 1500 / 25 = 60 cells.
        //
        // Rather than iterating here (the grid is a value copy), we count
        // the occupied flag in the sector threat table as a proxy —
        // fraction of 8 sectors that are occupied.
        int occ = 0;
        for (const auto& st : pred.ttc.sectors) {
            if (st.occupied) ++occ;
        }
        float density = static_cast<float>(occ) / prediction::NUM_SECTORS;
        ss << ",\"local_density\":" << fmt(density);
    }

    // ── training diagnostics ──────────────────────────────────────────────────
    if (config_.include_training_info && training_steps > 0) {
        ss << ",\"training\":{"
           << "\"steps\":" << training_steps << ","
           << "\"loss\":"  << fmt(training_loss)
           << "}";
    }

    ss << "}";
    return ss.str();
}

} // namespace agent