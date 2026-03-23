#pragma once

// ─── tracker.h ───────────────────────────────────────────────────────────────
// Multi-object Kalman filter tracker with Hungarian algorithm assignment.
//
// Design
// ──────
// Each detected Cluster from the DBSCAN clusterer is matched to an existing
// track (or spawned as a new one) every frame. The tracker maintains a
// persistent identity for each physical obstacle across frames, and uses a
// Kalman filter per track to:
//   1. Predict where the obstacle will be this frame before measurement arrives
//   2. Correct the prediction with the actual measurement
//   3. Derive a smooth velocity estimate (vx, vy) in mm/s
//
// This velocity is the critical input to the TTC (time-to-collision) engine
// and the risk predictor MLP features. Without it, we can only react to where
// things are. With it, we predict where they will be.
//
// Kalman state vector  x = [ px, py, vx, vy ]^T
// ────────────────────────────────────────────
//   px, py : centroid position in mm (world frame, origin = user)
//   vx, vy : velocity in mm/s
//
// State transition (constant-velocity model):
//   px' = px + vx * dt
//   py' = py + vy * dt
//   vx' = vx
//   vy' = vy
//
//   F = | 1  0  dt  0 |
//       | 0  1   0 dt |
//       | 0  0   1  0 |
//       | 0  0   0  1 |
//
// Measurement model (we observe position only, not velocity):
//   z = [ px, py ]^T
//   H = | 1  0  0  0 |
//       | 0  1  0  0 |
//
// Process noise Q: tuned for typical pedestrian acceleration (~500 mm/s²)
// Measurement noise R: tuned for LD06 centroid accuracy (~40 mm std dev)
//
// Association
// ───────────
// Hungarian algorithm (Munkres, O(N³)) minimises total assignment cost.
// Cost metric: Euclidean distance between predicted track position and
// new cluster centroid.
// Gating threshold: 800 mm — matches beyond this distance are rejected
// (avoids wrongly linking two different objects that happened to swap positions).
//
// Track lifecycle
// ───────────────
//   TENTATIVE : just created, not yet confirmed (age < MIN_HITS)
//   CONFIRMED : seen consistently (age >= MIN_HITS)
//   LOST      : no matching cluster for 1–MAX_LOST_FRAMES frames
//               (track still predicts forward during this period)
//   DEAD      : lost for > MAX_LOST_FRAMES — removed from active set
//
// Parameters
// ──────────
//   MIN_HITS        : frames needed to confirm a new track (default 3)
//   MAX_LOST_FRAMES : frames without a match before deletion (default 5)
//   GATE_MM         : max assignment distance in mm (default 800)
//   DT_S            : nominal timestep in seconds (default 0.1 = 10 Hz)
//
// Thread safety
// ─────────────
//   Not internally locked. The caller (main pipeline thread) is the sole user.

#include "clusterer.h"

#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <string>
#include <chrono>
#include <optional>

namespace perception {

// ─── TrackState ───────────────────────────────────────────────────────────────

enum class TrackState : uint8_t {
    TENTATIVE = 0,   // waiting for confirmation
    CONFIRMED = 1,   // reliably tracked
    LOST      = 2,   // temporarily missing
    DEAD      = 3,   // to be removed
};

inline const char* track_state_name(TrackState s) {
    switch (s) {
        case TrackState::TENTATIVE: return "TENTATIVE";
        case TrackState::CONFIRMED: return "CONFIRMED";
        case TrackState::LOST:      return "LOST";
        case TrackState::DEAD:      return "DEAD";
        default:                    return "UNKNOWN";
    }
}

// ─── Mat4 / Vec4 ──────────────────────────────────────────────────────────────
// Minimal 4×4 matrix and 4-vector for Kalman math.
// Row-major: M(i,j) = data[i*4+j].
// Using raw arrays keeps this header-only and dependency-free.

struct Vec4 {
    std::array<float, 4> v{};

    float& operator[](int i)       { return v[static_cast<size_t>(i)]; }
    float  operator[](int i) const { return v[static_cast<size_t>(i)]; }

    Vec4 operator+(const Vec4& o) const {
        return {{ v[0]+o[0], v[1]+o[1], v[2]+o[2], v[3]+o[3] }};
    }
    Vec4 operator-(const Vec4& o) const {
        return {{ v[0]-o[0], v[1]-o[1], v[2]-o[2], v[3]-o[3] }};
    }
    Vec4 operator*(float s) const {
        return {{ v[0]*s, v[1]*s, v[2]*s, v[3]*s }};
    }
};

struct Vec2 {
    std::array<float, 2> v{};
    float& operator[](int i)       { return v[static_cast<size_t>(i)]; }
    float  operator[](int i) const { return v[static_cast<size_t>(i)]; }
    Vec2 operator-(const Vec2& o) const { return {{ v[0]-o[0], v[1]-o[1] }}; }
};

struct Mat4 {
    std::array<float, 16> m{};

    float& at(int r, int c)       { return m[static_cast<size_t>(r*4+c)]; }
    float  at(int r, int c) const { return m[static_cast<size_t>(r*4+c)]; }

    static Mat4 identity() {
        Mat4 I;
        I.at(0,0) = I.at(1,1) = I.at(2,2) = I.at(3,3) = 1.0f;
        return I;
    }

    Mat4 operator+(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 16; ++i) r.m[i] = m[i] + o.m[i];
        return r;
    }

    Mat4 operator-(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 16; ++i) r.m[i] = m[i] - o.m[i];
        return r;
    }

    // Matrix × matrix
    Mat4 operator*(const Mat4& B) const {
        Mat4 C;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    C.at(i,j) += at(i,k) * B.at(k,j);
        return C;
    }

    // Matrix × vector
    Vec4 operator*(const Vec4& x) const {
        Vec4 y;
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k)
                y[i] += at(i,k) * x[k];
        return y;
    }

    Mat4 transpose() const {
        Mat4 T;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                T.at(j,i) = at(i,j);
        return T;
    }

    Mat4 scaled(float s) const {
        Mat4 r;
        for (int i = 0; i < 16; ++i) r.m[i] = m[i] * s;
        return r;
    }
};

// 2×2 matrix used for the innovation covariance S and its inverse.
struct Mat2 {
    float a=0,b=0,c=0,d=0;   // | a b |
                               // | c d |

    float det() const { return a*d - b*c; }

    // Returns the inverse, or identity if singular.
    Mat2 inv() const {
        float det_val = det();
        if (std::abs(det_val) < 1e-9f) return {1,0,0,1};
        float inv_det = 1.0f / det_val;
        return { d*inv_det, -b*inv_det, -c*inv_det, a*inv_det };
    }
};

// H matrix (2×4): extracts [px, py] from state [px, py, vx, vy].
// H * x_4 → z_2
inline Vec2 H_times(const Vec4& x) {
    return {{ x[0], x[1] }};
}

// Computes (H * P * H^T) + R  where P is 4×4, H is 2×4, R is 2×2 diagonal.
// Result is a 2×2 innovation covariance matrix.
// Exploiting sparsity of H (only first two rows non-zero) for efficiency.
inline Mat2 innovation_covariance(const Mat4& P, float r_pos) {
    // S = H*P*H^T + R
    // H*P extracts first two rows of P.
    // (H*P)*H^T extracts first two columns of (H*P).
    // So S = top-left 2×2 of P + diag(r_pos, r_pos)
    return {
        P.at(0,0) + r_pos,
        P.at(0,1),
        P.at(1,0),
        P.at(1,1) + r_pos
    };
}

// Computes the 4×2 Kalman gain K = P * H^T * S^{-1}.
// K[:,0] = P * H^T[:,0] = P * e0 = first column of P  → scaled by S^{-1} row 0
// K[:,1] = second column of P                          → scaled by S^{-1} row 1
// Returns K as two Vec4 columns: k0, k1.
inline void kalman_gain(const Mat4& P, const Mat2& S_inv,
                        Vec4& k0_out, Vec4& k1_out)
{
    // P * H^T: H^T has non-zero entries only in columns 0,1 of rows 0,1.
    // So (P * H^T)[:, 0] = P[:, 0]  and  (P * H^T)[:, 1] = P[:, 1].
    // K = (P * H^T) * S_inv
    for (int i = 0; i < 4; ++i) {
        float ph0 = P.at(i, 0);   // (P * H^T)[i, 0]
        float ph1 = P.at(i, 1);   // (P * H^T)[i, 1]
        k0_out[i] = ph0 * S_inv.a + ph1 * S_inv.c;
        k1_out[i] = ph0 * S_inv.b + ph1 * S_inv.d;
    }
}

// ─── TrackedObject ────────────────────────────────────────────────────────────
//
// The public output type of the tracker.
// Represents one persistently-tracked obstacle with Kalman-smoothed state.

struct TrackedObject {
    // ── Identity ──────────────────────────────────────────────────────────────

    uint32_t    id          = 0;        // monotonically increasing, never reused
    TrackState  state       = TrackState::TENTATIVE;

    // ── Kalman-filtered state ─────────────────────────────────────────────────

    float   px = 0.0f;   // position X (forward), mm
    float   py = 0.0f;   // position Y (left),    mm
    float   vx = 0.0f;   // velocity X, mm/s  (positive = moving toward user)
    float   vy = 0.0f;   // velocity Y, mm/s

    // ── Derived geometry ──────────────────────────────────────────────────────

    float   distance_mm  = 0.0f;   // Euclidean distance from user, mm
    float   bearing_deg  = 0.0f;   // bearing from user, degrees [0,360)
    float   speed_mm_s   = 0.0f;   // |velocity| in mm/s

    // Closing speed: component of velocity directed toward the user.
    // Positive = approaching, negative = receding.
    float   closing_speed_mm_s = 0.0f;

    // ── Cluster geometry (from last matched cluster) ───────────────────────────

    float   width_mm   = 0.0f;
    float   depth_mm   = 0.0f;
    int     point_count = 0;
    const char* size_label = "unknown";

    // ── Lifecycle counters ────────────────────────────────────────────────────

    int     age          = 0;    // frames since track was created
    int     hits         = 0;    // total frames with a successful match
    int     lost_frames  = 0;    // consecutive frames without a match

    // ── Timestamp ─────────────────────────────────────────────────────────────

    std::chrono::time_point<std::chrono::steady_clock> last_seen;

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Returns true if this track is reliable enough to use for TTC / alerts.
    bool is_confirmed() const { return state == TrackState::CONFIRMED; }

    // Returns true if the object is approaching (positive closing speed).
    bool is_approaching() const { return closing_speed_mm_s > 50.0f; }

    // True if velocity estimate is reliable (enough hits for Kalman to settle).
    bool velocity_reliable() const { return hits >= 4; }

    // Direction label based on bearing.
    const char* direction() const {
        if (bearing_deg <= 22.5f || bearing_deg > 337.5f) return "ahead";
        if (bearing_deg <= 67.5f)  return "ahead-right";
        if (bearing_deg <= 112.5f) return "right";
        if (bearing_deg <= 157.5f) return "behind-right";
        if (bearing_deg <= 202.5f) return "behind";
        if (bearing_deg <= 247.5f) return "behind-left";
        if (bearing_deg <= 292.5f) return "left";
        return "ahead-left";
    }

    std::string str() const;
};

using TrackedObjectList = std::vector<TrackedObject>;

// ─── KalmanTrack ──────────────────────────────────────────────────────────────
//
// Internal track state. Not exposed to consumers — only TrackedObject is public.

struct KalmanTrack {
    uint32_t   id;
    TrackState state     = TrackState::TENTATIVE;
    int        age       = 0;
    int        hits      = 0;
    int        lost      = 0;

    Vec4 x_est;    // state estimate: [px, py, vx, vy]
    Mat4 P;        // estimate covariance

    // Cluster geometry from last match (carried through for TrackedObject output)
    float   width_mm    = 0.0f;
    float   depth_mm    = 0.0f;
    int     point_count = 0;
    const char* size_label = "unknown";

    std::chrono::time_point<std::chrono::steady_clock> last_seen;

    // Convenience: predicted position (px, py) as Vec2.
    Vec2 predicted_pos() const { return {{ x_est[0], x_est[1] }}; }

    // Build a TrackedObject from this internal track.
    TrackedObject to_tracked_object() const;
};

// ─── Tracker ──────────────────────────────────────────────────────────────────

class Tracker {
public:
    // ── Parameters ────────────────────────────────────────────────────────────

    static constexpr int   MIN_HITS         = 3;      // frames to confirm
    static constexpr int   MAX_LOST_FRAMES  = 5;      // frames before deletion
    static constexpr float GATE_MM          = 1200.0f; // max assignment distance
    static constexpr float DT_S             = 0.10f;  // nominal timestep (10 Hz)

    // Process noise: uncertainty added per step (tuned for pedestrian motion).
    // Higher = trust measurements more, lower = trust model more.
    static constexpr float Q_POS  = 100.0f;   // position process noise (mm²)
    static constexpr float Q_VEL  = 2500.0f;  // velocity process noise (mm²/s²)

    // Measurement noise: LD06 centroid position std dev ≈ 40 mm → R = 40² = 1600
    static constexpr float R_POS  = 1600.0f;  // measurement noise (mm²)

    // Initial covariance for new tracks.
    static constexpr float P0_POS = 10000.0f; // large initial position uncertainty
    static constexpr float P0_VEL = 40000.0f; // very large initial velocity uncertainty

    // ── Construction ──────────────────────────────────────────────────────────

    Tracker();

    // Reset: clears all tracks. Call when restarting the pipeline.
    void reset();

    // ── Main entry point ──────────────────────────────────────────────────────

    // Update tracker with new clusters from one scan frame.
    //
    // dt_s: actual elapsed time since last call (seconds).
    //       Pass 0 to use the default DT_S constant.
    //       The pipeline measures wall-clock time between ScanFrames and
    //       passes it here for accurate velocity integration.
    //
    // Returns the list of currently active (non-dead) tracks as TrackedObjects,
    // sorted by distance to user ascending.
    TrackedObjectList update(const ClusterList& clusters, float dt_s = 0.0f);

    // ── Accessors ─────────────────────────────────────────────────────────────

    int  track_count()     const { return static_cast<int>(tracks_.size()); }
    int  confirmed_count() const;
    uint32_t next_id()     const { return next_id_; }

private:
    // ── Track store ───────────────────────────────────────────────────────────

    std::vector<KalmanTrack> tracks_;
    uint32_t                 next_id_ = 1;

    // ── Kalman operations ─────────────────────────────────────────────────────

    // Predict all tracks forward by dt_s seconds using constant-velocity model.
    void predict(float dt_s);

    // Build the state transition matrix F for timestep dt_s.
    static Mat4 make_F(float dt_s);

    // Build the process noise matrix Q for timestep dt_s.
    static Mat4 make_Q(float dt_s);

    // Apply a measurement update (centroid_x, centroid_y) to track `t`.
    static void update_track(KalmanTrack& t, float meas_x, float meas_y);

    // Initialise a new KalmanTrack from a Cluster.
    KalmanTrack init_track(const Cluster& c);

    // ── Hungarian assignment ──────────────────────────────────────────────────

    // Solves the assignment problem:
    //   rows = existing tracks, cols = new clusters
    //   cost[i][j] = Euclidean distance from track[i] predicted pos to cluster[j] centroid
    //   gating: cost > GATE_MM → treat as infinity (no match allowed)
    //
    // Returns assignment vector: assign[i] = j means track[i] matched cluster[j].
    //                            assign[i] = -1 means track[i] had no match.
    // unmatched_clusters is filled with cluster indices that no track claimed.
    std::vector<int> hungarian_assign(
        const std::vector<KalmanTrack>& tracks,
        const ClusterList& clusters,
        std::vector<int>& unmatched_clusters) const;

    // ── Post-update helpers ────────────────────────────────────────────────────

    // Derives distance, bearing, speed, closing_speed from Kalman state.
    static void fill_derived(TrackedObject& obj);

    // Remove all DEAD tracks from tracks_.
    void prune_dead();
};

} // namespace perception
