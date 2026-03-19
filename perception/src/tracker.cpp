// ─── tracker.cpp ─────────────────────────────────────────────────────────────
// Kalman filter multi-object tracker with Hungarian assignment.
// See include/perception/tracker.h for the full design notes.

#include "perception/tracker.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace perception {

// ─── TrackedObject::str ───────────────────────────────────────────────────────

std::string TrackedObject::str() const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0);
    ss << "Track{id=" << id
       << " state=" << track_state_name(state)
       << " pos=(" << px << ", " << py << ") mm"
       << " vel=(" << vx << ", " << vy << ") mm/s"
       << " dist=" << distance_mm << " mm"
       << " bearing=" << std::setprecision(1) << bearing_deg << "°"
       << std::setprecision(0)
       << " closing=" << closing_speed_mm_s << " mm/s"
       << " age=" << age << " hits=" << hits
       << "}";
    return ss.str();
}

// ─── KalmanTrack::to_tracked_object ──────────────────────────────────────────

TrackedObject KalmanTrack::to_tracked_object() const
{
    TrackedObject obj;
    obj.id          = id;
    obj.state       = state;
    obj.px          = x_est[0];
    obj.py          = x_est[1];
    obj.vx          = x_est[2];
    obj.vy          = x_est[3];
    obj.age         = age;
    obj.hits        = hits;
    obj.lost_frames = lost;
    obj.last_seen   = last_seen;
    obj.width_mm    = width_mm;
    obj.depth_mm    = depth_mm;
    obj.point_count = point_count;
    obj.size_label  = size_label;
    return obj;
}

// ─── Tracker construction ─────────────────────────────────────────────────────

Tracker::Tracker()
{
    tracks_.reserve(32);
}

void Tracker::reset()
{
    tracks_.clear();
    next_id_ = 1;
}

// ─── make_F ───────────────────────────────────────────────────────────────────
//
// State transition matrix for constant-velocity model:
//
//   px' = px + vx*dt
//   py' = py + vy*dt
//   vx' = vx
//   vy' = vy
//
//   F = | 1  0  dt  0 |
//       | 0  1   0 dt |
//       | 0  0   1  0 |
//       | 0  0   0  1 |

Mat4 Tracker::make_F(float dt_s)
{
    Mat4 F = Mat4::identity();
    F.at(0, 2) = dt_s;
    F.at(1, 3) = dt_s;
    return F;
}

// ─── make_Q ───────────────────────────────────────────────────────────────────
//
// Discrete process noise matrix using the continuous white-noise acceleration
// model. For a constant-velocity model with process noise on acceleration:
//
//   Q = sigma_a² * G * G^T
//
// where G = [dt²/2, dt²/2, dt, dt]^T (influence of acceleration on state).
//
// We use a simplified diagonal approximation for efficiency:
//
//   Q = diag(Q_POS * dt², Q_POS * dt², Q_VEL * dt², Q_VEL * dt²)
//
// This correctly scales uncertainty with timestep — a longer gap between
// frames means more uncertainty accumulated in both position and velocity.

Mat4 Tracker::make_Q(float dt_s)
{
    Mat4 Q;
    float dt2 = dt_s * dt_s;
    Q.at(0, 0) = Q_POS * dt2;
    Q.at(1, 1) = Q_POS * dt2;
    Q.at(2, 2) = Q_VEL * dt2;
    Q.at(3, 3) = Q_VEL * dt2;
    return Q;
}

// ─── init_track ───────────────────────────────────────────────────────────────
//
// Spawns a new KalmanTrack from a Cluster.
// Initial state: position = centroid, velocity = (0, 0).
// Initial covariance: large diagonal (high uncertainty about velocity).

KalmanTrack Tracker::init_track(const Cluster& c)
{
    KalmanTrack t;
    t.id    = next_id_++;
    t.state = TrackState::TENTATIVE;
    t.age   = 1;
    t.hits  = 1;
    t.lost  = 0;

    // State: [px, py, vx, vy]
    t.x_est[0] = c.centroid_x;
    t.x_est[1] = c.centroid_y;
    t.x_est[2] = 0.0f;
    t.x_est[3] = 0.0f;

    // Initial covariance: small position uncertainty, large velocity uncertainty.
    t.P = Mat4();
    t.P.at(0, 0) = P0_POS;
    t.P.at(1, 1) = P0_POS;
    t.P.at(2, 2) = P0_VEL;
    t.P.at(3, 3) = P0_VEL;

    // Carry cluster geometry.
    t.width_mm    = c.width_mm;
    t.depth_mm    = c.depth_mm;
    t.point_count = c.point_count();
    t.size_label  = c.size_label();

    t.last_seen = std::chrono::steady_clock::now();

    return t;
}

// ─── predict ─────────────────────────────────────────────────────────────────
//
// Kalman predict step for all active tracks:
//
//   x_pred = F * x_est
//   P_pred = F * P * F^T + Q
//
// This runs before we have the new measurements and moves each track's
// estimate forward in time using the constant-velocity motion model.

void Tracker::predict(float dt_s)
{
    if (dt_s <= 0.0f) dt_s = DT_S;

    const Mat4 F  = make_F(dt_s);
    const Mat4 FT = F.transpose();
    const Mat4 Q  = make_Q(dt_s);

    for (auto& t : tracks_) {
        if (t.state == TrackState::DEAD) continue;

        // x_pred = F * x_est
        t.x_est = F * t.x_est;

        // P_pred = F * P * F^T + Q
        t.P = (F * t.P) * FT + Q;
    }
}

// ─── update_track ────────────────────────────────────────────────────────────
//
// Kalman update step for a single track given a new position measurement.
//
//   Innovation:       y  = z - H * x_pred          (z = [meas_x, meas_y])
//   Innovation cov:   S  = H * P * H^T + R
//   Kalman gain:      K  = P * H^T * S^{-1}
//   Updated state:    x  = x_pred + K * y
//   Updated cov:      P  = (I - K * H) * P_pred     (Joseph form for stability)

void Tracker::update_track(KalmanTrack& t, float meas_x, float meas_y)
{
    // Innovation: y = z - H*x  (H extracts px, py from [px,py,vx,vy])
    float y0 = meas_x - t.x_est[0];
    float y1 = meas_y - t.x_est[1];

    // Innovation covariance S = H*P*H^T + R (2×2, exploiting H sparsity)
    Mat2 S = innovation_covariance(t.P, R_POS);
    Mat2 S_inv = S.inv();

    // Kalman gain K (4×2): K = P * H^T * S^{-1}
    Vec4 k0, k1;
    kalman_gain(t.P, S_inv, k0, k1);

    // State update: x = x + K * y
    for (int i = 0; i < 4; ++i) {
        t.x_est[i] += k0[i] * y0 + k1[i] * y1;
    }

    // Covariance update: P = (I - K*H) * P
    // K*H is 4×4: (K[:,0] * H[0,:]) + (K[:,1] * H[1,:])
    // H[0,:] = [1,0,0,0],  H[1,:] = [0,1,0,0]
    // So K*H = outer(k0, e0) + outer(k1, e1)
    //        where e0=[1,0,0,0], e1=[0,1,0,0]
    // (I - K*H)[i,j] = delta(i,j) - k0[i]*delta(j,0) - k1[i]*delta(j,1)
    //
    // Apply: P_new[i,j] = sum_k (I-KH)[i,k] * P[k,j]
    // = P[i,j] - k0[i]*P[0,j] - k1[i]*P[1,j]
    Mat4 P_new;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            P_new.at(i, j) = t.P.at(i, j)
                           - k0[i] * t.P.at(0, j)
                           - k1[i] * t.P.at(1, j);
        }
    }

    // Joseph form stabilisation: P = 0.5*(P + P^T) to enforce symmetry
    // and prevent numerical drift from compounding floating-point errors.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            t.P.at(i, j) = 0.5f * (P_new.at(i, j) + P_new.at(j, i));
        }
    }
}

// ─── hungarian_assign ────────────────────────────────────────────────────────
//
// Solves the linear assignment problem using the Hungarian algorithm (Munkres).
//
// Cost matrix C[i][j] = Euclidean distance from track[i] predicted position
// to cluster[j] centroid.  Entries beyond GATE_MM are set to a large value
// (infinity sentinel) to prevent cross-gating assignments.
//
// Algorithm overview (O(N³)):
//   1. Subtract row minima (row reduction).
//   2. Subtract column minima (column reduction).
//   3. Find a maximum matching using zero-cost entries.
//   4. If matching is not perfect, augment with the minimum uncovered value
//      and repeat from step 3.
//
// This is a clean, self-contained implementation — no external library needed.

std::vector<int> Tracker::hungarian_assign(
    const std::vector<KalmanTrack>& tracks,
    const ClusterList& clusters,
    std::vector<int>& unmatched_clusters) const
{
    const int T = static_cast<int>(tracks.size());
    const int C = static_cast<int>(clusters.size());

    // Result vectors
    std::vector<int> assign(static_cast<size_t>(T), -1);
    unmatched_clusters.clear();

    if (T == 0) {
        for (int j = 0; j < C; ++j) unmatched_clusters.push_back(j);
        return assign;
    }

    if (C == 0) {
        return assign;
    }

    // Build square cost matrix (pad to max(T,C) × max(T,C) with large values).
    const int N = std::max(T, C);
    constexpr float INF = 1e9f;
    const float GATE = GATE_MM;

    // cost[i][j] stored row-major, size N×N.
    std::vector<float> cost(static_cast<size_t>(N * N), INF);

    for (int i = 0; i < T; ++i) {
        const auto& t = tracks[static_cast<size_t>(i)];
        if (t.state == TrackState::DEAD) continue;

        for (int j = 0; j < C; ++j) {
            float dx = t.x_est[0] - clusters[static_cast<size_t>(j)].centroid_x;
            float dy = t.x_est[1] - clusters[static_cast<size_t>(j)].centroid_y;
            float dist = std::sqrt(dx*dx + dy*dy);
            cost[static_cast<size_t>(i * N + j)] = (dist <= GATE) ? dist : INF;
        }
    }

    // ── Hungarian algorithm ───────────────────────────────────────────────────

    std::vector<float> u(static_cast<size_t>(N), 0.0f);  // row potentials
    std::vector<float> v(static_cast<size_t>(N), 0.0f);  // col potentials
    std::vector<int>   p(static_cast<size_t>(N + 1), 0); // col → row assignment (1-indexed rows)
    std::vector<int>   way(static_cast<size_t>(N + 1), 0);

    // Jonker-Volgenant style Hungarian (numerically robust, clean O(N³)).
    for (int i = 1; i <= N; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(static_cast<size_t>(N + 1), INF);
        std::vector<bool>  used(static_cast<size_t>(N + 1), false);

        do {
            used[static_cast<size_t>(j0)] = true;
            int i0 = p[static_cast<size_t>(j0)];
            float delta = INF;
            int   j1 = -1;

            for (int j = 1; j <= N; ++j) {
                if (used[static_cast<size_t>(j)]) continue;

                // Cost with potential subtracted (reduced cost).
                float cur = cost[static_cast<size_t>((i0 - 1) * N + (j - 1))]
                          - u[static_cast<size_t>(i0 - 1)]
                          - v[static_cast<size_t>(j - 1)];

                if (cur < minv[static_cast<size_t>(j)]) {
                    minv[static_cast<size_t>(j)] = cur;
                    way[static_cast<size_t>(j)]  = j0;
                }
                if (minv[static_cast<size_t>(j)] < delta) {
                    delta = minv[static_cast<size_t>(j)];
                    j1 = j;
                }
            }

            // Update potentials
            for (int j = 0; j <= N; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(j)] - 1)] += delta;
                    v[static_cast<size_t>(j)] -= delta;
                } else {
                    minv[static_cast<size_t>(j)] -= delta;
                }
            }

            j0 = j1;
        } while (p[static_cast<size_t>(j0)] != 0);

        // Augment path
        do {
            int j1 = way[static_cast<size_t>(j0)];
            p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    // ── Extract assignments ───────────────────────────────────────────────────
    // p[j] = i means column j (cluster j-1) is assigned to row i (track i-1).
    // We only accept assignments where the original cost was < INF (within gate).

    std::vector<bool> cluster_matched(static_cast<size_t>(C), false);

    for (int j = 1; j <= N; ++j) {
        int i = p[static_cast<size_t>(j)];
        if (i == 0) continue;                  // padding row — skip
        if (i  > T) continue;                  // padding track
        if (j  > C) continue;                  // padding cluster

        float c = cost[static_cast<size_t>((i - 1) * N + (j - 1))];
        if (c >= INF) continue;                // gated out

        assign[static_cast<size_t>(i - 1)]    = j - 1;
        cluster_matched[static_cast<size_t>(j - 1)] = true;
    }

    // Collect unmatched clusters.
    for (int j = 0; j < C; ++j) {
        if (!cluster_matched[static_cast<size_t>(j)]) {
            unmatched_clusters.push_back(j);
        }
    }

    return assign;
}

// ─── fill_derived ─────────────────────────────────────────────────────────────
//
// Computes distance, bearing, speed, and closing_speed from Kalman state.
// Called after every update so consumers always see fresh derived fields.

void Tracker::fill_derived(TrackedObject& obj)
{
    // Distance from user (origin).
    obj.distance_mm = std::sqrt(obj.px * obj.px + obj.py * obj.py);

    // Bearing: angle from +X (forward), clockwise positive.
    float bearing_rad = std::atan2(-obj.py, obj.px);
    float bearing_deg = bearing_rad * (180.0f / static_cast<float>(M_PI));
    if (bearing_deg < 0.0f) bearing_deg += 360.0f;
    obj.bearing_deg = bearing_deg;

    // Speed magnitude.
    obj.speed_mm_s = std::sqrt(obj.vx * obj.vx + obj.vy * obj.vy);

    // Closing speed: negative dot product of velocity with the unit vector
    // pointing from the object toward the user.
    // Unit vector from object to user = (-px, -py) / distance
    // closing_speed = dot(vel, unit_toward_user)
    //               = dot((vx,vy), (-px,-py)/dist)
    //               = -(vx*px + vy*py) / dist
    if (obj.distance_mm > 1.0f) {
        obj.closing_speed_mm_s =
            -(obj.vx * obj.px + obj.vy * obj.py) / obj.distance_mm;
    } else {
        obj.closing_speed_mm_s = 0.0f;
    }
}

// ─── confirmed_count ─────────────────────────────────────────────────────────

int Tracker::confirmed_count() const
{
    int n = 0;
    for (const auto& t : tracks_) {
        if (t.state == TrackState::CONFIRMED) ++n;
    }
    return n;
}

// ─── prune_dead ───────────────────────────────────────────────────────────────

void Tracker::prune_dead()
{
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [](const KalmanTrack& t) { return t.state == TrackState::DEAD; }),
        tracks_.end());
}

// ─── update ───────────────────────────────────────────────────────────────────
//
// Main entry point. Called once per ScanFrame with the new ClusterList.
//
// Steps:
//   1. Predict all active tracks forward by dt_s.
//   2. Hungarian assignment: match tracks to clusters.
//   3. Update matched tracks with Kalman measurement update.
//   4. Mark unmatched tracks as LOST (increment lost_frames).
//      Kill tracks that have been lost too long.
//   5. Spawn new tracks from unmatched clusters.
//   6. Promote TENTATIVE tracks that have enough hits.
//   7. Prune DEAD tracks.
//   8. Build and return TrackedObjectList sorted by distance.

TrackedObjectList Tracker::update(const ClusterList& clusters, float dt_s)
{
    if (dt_s <= 0.0f) dt_s = DT_S;

    // ── 1. Predict ────────────────────────────────────────────────────────────
    predict(dt_s);

    // ── 2. Assign clusters to tracks ─────────────────────────────────────────
    std::vector<int> unmatched_clusters;
    std::vector<int> assignment = hungarian_assign(tracks_, clusters,
                                                   unmatched_clusters);

    // ── 3. Update matched tracks ──────────────────────────────────────────────
    for (size_t i = 0; i < tracks_.size(); ++i) {
        int j = assignment[i];
        auto& t = tracks_[i];
        if (t.state == TrackState::DEAD) continue;

        if (j >= 0) {
            // Matched: Kalman update with cluster centroid.
            const auto& c = clusters[static_cast<size_t>(j)];
            update_track(t, c.centroid_x, c.centroid_y);

            // Update cluster geometry fields.
            t.width_mm    = c.width_mm;
            t.depth_mm    = c.depth_mm;
            t.point_count = c.point_count();
            t.size_label  = c.size_label();

            ++t.hits;
            ++t.age;
            t.lost    = 0;
            t.last_seen = std::chrono::steady_clock::now();
        }
    }

    // ── 4. Handle unmatched tracks ────────────────────────────────────────────
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (assignment[i] >= 0) continue;
        auto& t = tracks_[i];
        if (t.state == TrackState::DEAD) continue;

        ++t.lost;
        ++t.age;

        if (t.lost > MAX_LOST_FRAMES) {
            t.state = TrackState::DEAD;
        } else {
            t.state = TrackState::LOST;
        }
    }

    // ── 5. Spawn new tracks from unmatched clusters ───────────────────────────
    for (int j : unmatched_clusters) {
        tracks_.push_back(init_track(clusters[static_cast<size_t>(j)]));
    }

    // ── 6. Promote TENTATIVE tracks ───────────────────────────────────────────
    for (auto& t : tracks_) {
        if (t.state == TrackState::TENTATIVE && t.hits >= MIN_HITS) {
            t.state = TrackState::CONFIRMED;
        }
    }

    // ── 7. Prune dead tracks ──────────────────────────────────────────────────
    prune_dead();

    // ── 8. Build output ───────────────────────────────────────────────────────
    TrackedObjectList result;
    result.reserve(tracks_.size());

    for (const auto& t : tracks_) {
        if (t.state == TrackState::DEAD) continue;

        TrackedObject obj = t.to_tracked_object();
        fill_derived(obj);
        result.push_back(std::move(obj));
    }

    // Sort by distance to user, closest first.
    std::sort(result.begin(), result.end(),
              [](const TrackedObject& a, const TrackedObject& b) {
                  return a.distance_mm < b.distance_mm;
              });

    return result;
}

} // namespace perception