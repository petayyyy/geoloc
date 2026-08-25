# Result of map matching, produced by geoloc_matcher.
#
# ARCHITECTURAL RULE: this message is published ALWAYS, even when quality is
# poor. geoloc_matcher never decides whether a fix is valid -- that decision
# lives only in geoloc_integrity. Keeping the decision isolated is what makes
# the integrity logic independently testable, and it is the system's main
# safety interlock.

std_msgs/Header header

# Correction of map_enu -> odom, expressed as a DELTA
float64 delta_east              # metres
float64 delta_north             # metres
float64 delta_yaw               # rad, normalised to [-pi, pi]
float64[9] covariance           # 3x3 row-major over (east, north, yaw)

# --- Channel ---
uint8 CHANNEL_XFEAT=0           # primary: XFeat* + MAGSAC
uint8 CHANNEL_PHASE_CORR=1      # fallback 1: log-polar phase correlation on gradients
uint8 CHANNEL_SEMANTIC=2        # fallback 2: segmentation vs OSM. NEVER yields fix_type=3
uint8 channel

# --- Quality metrics: the input geoloc_integrity gates on ---
uint32 n_correspondences        # total correspondences before RANSAC
uint32 n_inliers                # after MAGSAC
float32 inlier_ratio

float32 covisibility            # fraction of patch area covered by inliers, 0..1
                                # COMPUTED OVER THE HIGH-CONFIDENCE REGION ONLY.
                                # Counting the DEM-only region inflates it and
                                # defeats the >=0.20 gate.

float32 peak_ratio              # best / second-best SPATIALLY SEPARATED hypothesis.
                                # NOT the ratio of the top two descriptor distances.
                                # On periodic structures (crop grids, warehouse rows)
                                # this is the ONLY signal distinguishing a correct
                                # match from one shifted by exactly one period.

float32 residual_rms_px         # RMS residual of inliers, pixels
float32 spatial_spread          # inlier distribution uniformity, 0..1 (0 = one corner)
float32 mean_confidence         # mean DSM confidence over inliers, 0..1

float32 scale_check             # estimated / known scale. Diagnostic only -- scale is
                                # NOT estimated (see ADR-002). Large disagreement is a
                                # useful bad-match signal.

float32 processing_time_ms
