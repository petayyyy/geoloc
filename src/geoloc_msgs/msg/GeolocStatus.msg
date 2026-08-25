# System diagnostics, published at 1 Hz by geoloc_fusion.

std_msgs/Header header

# --- Mode (see plan/02-architecture.md section 4) ---
uint8 MODE_INIT=0               # waiting for odometry and first fix; window = +-R0
uint8 MODE_CONVERGING=1         # 1-4 fixes accepted
uint8 MODE_NOMINAL=2            # >=5 fixes, sigma_yaw < 1 deg
uint8 MODE_COASTING=3           # no valid fixes > 20 s; odometry only
uint8 MODE_LOST=4               # no fixes > 180 s or covariance over threshold
uint8 MODE_DEGRADED=5           # primary channel failed, fallback active
uint8 mode

# --- Degradation level for the operator (see T27) ---
uint8 LEVEL_NORMAL=0
uint8 LEVEL_ATTENTION=1
uint8 LEVEL_WARNING=2
uint8 LEVEL_FAILURE=3
uint8 LEVEL_FAULT=4
uint8 degradation_level

# --- Fix statistics ---
uint32  fixes_accepted_total
uint32  fixes_rejected_total
float32 fixes_per_km
float32 time_since_last_fix_s
float32 distance_since_last_fix_m

# --- Estimate state ---
float64 prior_window_radius_m   # current R fed back to the matcher
float64 sigma_east_m
float64 sigma_north_m
float64 sigma_yaw_deg

# --- Diagnostics ---
string  last_rejection_reason   # which gate fired; empty if none
uint32  active_channel          # see SE2Fix CHANNEL_*

# --- Runtime (see T33: managed degradation beats uncontrolled throttling) ---
float32 cpu_load_percent
float32 soc_temperature_c
float32 fix_rate_hz             # actual, may be reduced by thermal management
