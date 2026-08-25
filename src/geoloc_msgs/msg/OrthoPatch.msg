# True-ortho patch with geo-referencing, produced by geoloc_ortho.
#
# The patch is already resampled to the basemap GSD and lives in the map_enu
# frame. It carries a per-pixel confidence map because the Livox Avia FOV
# (70.4 x 77.2 deg) is NARROWER than the camera frame: at 100 m AGL the cloud
# covers ~141 x 155 m while the frame covers 200 x 150 m. Roughly 30% of the
# patch width has no lidar coverage and is filled from the global DEM.
# Everything downstream MUST weight by confidence.

std_msgs/Header header          # stamp = time of the central frame; frame_id = "map_enu"

sensor_msgs/Image image         # mono8 or bgr8, already at basemap GSD
sensor_msgs/Image confidence    # mono8: 255 = dense lidar coverage
                                #          0 = no data (outside frame, water, occlusion)
                                #    between = sparse lidar / transition / DEM-only

float64 origin_east             # ENU coordinate of the top-left corner, metres
float64 origin_north
float64 gsd                     # metres per pixel
float64 yaw_map                 # patch rotation relative to North, rad (normally 0)

float64 agl                     # height above the underlying surface, metres
float64[9] pose_covariance      # camera pose covariance at frame time, row-major (x, y, yaw)
float32 lidar_coverage_ratio    # fraction of pixels with lidar coverage, 0..1
