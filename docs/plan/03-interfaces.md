# 03 — Интерфейсы и контракты данных

Этот документ — источник истины для всех агентов-исполнителей. Любое изменение здесь требует обновления зависимых задач.

---

## 1. Пакет сообщений `geoloc_msgs`

### 1.1 `OrthoPatch.msg`

True-ortho патч с геопривязкой.

```
std_msgs/Header header          # stamp = время центрального кадра, frame_id = "map_enu"

sensor_msgs/Image image         # mono8 или bgr8, уже в GSD карты
sensor_msgs/Image confidence    # mono8: 255 = лидарное покрытие, 0 = нет данных,
                                #        промежуточное = Copernicus DEM

float64 origin_east             # ENU-координата левого верхнего угла, м
float64 origin_north
float64 gsd                     # м/пиксель
float64 yaw_map                 # поворот патча относительно оси North, рад (обычно 0)

float64 agl                     # высота над подстилающей, м
float64[9] pose_covariance      # ковариация позы камеры на момент кадра (x,y,yaw)
float32 lidar_coverage_ratio    # доля пикселей с лидарным покрытием, 0..1
```

### 1.2 `SE2Fix.msg`

Результат матчинга. Публикуется **всегда**, даже если качество низкое — решение принимает `geoloc_integrity`.

```
std_msgs/Header header

# Коррекция map_enu -> odom, ВЫРАЖЕННАЯ КАК ПОПРАВКА
float64 delta_east              # м
float64 delta_north             # м
float64 delta_yaw               # рад, нормализован в [-pi, pi]
float64[9] covariance           # 3x3 row-major по (east, north, yaw)

# Метрики качества (вход для integrity)
uint8  channel                  # 0=XFEAT, 1=PHASE_CORR, 2=SEMANTIC
uint32 n_correspondences        # всего соответствий до RANSAC
uint32 n_inliers                # после MAGSAC
float32 inlier_ratio
float32 covisibility            # доля площади патча, покрытая инлаерами, 0..1
float32 peak_ratio              # острота: best / second_best. >1.3 хорошо
float32 residual_rms_px         # RMS остатка инлаеров, пиксели
float32 spatial_spread          # мера равномерности распределения инлаеров, 0..1
float32 mean_confidence         # средняя уверенность DSM по инлаерам, 0..1

float32 processing_time_ms
```

### 1.3 `GeolocStatus.msg`

Диагностика, публикуется 1 Гц.

```
std_msgs/Header header

uint8 MODE_INIT=0
uint8 MODE_CONVERGING=1
uint8 MODE_NOMINAL=2
uint8 MODE_COASTING=3
uint8 MODE_LOST=4
uint8 MODE_DEGRADED=5
uint8 mode

uint32  fixes_accepted_total
uint32  fixes_rejected_total
float32 fixes_per_km
float32 time_since_last_fix_s
float32 distance_since_last_fix_m

float64 prior_window_radius_m   # текущий R
float64 sigma_east_m
float64 sigma_north_m
float64 sigma_yaw_deg

string  last_rejection_reason
float32 cpu_load_percent
float32 soc_temperature_c
```

### 1.4 `MapWindow.srv`

```
# запрос
float64 center_east
float64 center_north
float64 radius_m
float64 gsd
bool    with_descriptors
---
# ответ
sensor_msgs/Image image
float64 origin_east
float64 origin_north
float64 gsd
bool    success
string  message
# дескрипторы (если запрошены)
uint32   n_keypoints
float32[] keypoints_xy          # 2*N
int8[]    descriptors           # N*64, int8
```

---

## 2. Топики

| Топик | Тип | Такт | Издатель |
|---|---|---|---|
| `/livox/lidar` | `livox_ros_driver2/CustomMsg` | 10 Гц | драйвер |
| `/livox/imu` | `sensor_msgs/Imu` | 200 Гц | драйвер |
| `/camera/image_raw` | `sensor_msgs/Image` | 10 Гц | драйвер |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | 10 Гц | драйвер |
| `/fast_livo2/odometry` | `nav_msgs/Odometry` | ≥10 Гц | fast_livo2 |
| `/fast_livo2/cloud_registered` | `sensor_msgs/PointCloud2` | 10 Гц | fast_livo2 |
| `/geoloc/ortho_patch` | `geoloc_msgs/OrthoPatch` | 1–2 Гц | geoloc_ortho |
| `/geoloc/fix_raw` | `geoloc_msgs/SE2Fix` | 1–2 Гц | geoloc_matcher |
| `/geoloc/fix` | `geoloc_msgs/SE2Fix` | ≤2 Гц | geoloc_integrity |
| `/geoloc/fix_rejected` | `geoloc_msgs/SE2Fix` | — | geoloc_integrity |
| `/geoloc/global_pose` | `geometry_msgs/PoseWithCovarianceStamped` | 10 Гц | geoloc_fusion |
| `/geoloc/status` | `geoloc_msgs/GeolocStatus` | 1 Гц | geoloc_fusion |
| `/mavros/gps_input/gps_input` | `mavros_msgs/GPSINPUT` | 5 Гц | geoloc_mavlink |

**QoS:** сенсорные топики — `SensorDataQoS` (best effort, depth 5). Фиксы и статус — `reliable`, depth 10, `transient_local` для статуса.

---

## 3. Контракт с PX4 (`GPS_INPUT`)

| Поле | Заполнение |
|---|---|
| `time_usec` | Время **кадра**, не публикации. Задержка компенсируется здесь |
| `gps_id` | 0 |
| `ignore_flags` | Игнорируем `vel_horiz`, `vel_vert`, `speed_accuracy` — скорость даёт FAST-LIVO2 через свой канал |
| `lat`, `lon` | Из `map_enu` → WGS84 через опорную точку, 1e-7 град |
| `alt` | Из фьюжна; источник высоты согласован с `EKF2_HGT_MODE` |
| `hdop` | `clamp(sqrt(σ_e² + σ_n²) / 5.0, 0.5, 99.0)` — эмпирическая шкала, калибруется в SITL |
| `vdop` | Аналогично по σ_up |
| `eph` | `sqrt(σ_e² + σ_n²)`, м — **напрямую из ковариации, без сглаживания** |
| `epv` | `σ_up`, м |
| `fix_type` | `NOMINAL/DEGRADED` → 3; `CONVERGING` → 2; `COASTING` → 2, затем 1 по росту σ; `LOST/INIT` → 0 |
| `satellites_visible` | `clamp(6 + n_recent_fixes, 6, 18)` — монотонно по качеству; EKF2 использует как грубый признак |
| `yaw` | Курс из фьюжна, cdeg. Заполняется только при σ_ψ < 2° |

> **Критично.** `eph` — это то, чему EKF2 верит. Занижение приводит к тому, что автопилот доверяет плохому фиксу; завышение — к тому, что игнорирует хороший. Шкала калибруется в SITL (T27) через проверку состоятельности: доля времени, когда истинная ошибка лежит внутри 1σ, должна быть ~68%.

### Параметры PX4 (стартовая точка, уточняется в T27)

| Параметр | Значение | Комментарий |
|---|---|---|
| `EKF2_AID_MASK` | GPS вкл., optical flow выкл., vision выкл. | Мы маскируемся под GNSS |
| `EKF2_HGT_MODE` | Range finder (лидар даёт z) | Согласовать с `EKF2_RNG_*` |
| `EKF2_GPS_CHECK` | Ослабить проверки на число спутников и дрейф | Иначе EKF2 отвергнет синтетический GNSS |
| `EKF2_REQ_EPH` | 25 м | Наш p95 — 30 м, порог должен быть выше типичного |
| `EKF2_REQ_SACC` | Ослабить | Точность скорости мы не заявляем |
| `EKF2_GPS_DELAY_MS` | Измеренная задержка контура | Не оставлять по умолчанию |

---

## 4. Формат пакета карты `.geopack`

Директория (не архив — чтобы читать mmap'ом):

```
<mission>.geopack/
  manifest.yaml          # версия, CRS, границы, провайдеры, даты съёмки, оценка bias
  ortho_a.tif            # COG, основная подложка, EPSG:4326 или UTM
  ortho_b.tif            # COG, второй провайдер (для OrthoSim и кросс-проверки)
  dem.tif                # Copernicus GLO-30, ресемплированный
  semantic.tif           # растр OSM-классов, uint8, 1 м/px
  descriptors/           # опционально: предвычисленные дескрипторы по сетке
  gcp.csv                # опционально: контрольные точки для оценки bias
```

`manifest.yaml`:

```yaml
version: 1
mission_id: "corridor-2026-08"
crs: "EPSG:32639"
bounds: {east_min: ..., east_max: ..., north_min: ..., north_max: ...}
origin: {lat: ..., lon: ..., alt: ...}       # опорная точка map_enu
layers:
  ortho_a: {provider: "esri_world_imagery", capture_date: "2024-06", gsd: 0.5,
            georef_bias: {east: 2.1, north: -1.4, sigma: 3.0}}
  ortho_b: {provider: "bing_aerial", capture_date: "2023-09", gsd: 0.5}
  dem:     {source: "copernicus_glo30", gsd: 30.0, vertical_datum: "EGM2008"}
  semantic:{source: "osm", extract_date: "2026-08-01", gsd: 1.0,
            classes: [background, road, building, water, farmland, forest]}
```

---

## 5. Соглашения

| Что | Соглашение |
|---|---|
| Углы | Радианы внутри, градусы только в диагностике и логах |
| Курс | ENU, против часовой от оси East. Нормализация в `[-π, π]` **всегда** |
| Время | ROS Time, источник — время сенсора, не время приёма |
| Ковариации | row-major, порядок `(east, north, yaw)` |
| Единицы | СИ везде |
| Именование | `snake_case` для топиков и параметров, `PascalCase` для типов сообщений |
| Ошибки | Узел никогда не падает молча: любая аномалия → `/geoloc/status.last_rejection_reason` + лог уровня WARN |

---

**Дальше:** `04-roadmap.md`
