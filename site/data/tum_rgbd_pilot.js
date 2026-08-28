window.CONPATH_REAL_PILOT = {
  "schema_version": 1,
  "project": "ConPath",
  "dataset": {
    "name": "TUM RGB-D Freiburg1/desk",
    "sequence": "rgbd_dataset_freiburg1_desk",
    "source_url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
    "download_url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "license": "CC BY 4.0 (TUM RGB-D dataset terms)",
    "rgb_frames_available": 613,
    "depth_frames_available": 595,
    "groundtruth_poses_available": 2335,
    "sampled_frames": 48,
    "synchronised_frames": 48
  },
  "protocol": {
    "type": "real-data reference-map pilot",
    "sampled_rgb_frames": 48,
    "temporal_split": {
      "observed_prefix_frames": 36,
      "future_query_frames": 12
    },
    "depth_stride_pixels": 8,
    "camera_intrinsics": {
      "fx": 517.3,
      "fy": 516.5,
      "cx": 318.6,
      "cy": 255.3,
      "depth_scale": 5000.0
    },
    "raster": {
      "resolution_m": 0.04,
      "support_height_m": 0.765,
      "support_tolerance_m": 0.09,
      "obstacle_height_m": 0.1,
      "support_repeat_threshold": 2,
      "obstacle_repeat_threshold": 2,
      "morphology": "3x3 close support / 3x3 obstacle dilation"
    },
    "queries": 18,
    "radii_cells": [
      0,
      1,
      2
    ],
    "monte_carlo_samples": 48,
    "seed": 6000
  },
  "map": {
    "height_cells": 79,
    "width_cells": 105,
    "bounds_m": [
      -2.364222012887548,
      -0.5453,
      1.8251,
      2.6065910347961565
    ],
    "support_cells": 3167,
    "obstacle_cells": 2918,
    "free_cells": 1409
  },
  "metrics": [
    {
      "id": "observed_prefix",
      "name": "Observed prefix (deterministic)",
      "brier": 0.24074074074074073,
      "nll": 2.7716377971605097,
      "ece": 0.24074074074074073,
      "false_safe_rate@0.8": 0.018518518518518517,
      "count": 648
    },
    {
      "id": "independent_cell",
      "name": "Independent-cell completion",
      "brier": 0.19075028247817302,
      "nll": 1.8515848903967045,
      "ece": 0.18999605610422046,
      "false_safe_rate@0.8": 0.018518518518518517,
      "count": 648
    },
    {
      "id": "correlated_temporal",
      "name": "Correlated temporal completion",
      "brier": 0.18306260186487705,
      "nll": 1.9717756261542718,
      "ece": 0.19235468102780023,
      "false_safe_rate@0.8": 0.018518518518518517,
      "count": 648
    }
  ],
  "claim_boundary": "This is a real RGB-D geometric reference-map pilot. TUM provides RGB/depth and camera motion, not traversability or collision labels; metrics must not be presented as a public navigation benchmark or as a validated neural P0 result.",
  "generated_at_utc": "2026-08-28T09:46:18.065263+00:00",
  "assets": {
    "teaser": "assets/tum_freiburg1_desk_teaser.jpg",
    "rgb": "assets/tum_freiburg1_desk_rgb.jpg",
    "depth": "assets/tum_freiburg1_desk_depth.jpg",
    "video": "assets/tum_freiburg1_desk_demo.mp4",
    "poster": "assets/tum_freiburg1_desk_demo_poster.jpg",
    "comparison": "assets/tum_freiburg1_desk_comparison.svg",
    "reliability": "assets/tum_freiburg1_desk_reliability.svg"
  }
};
