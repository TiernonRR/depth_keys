"""
config.py
Configuration constants for keypoints and skeleton definitions.
"""

# Biological definitions
INCL_KPOINTS_FIT_TRANSFORM = [
    "back_bottom", "back_middle_lower", "back_middle_upper", "back_top",
    "left_hip", "right_hip", "left_shoulder", "right_shoulder",
    "left_ear", "right_ear"
]

INCL_KPOINTS_POST_PROCESSING = [
    "tail_tip", "tail_middle", "tail_base",
    "back_bottom", "back_middle_lower", "back_middle_upper", "back_top",
    "left_hip", "right_hip", "left_shoulder", "right_shoulder",
    "left_ear", "right_ear", "snout"
]

NOISY_KEYPOINTS = ["tail_tip", "tail_middle", "tail_base", "snout"]

SKELETON_DEFINITIONS = [
    ("tail_tip", "tail_middle"),
    ("tail_middle", "tail_base"),
    ("back_bottom", "back_middle_lower"),
    ("back_middle_lower", "back_middle_upper"),
    ("back_middle_upper", "back_top"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("back_top", "left_shoulder"),
    ("back_top", "right_shoulder"),
    ("back_bottom", "left_hip"),
    ("back_bottom", "right_hip"),
    ("snout", "left_ear"),
    ("snout", "right_ear")
]

# Analysis parameters
SMOOTHING_PARAMS = {
    "not_noisy": {"window_length": int(5), "poly_order": int(2)},
    "noisy": {"window_length": int(25), "poly_order": int(2)},
}

FPS = 100