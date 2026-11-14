"""Shared constants. The order here must match the CSV column order."""

KEYPOINT_NAMES = ["ofd_1", "ofd_2", "bpd_1", "bpd_2"]
COORD_COLUMNS = [f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y")]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
OFD_IDX = (0, 1)   # indices of the two OFD end-points
BPD_IDX = (2, 3)   # indices of the two BPD end-points
