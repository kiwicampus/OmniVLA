"""Robot heading from the GPS track, in the compass convention the goal pose uses."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import utm


def course_heading_deg(
    past_lonlat: Sequence[float], now_lonlat: Sequence[float], min_move: float = 0.5
) -> Optional[float]:
    """Direction of travel in degrees clockwise from UTM north (0 = north, 90 = east).

    Returns None when a fix is missing or the robot moved less than `min_move` metres.
    """
    past, now = np.asarray(past_lonlat, float), np.asarray(now_lonlat, float)
    if past.shape != (2,) or now.shape != (2,) or not (np.isfinite(past).all() and np.isfinite(now).all()):
        return None
    # ponytail: no forced UTM zone, a track straddling a zone edge gives a wrong course
    east_a, north_a = utm.from_latlon(past[1], past[0])[:2]
    east_b, north_b = utm.from_latlon(now[1], now[0])[:2]
    move_e, move_n = east_b - east_a, north_b - north_a
    if math.hypot(move_e, move_n) < min_move:
        return None
    return math.degrees(math.atan2(move_e, move_n)) % 360.0
