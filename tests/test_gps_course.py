import unittest

from omnivla_training.gps_course import course_heading_deg

LAT, LON = 4.6, -74.07
STEP = 0.0001  # about 11 m


def _gap(a, b):
    return abs((a - b + 180) % 360 - 180)


class GpsCourseTests(unittest.TestCase):
    def test_compass_directions(self):
        self.assertLess(_gap(course_heading_deg([LON, LAT], [LON, LAT + STEP]), 0.0), 2)
        self.assertLess(_gap(course_heading_deg([LON, LAT], [LON + STEP, LAT]), 90.0), 2)
        self.assertLess(_gap(course_heading_deg([LON, LAT], [LON, LAT - STEP]), 180.0), 2)
        self.assertLess(_gap(course_heading_deg([LON, LAT], [LON - STEP, LAT]), 270.0), 2)

    def test_none_when_barely_moved_or_fix_missing(self):
        self.assertIsNone(course_heading_deg([LON, LAT], [LON, LAT + 1e-6]))
        self.assertIsNone(course_heading_deg([float("nan"), LAT], [LON, LAT + STEP]))
        self.assertIsNone(course_heading_deg([], [LON, LAT]))


if __name__ == "__main__":
    unittest.main()
