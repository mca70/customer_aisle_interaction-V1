import cv2
import numpy as np


class HumanPoseEstimation:
    def __init__(self, roi_coordinates):
        self.roi_coordinates = roi_coordinates

    def is_point_in_roi(self, point):
        """
        Check if a point is inside any of the provided regions of interest (ROIs).
        """
        for roi in self.roi_coordinates:
            polygon = np.array(roi, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            if cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0:
                return True
        return False

    def get_human_pose(self, keypoints):
        """
        Check if keypoints of a single human are inside the ROIs.
        """
        # Define the target keypoints indices
        target_points = {
            "head": 0,
            "right_hand": 10,
            "left_hand": 9,
            "right_shoulder": 8,
            "left_shoulder": 7
        }

        for label, idx in target_points.items():
            x, y = keypoints[idx]

            if x == 0 and y == 0:
                continue

            point = (int(x), int(y))
            is_in_region = self.is_point_in_roi(point)

            # Return True if any keypoint is inside the region
            if is_in_region:
                return True

        return False
