import numpy as np
import cv2

import utilities.load_config as LOAD_CONFIG   

class ROIProcessor:
    """
    A class to handle operations related to Regions of Interest (ROI), such as reading coordinates,
    checking if a point is inside multiple ROIs, and detecting human presence inside ROIs.
    """

    @staticmethod
    def read_coordinates(file_path: str):
        """
        Reads and parses ROI coordinates from a file.

        Args:
            file_path (str): Path to the file containing ROI coordinates.

        Returns:
            list: A list of lists containing tuples of ROI coordinates.
        """
        final_list = []
        try:
            with open(file_path, 'r') as file:
                data = file.read()
                coordinate_lists = data.strip().split('][')

                for sublist in coordinate_lists:
                    sublist = sublist.strip('[]')
                    tuples = sublist.split('), (')
                    tuples_list = [
                        tuple(map(float, tup.replace('(', '').replace(')', '').split(', ')))
                        for tup in tuples
                    ]
                    final_list.append(tuples_list)
            return final_list
        except FileNotFoundError:
            print("Error: coordinates.txt file not found")
            return None

    @staticmethod
    def point_inside_multi_rois(point: tuple, rois: list):
        """
        Checks if a point is inside any of the provided ROIs.

        Args:
            point (tuple): The (x, y) coordinates of the point.
            rois (list): List of ROIs, where each ROI is a list of coordinates.

        Returns:
            list: A list of boolean values indicating whether the point is inside each ROI.
        """
        x, y = point
        inside_rois = []
        for roi in rois:
            inside = False
            n = len(roi)
            p1x, p1y = roi[0]
            for i in range(n + 1):
                p2x, p2y = roi[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            inside_rois.append(inside)
        return inside_rois

    @staticmethod
    def check_intersection_and_area(bbox: tuple, polygon: list):
        """
        Checks for intersection between a bounding box and a polygon (ROI) and calculates the intersection area percentage.

        Args:
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            polygon (list): List of polygon vertices.

        Returns:
            tuple: A boolean indicating if there is an intersection, and the percentage of the bounding box area that intersects the polygon.
        """
        polygon = np.array(polygon, dtype=np.int32)
        polygon = np.array([polygon])

        x1, y1, x2, y2 = bbox
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)

        bbox_area = w * h

        bbox_vertices = np.array([
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ], dtype=np.int32)
        bbox_polygon = np.array([bbox_vertices])

        all_points = np.concatenate((polygon[0], bbox_vertices))
        max_x = np.max(all_points[:, 0])
        max_y = np.max(all_points[:, 1])

        img1 = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
        img2 = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)

        cv2.polylines(img1, polygon, isClosed=True, color=1, thickness=2)
        cv2.polylines(img2, bbox_polygon, isClosed=True, color=1, thickness=2)

        cv2.fillPoly(img1, polygon, 1)
        cv2.fillPoly(img2, bbox_polygon, 1)

        intersection = cv2.bitwise_and(img1, img2)
        intersection_area = np.sum(intersection)

        intersection_area_percentage = (intersection_area / bbox_area) * 100

        return intersection_area > 0, intersection_area_percentage


class HumanDetector:
    """
    A class responsible for detecting humans inside ROIs based on bounding boxes or key points.
    """

    def __init__(self, model):
        """
        Initializes the HumanDetector with a given YOLO model.

        Args:
            model: The YOLO model used for human detection.
        """
        self.model = model

    def detect_humans_inside_regions_by_point(self, frame: np.ndarray, roi_coordinates: list):
        """
        Detects humans inside the provided ROIs by checking the center point of the detected bounding box.

        Args:
            frame (np.ndarray): The video frame to process.
            roi_coordinates (list): List of ROI coordinates.

        Returns:
            bool: True if a human is detected inside any ROI, otherwise False.
        """
        results = self.model(frame, classes=0, verbose=False)[0]

        if results.boxes.xyxy.numel() == 0:
            return False

        for result in list(results.boxes):
            x1, y1, x2, y2 = result.xyxy.tolist()[0]

            center_x = (int(x1) + int(x2)) / 2
            center_y = (int(y1) + int(y2)) / 2

            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 255), -1)

            if ROIProcessor.point_inside_multi_rois((int(center_x), int(center_y)), roi_coordinates):
                return True

        return False

    def detect_humans_inside_regions_by_bbox(self, frame: np.ndarray, roi_coordinates: list, area_percentage: int):
        """
        Detects humans inside the provided ROIs based on bounding box overlap percentage.

        Args:
            frame (np.ndarray): The video frame to process.
            roi_coordinates (list): List of ROI coordinates.
            area_percentage (int): The minimum percentage of bounding box overlap required to consider a detection valid.

        Returns:
            bool: True if a human is detected inside any ROI, otherwise False.
        """
        results = self.model(frame, conf=LOAD_CONFIG.human_det_model_conf, classes = 0, verbose=False)[0]

        if results.boxes.xyxy.numel() == 0:
            return False

        for result in list(results.boxes):
            x1, y1, x2, y2 = result.xyxy.tolist()[0]

            for roi in roi_coordinates:
                intersected_roi, percentage = ROIProcessor.check_intersection_and_area((x1, y1, x2, y2), roi)

                if percentage > area_percentage:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 182, 193), 2)
                    return True

        return False
