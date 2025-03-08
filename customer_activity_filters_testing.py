from collections import defaultdict, deque
import cv2
import numpy as np
from glob import glob
import random
from ultralytics import YOLO
import traceback

from utilities import load_config as LOAD_CONFIG
from utilities.region_operations import ROIProcessor
from utilities.human_pose import HumanPoseEstimation
from utilities.store_related import detect_staff_and_trolley_batch

import time


class HumanTracker:
    def __init__(self, model_path = LOAD_CONFIG.human_pose_model, aisle_id=None, frame_skip=5, movement_threshold=8, 
                        walking_buffer_threshold=2, inside_buffer_threshold=3):
        # print(model_path)
        self.model = YOLO(model_path)
        self.model.to('cuda')
        self.staff_model = YOLO('../weights/staff_iceland.pt')
        self.staff_model.to('cuda')
        self.trolley_model = YOLO('../weights/trolley_iceland.pt')
        # print('tracking model loaded for ', aisle_id)
        self.aisle_id = aisle_id
        self.frame_skip = frame_skip
        self.movement_threshold = movement_threshold
        self.stationary_threshold = int((LOAD_CONFIG.customer_aisle_duration * 25) / frame_skip)  # Assume 30 FPS as default
        self.walking_buffer_threshold = walking_buffer_threshold
        self.inside_buffer_threshold = inside_buffer_threshold
        self.stationary_detected = False  # To track if stationary detection is found

    def _initialize_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.track_history = defaultdict(lambda: [])
        self.stationary_counter = defaultdict(lambda: 0)
        self.walking_buffer_counter = defaultdict(lambda: 0)
        self.inside_buffer_counter = defaultdict(lambda: 0)

        self.frame_count = 0

        # Extract aisle ID and load ROI coordinates
        self.roi_coordinates = ROIProcessor.read_coordinates(f"../regions/coordinates_{self.aisle_id}.txt")
        self.all_rois = self._normalize_rois()
        self.pose_estimator = HumanPoseEstimation(self.all_rois)

    def _normalize_rois(self):
        all_rois = []
        for roi_coordinate in self.roi_coordinates:
            roi_normalized = [(int(x * self.width), int(y * self.height)) for x, y in roi_coordinate]
            all_rois.append(roi_normalized)
        return all_rois

    def process_frame(self, frame):
        # self.frame_count += 1

        # Skip frames
        # if self.frame_count % self.frame_skip != 0:
            # return None

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.model.track(frame, persist=True, classes=[0], tracker = 'bytetrack.yaml', verbose=False)

        if results[0].boxes.id is None:
            return frame

        # Get the boxes, track IDs, and keypoints
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xy.tolist()

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        # annotated_frame = frame

        # Draw the ROI coordinates on the frame
        # for roi in self.all_rois:
            # polygon = np.array(roi, np.int32)
            # polygon = polygon.reshape((-1, 1, 2))
            # cv2.polylines(annotated_frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

        # Process each track
        for box, track_id, keypoint in zip(boxes, track_ids, keypoints):
            stationary_detected = self._process_track(box, track_id, keypoint)
            if stationary_detected:
                self.stationary_detected = True  # Set the flag to True
                return   # Early exit once a stationary customer is detected

        return 

    def _process_track(self, box, track_id, keypoint):
        x, y, w, h = box
        track = self.track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point

        if len(track) > 30:  # retain 30 tracks for 30 frames
            track.pop(0)

        is_inside = self.pose_estimator.get_human_pose(keypoint)
        if not is_inside:
            self.inside_buffer_counter[track_id] += 1
            if self.inside_buffer_counter[track_id] > self.inside_buffer_threshold:
                self.stationary_counter[track_id] = 0
                self.walking_buffer_counter[track_id] = 0
        else:
            self.inside_buffer_counter[track_id] = 0

        if len(self.track_history[track_id]) >= 2:
            prev_center = self.track_history[track_id][-2]
            curr_center = self.track_history[track_id][-1]
            displacement = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)

            if displacement > self.movement_threshold:
                self.walking_buffer_counter[track_id] += 1
                if self.walking_buffer_counter[track_id] > self.walking_buffer_threshold:
                    # print(f"human id {track_id} walking...")
                    # points = np.array(self.track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    self.stationary_counter[track_id] = 0
            else:
                self.walking_buffer_counter[track_id] = 0
                self.stationary_counter[track_id] += 1
                if self.stationary_counter[track_id] > self.stationary_threshold:
                    # print(f"human id {track_id} has been stationary for more than 3 seconds")
                    return True

        return False

    def run(self, video_path):
        self.stationary_detected = False
        self._initialize_video(video_path)

        staff_bbox_count = 0
        trolley_bbox_count = 0
        
        frames_batch = []
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success:
                break

            self.frame_count += 1

            if self.frame_count % self.frame_skip != 0:
                continue  # Skip this frame and go to the next

            # Validate the frame before adding it to the batch
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                print(f"Skipping invalid frame at frame_count {self.frame_count}.")
                continue
                
            frames_batch.append(frame)
            
            if len(frames_batch) >= 5:
                count = detect_staff_and_trolley_batch(self.staff_model, frames_batch)
                staff_bbox_count += count
                if staff_bbox_count > 5:
                    print("\033[93m" + f"Staff detected in {video_path}." + "\033[0m")
                    return False
                    
                count = detect_staff_and_trolley_batch(self.trolley_model, frames_batch)
                trolley_bbox_count += count
                if trolley_bbox_count > 5:
                    print("\033[94m" + f"Trolley detected in {video_path}." + "\033[0m")
                    return False

                for f in frames_batch:
                    _ = self.process_frame(f)
                    if self.stationary_detected:  # If stationary detection is found
                        print("\033[92m" + f"Stationary human detected in {video_path}." + "\033[0m")
                        return True
                    
                if self.frame_count > self.fps * 16:
                    return False
                
                frames_batch.clear()
            else:
                continue

        self.cap.release()
        print("\033[91m" + f"No stationary human detected in {video_path}." + "\033[0m")
        return False


# try:
    # track = HumanTracker(aisle_id = '118', frame_skip = 6)
    # start = time.time()
    # out = track.run('../data/2024-09-09_ietr/videos/118/5.mp4')
    # print("Execution Time : ", time.time() - start)
# except Exception as e:
    # traceback.print_tb(e.__traceback__)

# print(out)

# if __name__ == "__main__":

    # from glob import glob
    # from time import time
    # videos = glob('data/2024-09-05/videos/65/*')
    
    # execution_times = []
    
    # track = HumanTracker(aisle_id = '65')
    
    # for video in videos:
        # aisle_id = video.split('\\')[-1].split('-')[0]
        # start = time()
        # out = track.run(video)
        # end = time()
        # execution_times.append(end - start)
        # print(out)
        # if out:
            # print(video)
    
    # print(execution_times)
    # print("max time : ", max(execution_times))
    # print("min time : ", min(execution_times))
    