import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from multiprocessing.pool import ThreadPool
import argparse

from video_operations import VideoOperations
from utilities import load_config as LOAD_CONFIG
from utilities.region_operations import ROIProcessor, HumanDetector


class CustomerInteractionMonitor:
    def __init__(self, rtsp_url: str, cam_no: int, area_percentage: int, stop_time_str: str):
        self.rtsp_url = rtsp_url
        self.cam_no = cam_no
        self.area_percentage = area_percentage
        stop_time = datetime.strptime(stop_time_str, '%I:%M %p')
        self.stop_time = datetime.combine(datetime.now(), stop_time.time())  
        self.model = self.load_yolo_model(LOAD_CONFIG.human_det_model)
        self.video_operations = VideoOperations(cam_no)
        self.thread_pool = ThreadPool(4)
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open RTSP stream at {rtsp_url}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output_file_path = self.prepare_output_directory(LOAD_CONFIG.video_base_path)
        self.all_rois = self.initialize_rois()
        self.human_detector = HumanDetector(self.model)


    def load_yolo_model(self, model_path: str, device: str = 'cuda') -> YOLO:
        """
        Loads the YOLOv8 model and transfers it to the specified device.
        """
        model = YOLO(model_path)
        model.to(device)
        return model

    def prepare_output_directory(self, base_path: str) -> str:
        """
        Prepares the output directory for storing video files.
        """
        output_directory = os.path.join(base_path, datetime.now().strftime("%Y-%m-%d"), "videos", str(self.cam_no))
        os.makedirs(output_directory, exist_ok=True)
        return output_directory

    def initialize_rois(self) -> list:
        """
        Initializes regions of interest (ROIs) based on the camera's configuration.
        """
        roi_coordinates = ROIProcessor.read_coordinates(f"../regions/coordinates_{self.cam_no}.txt")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return [[(int(x * width), int(y * height)) for x, y in roi] for roi in roi_coordinates]

    def draw_roi_on_frame(self, frame: np.ndarray) -> None:
        """
        Draws regions of interest (ROIs) on the given frame.
        """
        for roi in self.all_rois:
            cv2.polylines(frame, [np.array(roi)], isClosed=True, color=(0, 255, 0), thickness=2)

    def human_detection_and_video_creation(self) -> None:
        """
        Main function for human detection and video creation based on the detection.
        """
        frame_count = 0
        frames_to_skip = int(self.fps // 5)
        interaction_count = 1
        human_check = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from RTSP stream")
                break

            frame_count += 1
            if frame_count % frames_to_skip != 0:
                continue  # Skip frames

            self.draw_roi_on_frame(frame)

            if self.human_detector.detect_humans_inside_regions_by_bbox(frame, self.all_rois, self.area_percentage):
                human_check.append(True)
                if len(human_check) >= 10 and sum(human_check) >= 3 and not self.video_operations.video_check:
                    print("Human detected! Initiating video cropping process...")

                    start_time = '00:00:00.0'
                    end_time = f'00:00:{LOAD_CONFIG.video_length}.0'

                    self.thread_pool.apply_async(self.video_operations.create_cropped_video,
                                                 args=(interaction_count, self.rtsp_url,
                                                       os.path.join(self.output_file_path, f'{interaction_count}.mp4'),
                                                       self.area_percentage, start_time, end_time))
                    interaction_count += 1
                    human_check.clear()
                elif len(human_check) >= 10 and sum(human_check) < 3:
                    human_check.clear()
            else:
                human_check.append(False)

            resized_frame = cv2.resize(frame, (480, 320))
            cv2.imshow(f"Cam : {self.cam_no}", resized_frame)

            remaining_time = self.stop_time - datetime.now()
            if cv2.waitKey(1) & 0xFF == ord('q') or remaining_time.total_seconds() <= 0:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        """
        Starts the monitoring process for customer interaction on RTSP stream.
        """
        self.human_detection_and_video_creation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor customer interaction on RTSP stream.')
    parser.add_argument('--rtsp', type=str, required=True, help='RTSP URL')
    parser.add_argument('--cam_no', type=int, required=True, help='Camera number')
    parser.add_argument('--area_percentage', type=int, required=True, help='Percentage of area interaction required to trigger video creation')

    args = parser.parse_args()

    stop_time = LOAD_CONFIG.process_runtime

    monitor = CustomerInteractionMonitor(rtsp_url=args.rtsp, cam_no=args.cam_no, area_percentage=args.area_percentage, stop_time_str=stop_time)
    monitor.run()


# NOTE-> this script name will be renamed to 'customer_interaction_monitoring'