import subprocess
import time
import os
import cv2
import numpy as np
import json
import base64
import time
from datetime import datetime
from ultralytics import YOLO

from lib.s3_bucket import s3_upload_file
from customer_activity_filters_testing import HumanTracker
from utilities.store_related import detect_staff_and_trolley
from utilities import load_config as LOAD_CONFIG

class VideoOperations:
    def __init__(self, aisle_id: int) -> None:
        self.aisle_id = aisle_id
        self.store_id = LOAD_CONFIG.store_id
        self.target_base_path = f"WIWO/{datetime.now().date().strftime('%Y-%m-%d')}/{self.store_id}/{self.aisle_id}/"
        self.video_check = False
        # Load YOLOv8 model
        # self.human_model = YOLO("yolov8n.pt")
        # self.human_model.to('cuda')
        # self.hands_model = YOLO("hands.pt")
        # self.hands_model.to('cuda')
        # self.trolley_model = YOLO('trolley_iceland.pt')
        # self.trolley_model.to('cuda')
        self.activity_filter = HumanTracker(aisle_id = self.aisle_id)
        
    def upload_video(self, video_file_path: str, filename):
        try:
            print("Uploading video to bucket...")
         
            target_file_path = self.target_base_path + filename + '.mp4'
            print(video_file_path)
            video_uri = s3_upload_file(video_file_path, target_file_path)
            print("Video uploaded successfully on bucket!!")
            print(video_uri)
        except Exception as e:
            print("Error in uploading video on bucket")
            print(str(e))

    def upload_image(self, image_file_path: str, count: int, filename):
        try:
            print("Uploading image to bucket...")
            print(image_file_path)
            target_file_path = self.target_base_path + filename + f'_{count + 1}.jpeg'
            image_uri = s3_upload_file(image_file_path, target_file_path)
            print("Image uploaded successfully on bucket!!")
            print(image_uri)
        except Exception as e:
            print("Error in uploading image on bucket")
            import traceback
            traceback.print_tb(e.__traceback__)
            print(e)

    def create_cropped_video(self, interaction_count, input_file: str, output_file: str, area_percentage, start_time, end_time):
        # Construct the ffmpeg command
        print("trimming video : ", output_file)
        
        ffmpeg_command = [
            "ffmpeg",
            "-loglevel", "fatal",
            "-i", input_file,
            "-ss", start_time,
            "-t", end_time,
            "-vf", "scale=480:320",     # Resize video frames to the desired width (w) and height (h)
            "-c:v", "libx264",      # Use H.264 video codec for compression
            #"-crf", "15",           # Constant Rate Factor (quality), lower value means better quality
            #"-preset", "medium",  # Preset for higher compression at the expense of encoding time
            "-c:a", "aac",          # Use AAC audio codec for compression
            #"-b:a", "96k",          # Audio bitrate (lower value for more compression)
            output_file
        ]
        
        #print(ffmpeg_command)
        self.video_check = True
        # Execute the ffmpeg command
        start_time = time.time()
        subprocess.run(ffmpeg_command)  
        print("Video created!!")
        
        try:
            # human_filter = self.check_human_in_video(output_file, area_percentage)
            start = time.time()
            print("activity start for ", self.aisle_id)
            human_filter = self.activity_filter.run(output_file)
            print("human filter : ", human_filter)
            print("execution time ; ", time.time() - start)
        except Exception as e:
            print('*'*50)
            import traceback as tb
            tb.print_tb(e.__traceback__)
        
        if human_filter:
            end_time = time.time()
            video_creation_time = end_time - start_time
            data = {"store_id": self.store_id, 
                "aisle_id": self.aisle_id, 
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "description": 'theft',
                "video_creation_time_time" : int(video_creation_time),
                "section": LOAD_CONFIG.map_with_counter[str(self.aisle_id)]
                }

            json_str = json.dumps(data)
            base64_encoded_filename = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
            frames = self.get_frames(output_file, interaction_count)
            print(data)
            self.upload_video(output_file, base64_encoded_filename)
            print("uploading images...")
            for idx, image_path in enumerate(frames):
                #self.upload_image(image_path, idx, base64_encoded_filename, date)
                self.upload_image(image_path, idx, base64_encoded_filename)
            print("images uploaded successfully!!")
        self.video_check = False         

    def check_human_in_video_old(self, video_path: str, area_percentage: int):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames_to_skip = fps // 5
        human_total_appearance_in_seconds = 3
        frame_count = 0
        consecutive_human_appearance_count = 0
        required_consecutive_frames = human_total_appearance_in_seconds * (fps // frames_to_skip)
        max_consecutive_missed_frames = 3

        # Read coordinates from coordinates.txt
        roi_coordinates = read_coordinates(f"../regions/coordinates_{self.aisle_id}.txt")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Specify the region of interest (ROI) coordinates
        all_rois = []
        for roi_coordinate in roi_coordinates:
            roi_normalized = [(int(x * width), int(y * height)) for x, y in roi_coordinate]
            all_rois.append(roi_normalized)

        if not cap.isOpened():
            print("Error: Could not open RTSP stream")
            exit()
                
        consecutive_missed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video")
                break
                    
            frame_count += 1
            if frame_count % frames_to_skip != 0:
                continue  # Skip frames   

            if detect_humans_inside_regions_by_bbox(frame, self.human_model, all_rois, -1, area_percentage):
                consecutive_human_appearance_count += 1
                consecutive_missed_frames = 0
                if consecutive_human_appearance_count >= required_consecutive_frames:
                    print("\033[92m" + "*" * 50 + "\033[0m")
                    print("\033[92m" + "HUMAN FOUND FOR STRAIGHT 3 SECONDS" + "\033[0m")
                    print("\033[92m" + "*" * 50 + "\033[0m")
                    return True
            else:
                consecutive_missed_frames += 1
                if consecutive_missed_frames > max_consecutive_missed_frames:
                    consecutive_human_appearance_count = 0
                    consecutive_missed_frames = 0

        print("\033[91m" + "*" * 50 + "\033[0m")
        print("\033[91m" + "HUMAN NOT FOUND FOR STRAIGHT 3 SECONDS" + "\033[0m")
        print("\033[91m" + "*" * 50 + "\033[0m")
        return False

    
    def check_human_in_video(self, video_path: str, area_percentage: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open RTSP stream")
            exit()
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames_to_skip = 5
        frames_in_1_second = fps // frames_to_skip

        human_total_appearance_in_seconds = LOAD_CONFIG.customer_aisle_duration
        frame_count = 0
        consecutive_human_appearance_count = 0
        required_consecutive_frames = human_total_appearance_in_seconds * frames_in_1_second
        max_consecutive_missed_frames = frames_in_1_second - int(0.3 * frames_in_1_second)
        
        total_trolley_frames = 0
        required_trolley_frames = frames_in_1_second

        # Read coordinates from coordinates.txt
        roi_coordinates = read_coordinates(f"regions/coordinates_{self.aisle_id}.txt")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Specify the region of interest (ROI) coordinates
        all_rois = []
        for roi_coordinate in roi_coordinates:
            roi_normalized = [(int(x * width), int(y * height)) for x, y in roi_coordinate]
            all_rois.append(roi_normalized)
                
        consecutive_missed_frames = 0
        timer_start = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video")
                break
            
            frame_count += 1
            if frame_count % frames_to_skip != 0:
                continue  # Skip frames   
                
            if total_trolley_frames > required_trolley_frames:
                print("\033[91m" + "*" * 50 + "\033[0m")
                print("\033[91m" + "TROLLEY DETECTEDD" + "\033[0m")
                print("\033[91m" + "*" * 50 + "\033[0m")
                return False

            if detect_staff_and_trolley(self.trolley_model, frame):
                total_trolley_frames += 1

            if detect_humans_inside_regions_by_bbox(frame, self.human_model, all_rois, -1, area_percentage):
                consecutive_human_appearance_count += 1
                consecutive_missed_frames = 0
                if timer_start is None:
                    timer_start = frame_count / frames_in_1_second  # Start the timer
                    
                time_inside_region = consecutive_human_appearance_count // frames_in_1_second
                if consecutive_human_appearance_count >= required_consecutive_frames:
                    print("\033[92m" + "*" * 50 + "\033[0m")
                    print("\033[92m" + "HUMAN FOUND FOR STRAIGHT 3 SECONDS" + "\033[0m")
                    print("\033[92m" + "*" * 50 + "\033[0m")
                    return True

            else:
                consecutive_missed_frames += 1
                consecutive_human_appearance_count += 1
                if consecutive_missed_frames > max_consecutive_missed_frames:
                    consecutive_human_appearance_count = 0
                    consecutive_missed_frames = 0
                    timer_start = None  # Reset the timer

        print("\033[91m" + "*" * 50 + "\033[0m")
        print("\033[91m" + "HUMAN NOT FOUND FOR STRAIGHT 3 SECONDS" + "\033[0m")
        print("\033[91m" + "*" * 50 + "\033[0m")
        return False
    
    def get_frames(self, video_path, interaction_count):
        try:
            current_date = datetime.now().date().strftime("%Y-%m-%d")
            target_path = f'D:\\Sai_Group\\customer_aisle_interaction\\data\\{current_date}\\images\\{self.aisle_id}\\'
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            
            # start_times = ["00:00:04", "00:00:05", "00:00:06", "00:00:07"]
            start_times = ["00:00:04"]
            image_files = [f"{target_path}\\{interaction_count}_{index}.jpg" for index in range(len(start_times))]
            
            for start_time, image_file in zip(start_times, image_files):
                # Construct the FFmpeg command
                command = [
                    "ffmpeg",
                    "-loglevel", "fatal",
                    "-ss", start_time,
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "10",
                    image_file
                ]

                
                subprocess.run(command)
            return image_files
        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
            print(str(e))
    
