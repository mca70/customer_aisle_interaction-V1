import cv2
import json
import numpy as np

from utilities import load_config

# Function to handle mouse events
def select_region(event, x, y, flags, param):
    global points, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cropping = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        cropping = False

    if cropping:
        for i in range(len(points) - 1):
            cv2.line(frame_copy, points[i], points[i + 1], (0, 255, 0), 2)

video_paths = load_config.video_paths

# List to store ROIs for each stream
rois = []

# Iterate over each RTSP stream URL
for idx, video_path in video_paths.items():
    cap = cv2.VideoCapture(video_path[0])

    # Read the first frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read the video frame for RTSP stream {idx}")
        continue

    frame_copy = frame.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_region)

    regions = []
    points = []
    cropping = False

    while True:
        cv2.imshow("image", frame_copy)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset the region selection
        if key == ord("r"):
            frame_copy = frame.copy()
            points = []

        # Press 's' to save the coordinates and image and move to the next stream
        elif key == ord("s"):
            if len(points) >= 3:
                print(f"Polygon Vertices for ROI {len(rois) + 1} in RTSP stream {idx}:", points)
                
                # Normalize the coordinates
                height, width, _ = frame.shape
                normalized_points = [(x / width, y / height) for x, y in points]
                
                # Draw the polygon on the original frame and save it
                pts = np.array(points, np.int32)
                cv2.polylines(frame, [pts], isClosed = True, color = (0, 255, 0), thickness = 2)
                cv2.imwrite(f"plots/{idx}_roi{len(rois) + 1}.jpg", frame)
                
                # Save the coordinates to a list of ROIs
                rois.append(normalized_points)
                regions.append(normalized_points)
            else:
                print(f"Please select at least three points to define a polygon for ROI {len(rois) + 1} in RTSP stream {idx}.")
        
        # Press 'c' to move to the next ROI without saving
        elif key == ord("c"):
            frame_copy = frame.copy()
            points = []

        # Press 'q' to move to the next RTSP stream
        elif key == ord("q"):
            # Save the coordinates to a file
            with open(f"../regions/coordinates_{idx}.txt", "w") as file:
                for r in regions:
                    file.write(str(r))
            break

    # Release the video capture object and close the window for the current stream
    cap.release()
    cv2.destroyAllWindows()
