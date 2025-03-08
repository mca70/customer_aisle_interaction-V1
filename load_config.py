import json

# Load JSON data from file
with open('D:\Sai_Group\customer_aisle_interaction\config.json', 'r') as file:
    config = json.load(file)

# Accessing values from the JSON data
bucket_creds = config['bucket_creds']
video_paths = config['video_paths']
store_id = config['store_id']
map_with_counter = config['map_with_counter']
video_length = config['video_length']
customer_aisle_duration = config['customer_aisle_interaction_min_duration_in_seconds']
video_base_path = config['data_base_path']
process_runtime = config['process_runtime']


human_det_model = config['human_detection_model']
human_det_model_conf = config['human_detection_model_conf']
human_pose_model = config['human_pose_model']
human_pose_model_conf = config['human_pose_model_conf']