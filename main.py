import os
import json
import random
import string
from datetime import datetime
from subprocess import Popen, PIPE

from utilities import load_config as LOAD_CONFIG    

# def check_and_rename_folder(base_path: str) -> str:
    # """
    # Checks if a folder with the current date exists. If it does, renames it with a random suffix.
    
    # Args:
        # base_path (str): Base path where the folder is checked.
    
    # Returns:
        # str: The path to the folder that will be used.
    # """
    # current_date = datetime.now().strftime('%Y-%m-%d')
    # target_path = os.path.join(base_path, current_date)
    
    # if os.path.exists(target_path):
        # random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        # new_folder_name = f"{target_path}_{random_suffix}"
        # os.rename(target_path, new_folder_name)
        # print(f"Folder '{target_path}' exists. Renamed to '{new_folder_name}'.")
        # return new_folder_name
    # else:
        # print(f"Folder '{target_path}' does not exist. Creating folder '{target_path}'.")
        # os.makedirs(target_path)
        # return target_path

# def run_monitoring_scripts() -> None:
    # """
    # Loads the RTSP URLs from a configuration file and starts monitoring processes for each camera.
    
    # Args:
        # config_path (str): Path to the configuration file.
    # """
    
    # rtsp_urls = LOAD_CONFIG.video_paths
    
    # commands = [
        # f'python monitor_aisle.py --rtsp {url[0]} --cam_no {cam} --area_percentage {url[1]}'
        # for cam, url in rtsp_urls.items()
    # ]

    # processes = [Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE) for cmd in commands]

    # for p in processes:
        # stdout, stderr = p.communicate()
        # if p.returncode == 0:
            # print(f"Process completed successfully: {stdout.decode()}")
        # else:
            # print(f"Process failed: {stderr.decode()}")

# if __name__ == "__main__":
    # base_path = LOAD_CONFIG.video_base_path
    
    # data_path = check_and_rename_folder(base_path)
    # run_monitoring_scripts()


def check_and_rename_folder():
    base_path = "D:\\Sai_Group\\customer_aisle_interaction\\data\\"
    current_date = datetime.now().strftime('%Y-%m-%d')
    target_path = os.path.join(base_path, current_date)
    
    if os.path.exists(target_path):
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k = 4))
        
        new_folder_name = f"{target_path}_{random_suffix}"
        os.rename(target_path, new_folder_name)
        
        print(f"Folder '{target_path}' exists. Renamed to '{new_folder_name}'.")
    else:
        print(f"Folder '{target_path}' does not exist.")

check_and_rename_folder()

config = json.load(open('../config.json', 'r'))


rtsp_urls = config["video_paths"]
print(rtsp_urls)

from subprocess import Popen

command = [f'python monitor_aisle.py --rtsp {url[0]} --cam_no {cam} --area_percentage {url[1]}' for cam, url in rtsp_urls.items()]

procs = [ Popen(i, shell = True) for i in command ]
for p in procs:
    p.wait()