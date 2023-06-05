import argparse

#Inner imports
from src.detect_crosswalk import detect_crosswalk
from src.detect_pedestrians import PedestrianDetector

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="dummy_video_1.mp4", help="Path to the video file if you want to use  webcam use 0")
parser.add_argument("--output", type=str, default ='output.mp4',  help="path to extracted video")
args = parser.parse_args()

out_path = args.output
crosswalk_coordinates = detect_crosswalk(args.video)
crossing_detection = PedestrianDetector(args.video, crosswalk_coordinates, out_path)
crossing_detection.detect_pedestrian()

