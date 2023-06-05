# Pedestrian Detector

The Pedestrian Detector is a Python-based application that utilizes the YOLO NAS model for real-time pedestrian detection in videos. It detects pedestrians in a video stream and highlights them with bounding boxes and arrows indicating their direction. The application also allows the user to specify a crosswalk region, and if a pedestrian is detected within the region, an arrow is displayed to indicate their direction of movement.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Supergradients

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/pedestrian-detector.git
   cd pedestrian-detector
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


## Usage

1. Run the application using the `main.py` script:
```bash
python main.py --video path/to/video_file.mp4 --crosswalk path/to/output_file.txt
```

| Argument      | Description                                             |
| ------------- | ------------------------------------------------------- |
| `--video`     | Path to the input video file. Use `0` for webcam.       |
| `--crosswalk` | Path to the output file to save crosswalk coordinates.  |

2. Select the Crosswalk Region:
   - The application will display the first frame of the video and prompt you to select the crosswalk region.
   - Click four points on the video frame in a clockwise manner to define the crosswalk region.
   - Once the region is selected, the application will start detecting pedestrians in real-time.

3. Observe the Pedestrian Detection:
   - The application will process each frame of the video and detect pedestrians using the YOLO NAS model.
   - Pedestrians within the crosswalk region will be highlighted with bounding boxes.
   - Arrows will be displayed above the pedestrians to indicate their direction of movement within the crosswalk.

4. Control the Application:
   - Press the 'q' key to stop the application and exit.

5. Output:
   - If the `--crosswalk` argument is provided, the crosswalk coordinates will be saved to the specified output file as a mp4 file.

## Additional Information

- The application uses the YOLO NAS model for pedestrian detection. The model is pretrained on the COCO dataset and can detect various objects, including pedestrians.
- The detected pedestrians are highlighted with bounding boxes and arrows indicating their direction of movement within the crosswalk region.
- The crosswalk region is manually defined by selecting four points on the video frame in a clockwise manner. The application uses the selected points to determine if a pedestrian is within the crosswalk.
- The application provides real-time detection and processing of video frames, allowing for quick analysis and monitoring of pedestrian movement.

That's it! You can now use the Pedestrian Detector to detect pedestrians and monitor their movement in videos. Enjoy!

