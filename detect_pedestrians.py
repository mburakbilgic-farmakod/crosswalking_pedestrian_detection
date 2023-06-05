import torch
from super_gradients.training import models
import cv2
import numpy as np
import math


class PedestrianDetector:
    def __init__(self, video_path, crosswalk_coordinates, output_path=None):
        """
        :param video_path: Path to the video file
        :param crosswalk_coordinates: Coordinates of the crosswalk
        :param output_path: Path to the output video file
        Here we are using the YOLOv5 model to detect pedestrians in the crosswalk region and draw an arrow indicating the direction of the pedestrian.
       """
        self.cap = cv2.VideoCapture(video_path)
        self.model = models.get("yolo_nas_s", pretrained_weights="coco")
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]
        self.crosswalk_coordinates = crosswalk_coordinates
        self.output_path = output_path
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.video_writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def detect_pedestrian(self):
        """
        Detect pedestrians in the crosswalk region and draw an arrow indicating the direction of the pedestrian.
        :return: None
        """
        count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            count += 1
            result = list(self.model.predict(frame, conf=0.35))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()

            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                classname = int(cls)
                class_name = self.class_names[classname]

                conf = math.ceil((confidence * 100)) / 100
                label = f'{class_name}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

                # Calculate the centroid of the bounding box
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2

                # Check if the centroid is inside the crosswalk region
                for crosswalk in self.crosswalk_coordinates:
                    crosswalk_poly = np.array(crosswalk)
                    is_inside_crosswalk = cv2.pointPolygonTest(crosswalk_poly, (centroid_x, centroid_y), False)
                    crosswalk2 = crosswalk.copy()
                    crosswalk2[1], crosswalk2[2] = crosswalk2[2], crosswalk2[1]
                    crosswalk_poly2 = np.array(crosswalk2)
                    is_inside_crosswalk2 = cv2.pointPolygonTest(crosswalk_poly2, (centroid_x, centroid_y), False)
                    # If inside the crosswalk, draw arrow indicating the direction
                    if is_inside_crosswalk >= 0:
                        arrow_start = (centroid_x, centroid_y)
                        arrow_end = (centroid_x, centroid_y - 40)  # Change the length of the arrow as needed
                        cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), thickness=2, tipLength=0.3)
                    elif is_inside_crosswalk2 >= 0:
                        arrow_start = (centroid_x, centroid_y)
                        arrow_end = (centroid_x, centroid_y - 40)
                        cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), thickness=2, tipLength=0.3)

            # Draw the manually defined crosswalk
            for crosswalk in self.crosswalk_coordinates:
                cv2.polylines(frame, [np.array(crosswalk)], isClosed=True, color=(0, 255, 0), thickness=2)
                crosswalk[1], crosswalk[2] = crosswalk[2], crosswalk[1]
                cv2.polylines(frame, [np.array(crosswalk)], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.imshow("video", frame)
            if self.video_writer:
                self.video_writer.write(frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()