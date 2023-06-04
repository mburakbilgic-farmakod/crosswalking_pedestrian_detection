import cv2

def detect_crosswalk(video_path):
    """
    Detect the crosswalk region in the video.
    :param video_path: Path to the video file.
    :return: List of coordinates of the crosswalk region.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    success, frame = video.read()

    if not success:
        print("Failed to read the video.")
        return None

    # Display the first frame
    cv2.imshow("Frame", frame)

    # Prompt the user to select the crosswalk region
    print("Select the crosswalk region by clicking four points in a clockwise manner.")

    # Store the selected points
    crosswalk_points = []

    # Mouse callback function to capture mouse events
    def select_crosswalk_region(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(crosswalk_points) < 4:
                crosswalk_points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Frame", frame)

    # Register the mouse callback function
    cv2.setMouseCallback("Frame", select_crosswalk_region)

    # Wait for the user to select the crosswalk region
    while len(crosswalk_points) < 4:
        cv2.waitKey(1)

    # Close the video window
    cv2.destroyAllWindows()

    # Define the crosswalk coordinates
    crosswalk_coordinates = [crosswalk_points]

    return crosswalk_coordinates


