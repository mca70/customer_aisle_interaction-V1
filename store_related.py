import cv2

def detect_staff_and_trolley(model, frame):
    """
        this function use to load custom trained staff and trolley models
        and detect the object in the given frame
    """
    results = model(frame, conf = 0.25, verbose = False)[0]

    if results.boxes.xyxy.numel() == 0:
        return False

    for result in list(results.boxes):
        bbox = result.xyxy.tolist()
        cls = result.cls.item()
        conf = result.conf.item()

        x1, y1, x2, y2 = bbox[0]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (192, 192, 192), 2)
    
    return True
    
def detect_staff_and_trolley_batch(model, frames):
    results = model(frames, conf = 0.25, verbose = False)
    
    count = 0
    for i in range(len(results)):
        if results[i].boxes.xyxy.numel() == 0:
            continue
        
        count += 1
    
    return count
            


def draw_square_around_point(image, point, size, color=(0, 255, 255), thickness=2):
    """
    Draws a square around a given point on the image.

    :param image: The image on which to draw the square.
    :param point: The (x, y) coordinates of the center point.
    :param size: The size of the square (length of the side).
    :param color: The color of the square (B, G, R).
    :param thickness: The thickness of the square border.
    """
    x, y = point

    # Calculate the top-left and bottom-right corners of the square
    half_size = size // 2
    top_left = (x - half_size, y - half_size)
    bottom_right = (x + half_size, y + half_size)

    # Draw the square on the image
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    
    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])