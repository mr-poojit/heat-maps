import cv2
import numpy as np
from ultralytics import YOLO

# Open the video file 'carros.mp4' for reading
video = cv2.VideoCapture('carros.mp4')

# Load the YOLO model with the specified weights file
modelo = YOLO('yolov8n.pt')

# Create a blank image to accumulate object counts (initially all values are 1)
blankImage = np.ones([720, 1270], np.uint32)

while True:
    # Read a frame from the video
    check, img = video.read()
    
    # Resize the frame to 1270x720 pixels
    img = cv2.resize(img, (1270, 720))

    # Perform object detection on the frame
    objetos = modelo(img, stream=True)

    # Iterate through detected objects
    for objeto in objetos:
        # Extract bounding box information from detected objects
        info = objeto.boxes
        for box in info:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Get the confidence score and convert it to percentage
            conf = int(box.conf[0]*100)/100
            # Get the class label
            classe = int(box.cls[0])

            # Uncomment the following line to draw a rectangle around detected objects
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # If the detected object is a car (class 2)
            if classe == 2:
                # Increment the pixel values in the blankImage corresponding to the bounding box area
                blankImage[y1:y2, x1:x2] += 1

    # Normalize the accumulation image to the range 0-255
    blankImageNorm = 255 * ((blankImage - blankImage.min()) / (blankImage.max() - blankImage.min()))
    blankImageNorm = blankImageNorm.astype('uint8')
    
    # Apply Gaussian blur to the normalized image
    blankImageNorm = cv2.GaussianBlur(blankImageNorm, (9, 9), 0)

    # Apply a color map to the heat map image
    heatMap = cv2.applyColorMap(blankImageNorm, cv2.COLORMAP_JET)
    
    # Blend the heat map with the original image
    imgFinal = cv2.addWeighted(heatMap, 0.5, img, 0.5, 0)

    # Display the final image with heat map overlay
    cv2.imshow('HeatMap', imgFinal)
    
    # Wait for 1 millisecond and check for user input (e.g., to close the window)
    cv2.waitKey(1)
