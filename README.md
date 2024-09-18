

# YOLO Object Detection with Heatmap Visualization

This project uses the YOLO (You Only Look Once) object detection model to identify cars in video frames and generates a heatmap visualization based on object occurrences.

## Prerequisites

1. **Python**: Ensure you have Python 3.6 or higher installed.
2. **Dependencies**: Install the required Python packages using the following command:

    ```bash
    pip install opencv-python-headless numpy ultralytics
    ```

3. **YOLO Model Weights**: Download the YOLO model weights file `yolov8n.pt`. Place it in the same directory as your script.

## Usage

1. **Prepare Your Video File**: Ensure you have a video file named `carros.mp4` in the same directory as the script.

2. **Run the Script**: Execute the Python script using the command:

    ```bash
    python your_script_name.py
    ```

    Replace `your_script_name.py` with the name of your Python script file.

## Script Explanation

- **Imports**:
    ```python
    import cv2
    import numpy as np
    from ultralytics import YOLO
    ```

- **Open Video File**:
    ```python
    video = cv2.VideoCapture('carros.mp4')
    ```

- **Load YOLO Model**:
    ```python
    modelo = YOLO('yolov8n.pt')
    ```

- **Create Blank Image**:
    ```python
    blankImage = np.ones([720, 1270], np.uint32)
    ```

- **Processing Video Frames**:
    ```python
    while True:
        check, img = video.read()
        img = cv2.resize(img, (1270, 720))
        objetos = modelo(img, stream=True)
        for objeto in objetos:
            info = objeto.boxes
            for box in info:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = int(box.conf[0]*100)/100
                classe = int(box.cls[0])
                if classe == 2:
                    blankImage[y1:y2, x1:x2] += 1
    ```

- **Generate Heatmap**:
    ```python
    blankImageNorm = 255 * ((blankImage - blankImage.min()) / (blankImage.max() - blankImage.min()))
    blankImageNorm = blankImageNorm.astype('uint8')
    blankImageNorm = cv2.GaussianBlur(blankImageNorm, (9, 9), 0)
    heatMap = cv2.applyColorMap(blankImageNorm, cv2.COLORMAP_JET)
    imgFinal = cv2.addWeighted(heatMap, 0.5, img, 0.5, 0)
    cv2.imshow('HeatMap', imgFinal)
    ```

- **Display and Close**:
    ```python
    cv2.waitKey(1)
    ```

## Notes

- The script assumes a specific class index for cars. Ensure the class index matches the configuration of your YOLO model.
- Adjust the `cv2.addWeighted` parameters to change the blending of the heatmap with the original image.
- To stop the script, close the display window or interrupt the process (e.g., by pressing `Ctrl+C`).

## Troubleshooting

- If the video does not play, check the path to `carros.mp4` and ensure the file exists.
- Ensure all dependencies are correctly installed and compatible with your Python version.

