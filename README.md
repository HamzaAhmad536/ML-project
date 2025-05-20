# Real-time Eye State Detection

This project uses MediaPipe and a CNN model to detect whether a person's eyes are open or closed in real-time using a webcam.

## Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your webcam is connected and working.

2. Run the eye detection script:
```bash
python eye_detection.py
```

3. The program will:
   - Train the model on the MRL Eye dataset
   - Open your webcam
   - Display the video feed with eye state detection
   - Show rectangles around detected eyes
   - Display whether eyes are open or closed

4. Press 'q' to quit the program.

## Features

- Real-time eye state detection
- Uses MediaPipe for accurate face and eye landmark detection
- CNN model trained on the MRL Eye dataset
- Visual feedback with bounding boxes and state labels
- Processes both eyes independently for better accuracy

## Dataset

The project uses the MRL Eye dataset, which contains:
- Open eye images
- Closed eye images
- Split into training, validation, and test sets 