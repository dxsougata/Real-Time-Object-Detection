# Real-Time-Object-Detection
The app leverages WebRTC to stream webcam video directly through the browser, making it ideal for low-latency, real-time inference in a cloud-deployed environment.
## Tech Stack
Framework: Streamlit

Real-time Streaming: Streamlit-webrtc

Deep Learning: Ultralytics (YOLO)

Image Processing: OpenCV & PyAV

Deployment: Hugging Face Spaces / Render

## Key Features
Live Webcam Inference: Processes video frames in real-time using webrtc_streamer.

Optimized Performance: Uses asynchronous frame processing to maintain a smooth UI.

Cross-Platform: Designed to run seamlessly on local machines and cloud platforms like Hugging Face.
