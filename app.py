import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
from ultralytics import YOLO 
# 1. Load your model
model = YOLO("yolov8n.pt") 

# 2. Define the processing logic
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Perform Detection
    results = model.predict(img, conf=0.5)
    
    # Draw results on the frame
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 3. Streamlit UI
st.title("Real-Time Object Detection")
st.write("Click 'Start' to open your webcam")

# STUN servers help bypass firewalls (important for hostel/college networks!)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)