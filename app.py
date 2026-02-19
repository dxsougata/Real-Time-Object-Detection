import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
from ultralytics import YOLO

# 1. Page Config
st.set_page_config(page_title="Real-Time Detection", layout="wide")
st.title("Real-Time Object Detection (YOLOv8)")

# 2. Load Model (Cached to prevent reloading on every frame)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 3. WebRTC Configuration (Crucial for bypass/hostel networks)
# Using Google's public STUN servers
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 4. Processing Callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Perform inference
    results = model.predict(img, conf=0.5)
    
    # Annotate frame
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 5. UI Layout
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

webrtc_streamer(
    key="yolo-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("Click 'Start' above to begin the live feed.")
