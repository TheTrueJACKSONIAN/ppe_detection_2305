import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

import av

model = YOLO("models/best.pt")

st.title("Live PPE Detection")

class PPEProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        results = model.predict(
            img,
            conf=0.25,
            verbose=False
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

webrtc_streamer(
    key="ppe-camera",
    video_processor_factory=PPEProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)