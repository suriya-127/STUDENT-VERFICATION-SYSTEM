import os
import cv2
import av
from streamlit_webrtc import VideoProcessorBase


class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.recording = False
        self.writer = None
        self.output_path = os.path.join("recordings", "student_answer.mp4")

        os.makedirs("recordings", exist_ok=True)

    def start_recording(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, 20.0, (width, height))
        self.recording = True

    def write_frame(self, image):
        if self.recording and self.writer is not None:
            self.writer.write(image)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.recording = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            self.write_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
