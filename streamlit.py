from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
import time

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO
import subprocess
import imageio_ffmpeg


# =========================================================
# 1. Configure the Streamlit page
# =========================================================

st.set_page_config(
    page_title="PPE Detection and Violation Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. Define application paths
# =========================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best_saad_26l_finetuned.pt"
CAPTURE_DIR = APP_DIR / "violation_captures"


# =========================================================
# 3. Load the YOLO model
# =========================================================

@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    """
    Load and cache the trained YOLO model.
    """
    return YOLO(str(model_path))


model: YOLO | None = None
model_ready = False
model_error: str | None = None

if not MODEL_PATH.exists():
    model_error = f"Model file not found: {MODEL_PATH}"
else:
    try:
        model = load_model(MODEL_PATH)
        model_ready = True
    except Exception as error:
        model_error = str(error)


# =========================================================
# 4. Read model class names
# =========================================================

def get_model_class_names(yolo_model: YOLO | None) -> list[str]:
    """
    Return the class names stored in the YOLO model.
    """
    if yolo_model is None:
        return []

    names = yolo_model.names

    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]

    return [str(name) for name in names]


CLASS_NAMES = get_model_class_names(model)


# =========================================================
# 5. Add custom CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #667085;
        margin-bottom: 22px;
    }

    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 20px rgba(16, 24, 40, 0.05);
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 19px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .card-description {
        font-size: 14px;
        color: #667085;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 13px;
        border: 1px solid #e5e7eb;
    }

    div.stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px;
        font-weight: 700;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 6. Define the violation configuration
# =========================================================

@dataclass
class ViolationConfig:
    """
    Store the settings used by the live video processor.
    """

    confidence: float
    image_size: int
    strategy: str
    violation_classes: set[str]
    person_class: str
    required_ppe_classes: set[str]
    grace_period: float
    absence_tolerance: float
    snapshot_cooldown: float
    save_snapshots: bool
    show_labels: bool
    show_confidence: bool


# =========================================================
# 7. Define shared violation state
# =========================================================

class ViolationState:
    """
    Store live violation information shared between Streamlit and WebRTC.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.active = False
        self.started_at: float | None = None
        self.last_violation_seen_at: float | None = None
        self.duration = 0.0
        self.reasons: list[str] = []
        self.event_count = 0
        self.last_snapshot_at: float | None = None
        self.last_snapshot_bgr: np.ndarray | None = None
        self.last_snapshot_path: str | None = None
        self.last_snapshot_time_text: str | None = None
        self.last_snapshot_reasons: list[str] = []

    def reset(self) -> None:
        """
        Reset all violation statistics and snapshots.
        """
        with self.lock:
            self.active = False
            self.started_at = None
            self.last_violation_seen_at = None
            self.duration = 0.0
            self.reasons = []
            self.event_count = 0
            self.last_snapshot_at = None
            self.last_snapshot_bgr = None
            self.last_snapshot_path = None
            self.last_snapshot_time_text = None
            self.last_snapshot_reasons = []

    def update(
        self,
        violation_detected: bool,
        reasons: list[str],
        annotated_frame: np.ndarray,
        config: ViolationConfig,
    ) -> tuple[bool, float]:
        """
        Update the violation timer and capture a snapshot when required.
        """
        now = time.monotonic()

        with self.lock:
            if violation_detected:
                self.last_violation_seen_at = now
                self.reasons = reasons

                if not self.active:
                    self.active = True
                    self.started_at = now
                    self.event_count += 1

            elif self.active:
                last_seen = self.last_violation_seen_at or now

                if now - last_seen > config.absence_tolerance:
                    self.active = False
                    self.started_at = None
                    self.last_violation_seen_at = None
                    self.duration = 0.0
                    self.reasons = []

            if self.active and self.started_at is not None:
                self.duration = now - self.started_at
            else:
                self.duration = 0.0

            should_capture = (
                self.active
                and self.duration >= config.grace_period
                and (
                    self.last_snapshot_at is None
                    or now - self.last_snapshot_at >= config.snapshot_cooldown
                )
            )

            if should_capture:
                self.last_snapshot_at = now
                self.last_snapshot_bgr = annotated_frame.copy()
                self.last_snapshot_time_text = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.last_snapshot_reasons = list(self.reasons)

                if config.save_snapshots:
                    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                    filename = datetime.now().strftime(
                        "ppe_violation_%Y%m%d_%H%M%S.jpg"
                    )
                    snapshot_path = CAPTURE_DIR / filename
                    cv2.imwrite(str(snapshot_path), annotated_frame)
                    self.last_snapshot_path = str(snapshot_path)
                else:
                    self.last_snapshot_path = None

            return self.active, self.duration

    def get_snapshot(self) -> dict:
        """
        Return a thread-safe copy of the current violation state.
        """
        with self.lock:
            return {
                "active": self.active,
                "duration": self.duration,
                "reasons": list(self.reasons),
                "event_count": self.event_count,
                "last_snapshot_bgr": (
                    None
                    if self.last_snapshot_bgr is None
                    else self.last_snapshot_bgr.copy()
                ),
                "last_snapshot_path": self.last_snapshot_path,
                "last_snapshot_time_text": self.last_snapshot_time_text,
                "last_snapshot_reasons": list(
                    self.last_snapshot_reasons
                ),
            }


if "violation_state" not in st.session_state:
    st.session_state.violation_state = ViolationState()

violation_state: ViolationState = st.session_state.violation_state


# =========================================================
# 8. Configure the sidebar
# =========================================================

def find_default_classes(
    class_names: list[str],
    keywords: tuple[str, ...],
) -> list[str]:
    """
    Find model classes that contain one of the supplied keywords.
    """
    defaults = []

    for class_name in class_names:
        normalized_name = class_name.lower().replace("-", "_").replace(" ", "_")

        if any(keyword in normalized_name for keyword in keywords):
            defaults.append(class_name)

    return defaults


default_violation_classes = find_default_classes(
    CLASS_NAMES,
    (
        "no_helmet",
        "nohelmet",
        "without_helmet",
        "no_vest",
        "novest",
        "without_vest",
        "no_glove",
        "noglove",
        "no_boot",
        "noboot",
    ),
)

default_person_classes = find_default_classes(
    CLASS_NAMES,
    ("person", "worker"),
)

default_required_ppe = find_default_classes(
    CLASS_NAMES,
    ("helmet", "vest"),
)

default_required_ppe = [
    class_name
    for class_name in default_required_ppe
    if class_name not in default_violation_classes
]

with st.sidebar:
    st.title("⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
    )

    image_size = st.selectbox(
        "Input image size",
        options=[320, 480, 640, 800],
        index=2,
    )

    show_labels = st.toggle(
        "Show class labels",
        value=True,
    )

    show_confidence = st.toggle(
        "Show confidence scores",
        value=True,
    )

    st.divider()
    st.markdown("### Violation Logic")

    strategy = st.radio(
        "Violation detection strategy",
        options=[
            "Explicit violation classes",
            "Scene-level missing PPE",
        ],
        help=(
            "Explicit violation classes are recommended when the model "
            "contains classes such as no_helmet or no_vest."
        ),
    )

    violation_classes = set(
        st.multiselect(
            "Violation classes",
            options=CLASS_NAMES,
            default=default_violation_classes,
            disabled=strategy != "Explicit violation classes",
        )
    )

    person_class = st.selectbox(
        "Person or worker class",
        options=CLASS_NAMES or [""],
        index=(
            CLASS_NAMES.index(default_person_classes[0])
            if default_person_classes
            else 0
        ),
        disabled=strategy != "Scene-level missing PPE",
    )

    required_ppe_classes = set(
        st.multiselect(
            "Required PPE classes",
            options=CLASS_NAMES,
            default=default_required_ppe,
            disabled=strategy != "Scene-level missing PPE",
        )
    )

    grace_period = st.slider(
        "Grace period before flagging",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="A snapshot is taken only after the violation lasts this long.",
    )

    absence_tolerance = st.slider(
        "Detection gap tolerance",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.1,
        help="Prevents the timer from resetting because of brief missed detections.",
    )

    snapshot_cooldown = st.slider(
        "Snapshot cooldown",
        min_value=1.0,
        max_value=60.0,
        value=10.0,
        step=1.0,
        help="Minimum number of seconds between violation snapshots.",
    )

    save_snapshots = st.toggle(
        "Save snapshots to disk",
        value=True,
    )

    st.divider()
    st.markdown("### Model Status")

    if model_ready:
        st.success("Model loaded successfully")
        st.caption(str(MODEL_PATH))
    else:
        st.error("Model could not be loaded")

        if model_error:
            st.code(model_error)

    if st.button("Reset violation history"):
        violation_state.reset()
        st.success("Violation history has been reset.")


current_config = ViolationConfig(
    confidence=confidence,
    image_size=image_size,
    strategy=strategy,
    violation_classes=violation_classes,
    person_class=person_class,
    required_ppe_classes=required_ppe_classes,
    grace_period=grace_period,
    absence_tolerance=absence_tolerance,
    snapshot_cooldown=snapshot_cooldown,
    save_snapshots=save_snapshots,
    show_labels=show_labels,
    show_confidence=show_confidence,
)


# =========================================================
# 9. Create detection helper functions
# =========================================================

def extract_detection_data(result) -> tuple[list[str], list[float]]:
    """
    Extract class names and confidence scores from one YOLO result.
    """
    detected_classes: list[str] = []
    confidence_scores: list[float] = []

    if len(result.boxes) == 0:
        return detected_classes, confidence_scores

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidence_scores = result.boxes.conf.cpu().numpy().tolist()

    detected_classes = [
        str(result.names[class_id])
        for class_id in class_ids
    ]

    return detected_classes, confidence_scores


def evaluate_violation(
    detected_classes: list[str],
    config: ViolationConfig,
) -> tuple[bool, list[str]]:
    """
    Decide whether the current frame contains a PPE violation.
    """
    detected_set = set(detected_classes)

    if config.strategy == "Explicit violation classes":
        matched_classes = sorted(
            detected_set.intersection(config.violation_classes)
        )
        reasons = [
            f"Detected violation class: {class_name}"
            for class_name in matched_classes
        ]
        return bool(matched_classes), reasons

    person_detected = config.person_class in detected_set

    if not person_detected:
        return False, []

    missing_classes = sorted(
        config.required_ppe_classes.difference(detected_set)
    )

    reasons = [
        f"Missing required PPE: {class_name}"
        for class_name in missing_classes
    ]

    return bool(missing_classes), reasons


def add_violation_banner(
    frame: np.ndarray,
    active: bool,
    duration: float,
    reasons: list[str],
    grace_period: float,
) -> np.ndarray:
    """
    Add a live violation banner and timer to the video frame.
    """
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]

    if active:
        confirmed = duration >= grace_period
        banner_text = (
            f"PPE VIOLATION - {duration:.1f}s"
            if confirmed
            else f"Checking violation - {duration:.1f}s"
        )
        banner_color = (0, 0, 220) if confirmed else (0, 165, 255)

        cv2.rectangle(
            output,
            (0, 0),
            (frame_width, 75),
            banner_color,
            thickness=-1,
        )

        cv2.putText(
            output,
            banner_text,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        reason_text = " | ".join(reasons[:2])

        cv2.putText(
            output,
            reason_text[:90],
            (18, 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.rectangle(
            output,
            (0, 0),
            (frame_width, 42),
            (40, 150, 70),
            thickness=-1,
        )

        cv2.putText(
            output,
            "PPE status: compliant",
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def run_image_detection(
    image: Image.Image,
    config: ViolationConfig,
) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Run YOLO detection on one uploaded or captured image.
    """
    if model is None:
        raise RuntimeError("The YOLO model is not available.")

    image_array = np.array(image)

    results = model.predict(
        source=image_array,
        conf=config.confidence,
        imgsz=config.image_size,
        verbose=False,
    )

    result = results[0]

    annotated_image = result.plot(
        labels=config.show_labels,
        conf=config.show_confidence,
    )

    annotated_image = annotated_image[:, :, ::-1]

    detected_classes, confidence_scores = extract_detection_data(result)

    return annotated_image, detected_classes, confidence_scores


# =========================================================
# 10. Create the live video processor
# =========================================================

class PPEViolationProcessor(VideoProcessorBase):
    """
    Detect PPE and monitor violations in every live video frame.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        state: ViolationState,
        config: ViolationConfig,
    ) -> None:
        self.model = yolo_model
        self.state = state
        self.config = config

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process one live video frame.
        """
        image = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=image,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            verbose=False,
        )

        result = results[0]

        annotated_frame = result.plot(
            labels=self.config.show_labels,
            conf=self.config.show_confidence,
        )

        detected_classes, _ = extract_detection_data(result)

        violation_detected, reasons = evaluate_violation(
            detected_classes,
            self.config,
        )

        active, duration = self.state.update(
            violation_detected=violation_detected,
            reasons=reasons,
            annotated_frame=annotated_frame,
            config=self.config,
        )

        display_frame = add_violation_banner(
            frame=annotated_frame,
            active=active,
            duration=duration,
            reasons=reasons,
            grace_period=self.config.grace_period,
        )

        return av.VideoFrame.from_ndarray(
            display_frame,
            format="bgr24",
        )


# =========================================================
# 11. Create result display functions
# =========================================================

def display_image_results(
    annotated_image: np.ndarray,
    detected_classes: list[str],
    confidence_scores: list[float],
    config: ViolationConfig,
) -> None:
    """
    Display image detection results and PPE status.
    """
    st.image(
        annotated_image,
        caption="Detection Result",
        use_container_width=True,
    )

    violation_detected, reasons = evaluate_violation(
        detected_classes,
        config,
    )

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Total Detections",
        len(detected_classes),
    )

    metric_2.metric(
        "Detected Classes",
        len(set(detected_classes)),
    )

    metric_3.metric(
        "PPE Status",
        "Violation" if violation_detected else "Compliant",
    )

    if violation_detected:
        st.error("PPE violation detected")

        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.success("No PPE violation detected")

    if detected_classes:
        class_counts = Counter(detected_classes)

        with st.expander("Detection details"):
            for class_name, count in class_counts.items():
                st.write(f"**{class_name}:** {count}")

            for index, (class_name, score) in enumerate(
                zip(detected_classes, confidence_scores),
                start=1,
            ):
                st.write(
                    f"{index}. {class_name} — {score:.1%}"
                )


def render_live_status_body() -> None:
    """
    Display the current violation timer and latest snapshot.
    """
    state_data = violation_state.get_snapshot()

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Current Status",
        "Violation" if state_data["active"] else "Compliant",
    )

    metric_2.metric(
        "Current Violation Duration",
        f'{state_data["duration"]:.1f} s',
    )

    metric_3.metric(
        "Violation Events",
        state_data["event_count"],
    )

    if state_data["active"]:
        st.error(
            "Active PPE violation: "
            + " | ".join(state_data["reasons"])
        )
    else:
        st.success("No active PPE violation")

    snapshot_bgr = state_data["last_snapshot_bgr"]

    if snapshot_bgr is not None:
        st.markdown("### Latest Violation Snapshot")

        snapshot_rgb = cv2.cvtColor(
            snapshot_bgr,
            cv2.COLOR_BGR2RGB,
        )

        st.image(
            snapshot_rgb,
            use_container_width=True,
        )

        if state_data["last_snapshot_time_text"]:
            st.caption(
                "Captured at: "
                + state_data["last_snapshot_time_text"]
            )

        if state_data["last_snapshot_reasons"]:
            st.caption(
                "Reason: "
                + " | ".join(
                    state_data["last_snapshot_reasons"]
                )
            )

        if state_data["last_snapshot_path"]:
            st.code(state_data["last_snapshot_path"])


# =========================================================
# Video-processing imports
# =========================================================

from collections import Counter
from pathlib import Path
import os
import tempfile

import cv2
import numpy as np
from PIL import Image


def convert_video_to_browser_mp4(
    input_path: str,
    output_path: str,
) -> None:
    """
    Convert an OpenCV-generated video into a browser-compatible
    H.264 MP4 file.
    """

    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg video conversion failed:\n"
            + result.stderr[-2000:]
        )

# =========================================================
# Process an uploaded video frame by frame
# =========================================================

def process_uploaded_video(
    uploaded_video,
    config,
    progress_bar=None,
    status_placeholder=None,
):
    """
    Process an uploaded video frame by frame using run_image_detection().

    Returns
    -------
    output_video_bytes : bytes
        Annotated MP4 video.
    detection_counts : dict
        Number of detections recorded for each class across all frames.
    processed_frames : int
        Total number of processed frames.
    """

    input_path = None
    intermediate_output_path = None
    output_path = None
    video_capture = None
    video_writer = None

    detection_counts = Counter()
    processed_frames = 0

    try:
        # -------------------------------------------------
        # Save the uploaded video temporarily
        # -------------------------------------------------

        input_suffix = Path(uploaded_video.name).suffix.lower()

        if not input_suffix:
            input_suffix = ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=input_suffix,
        ) as temporary_input:
            temporary_input.write(uploaded_video.getbuffer())
            input_path = temporary_input.name

        # -------------------------------------------------
        # Open the uploaded video
        # -------------------------------------------------

        video_capture = cv2.VideoCapture(input_path)

        if not video_capture.isOpened():
            raise ValueError("OpenCV could not open the uploaded video.")

        fps = video_capture.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 25.0

        total_frames = int(
            video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Read the first frame to determine video dimensions
        frame_available, frame = video_capture.read()

        if not frame_available:
            raise ValueError(
                "The uploaded video does not contain readable frames."
            )

        frame_height, frame_width = frame.shape[:2]

        # -------------------------------------------------
        # Create the output MP4 file
        # -------------------------------------------------

        temporary_output = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".avi",
        )

        intermediate_output_path = temporary_output.name
        temporary_output.close()

        temporary_browser_output = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        )

        output_path = temporary_browser_output.name
        temporary_browser_output.close()

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        video_writer = cv2.VideoWriter(
            intermediate_output_path,
            fourcc,
            fps,
            (frame_width, frame_height),
        )

        if not video_writer.isOpened():
            raise ValueError(
                "OpenCV could not create the intermediate video."
            )


        # -------------------------------------------------
        # Process every video frame
        # -------------------------------------------------

        while frame_available:
            processed_frames += 1

            # OpenCV uses BGR, while PIL uses RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            pil_frame = Image.fromarray(rgb_frame)

            (
                annotated_image,
                detected_classes,
                confidence_scores,
            ) = run_image_detection(
                pil_frame,
                config,
            )

            # Convert the annotated image into an RGB array
            if isinstance(annotated_image, Image.Image):
                annotated_rgb = np.asarray(
                    annotated_image.convert("RGB")
                )
            else:
                annotated_rgb = np.asarray(
                    annotated_image
                )

            # Ensure that the processed frame has three channels
            if annotated_rgb.ndim == 2:
                annotated_rgb = cv2.cvtColor(
                    annotated_rgb,
                    cv2.COLOR_GRAY2RGB,
                )

            if annotated_rgb.shape[2] == 4:
                annotated_rgb = cv2.cvtColor(
                    annotated_rgb,
                    cv2.COLOR_RGBA2RGB,
                )

            # Ensure that every frame has the original dimensions
            if (
                annotated_rgb.shape[1] != frame_width
                or annotated_rgb.shape[0] != frame_height
            ):
                annotated_rgb = cv2.resize(
                    annotated_rgb,
                    (frame_width, frame_height),
                )

            annotated_bgr = cv2.cvtColor(
                annotated_rgb,
                cv2.COLOR_RGB2BGR,
            )

            video_writer.write(annotated_bgr)

            # Count detections across frames
            detection_counts.update(
                str(class_name)
                for class_name in detected_classes
            )

            # Update progress periodically
            if (
                progress_bar is not None
                and total_frames > 0
                and (
                    processed_frames % 5 == 0
                    or processed_frames == total_frames
                )
            ):
                progress_value = min(
                    processed_frames / total_frames,
                    1.0,
                )

                progress_bar.progress(progress_value)

            if (
                status_placeholder is not None
                and (
                    processed_frames % 10 == 0
                    or processed_frames == total_frames
                )
            ):
                if total_frames > 0:
                    status_placeholder.info(
                        f"Processing frame "
                        f"{processed_frames:,} of "
                        f"{total_frames:,}"
                    )
                else:
                    status_placeholder.info(
                        f"Processed "
                        f"{processed_frames:,} frames"
                    )

            frame_available, frame = video_capture.read()

        # Release the files before reading the output
        video_capture.release()
        video_capture = None

        video_writer.release()
        video_writer = None

        if status_placeholder is not None:
            status_placeholder.info(
                "Converting video to browser-compatible MP4..."
            )

        convert_video_to_browser_mp4(
            input_path=intermediate_output_path,
            output_path=output_path,
            )

        if progress_bar is not None:
            progress_bar.progress(1.0)

        output_video_bytes = Path(output_path).read_bytes()

        return (
            output_video_bytes,
            dict(detection_counts),
            processed_frames,
        )

    finally:
        if video_capture is not None:
            video_capture.release()

        if video_writer is not None:
            video_writer.release()

        for temporary_path in [
            input_path,
            intermediate_output_path,
            output_path,
        ]:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass


# =========================================================
# 12. Create the page header and tabs
# =========================================================

st.markdown(
    '<div class="main-title">🦺 PPE Detection and Violation Monitor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Detect PPE from live video, capture violation evidence,
        monitor violation duration, or analyze a single image.
    </div>
    """,
    unsafe_allow_html=True,
)

live_tab, camera_tab, upload_tab, video_upload_tab, about_tab = st.tabs(
    [
        "🎥 Live Video",
        "📷 Camera",
        "🖼️ Upload Image",
        "📹 Upload Video",
        "ℹ️ About",
    ]
)


# =========================================================
# 13. Create the live video tab
# =========================================================

with live_tab:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">Live PPE Violation Monitoring</div>
            <div class="card-description">
                Start the webcam to detect PPE continuously. The application
                flags violations, measures their duration, and captures evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error(
            "The model is not available. Check the model path in the sidebar."
        )
    elif (
        strategy == "Explicit violation classes"
        and not violation_classes
    ):
        st.warning(
            "Select at least one violation class in the sidebar."
        )
    elif (
        strategy == "Scene-level missing PPE"
        and not required_ppe_classes
    ):
        st.warning(
            "Select at least one required PPE class in the sidebar."
        )
    else:
        webrtc_streamer(
            key="ppe-violation-camera",
            video_processor_factory=lambda: PPEViolationProcessor(
                yolo_model=model,
                state=violation_state,
                config=current_config,
            ),
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

        st.info(
            "After changing settings, stop and restart the live video "
            "so the processor uses the new configuration."
        )

    st.markdown("### Live Violation Status")

    if hasattr(st, "fragment"):
        @st.fragment(run_every=1.0)
        def render_live_status() -> None:
            render_live_status_body()

        render_live_status()
    else:
        render_live_status_body()

        if st.button("Refresh live status"):
            st.rerun()


# =========================================================
# 14. Create the camera tab
# =========================================================

with camera_tab:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">Camera Snapshot Detection</div>
            <div class="card-description">
                Take one picture with the camera and analyze its PPE status.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    camera_image = st.camera_input(
        "Take a picture",
        label_visibility="collapsed",
    )

    if camera_image is not None:
        try:
            camera_pil_image = Image.open(
                camera_image
            ).convert("RGB")

            original_column, result_column = st.columns(2)

            with original_column:
                st.markdown("### Original Image")
                st.image(
                    camera_pil_image,
                    use_container_width=True,
                )

                camera_button = st.button(
                    "🔍 Detect Camera Image",
                    type="primary",
                    key="camera_detection",
                )

            if camera_button:
                with result_column:
                    with st.spinner("Running PPE detection..."):
                        (
                            annotated_image,
                            detected_classes,
                            confidence_scores,
                        ) = run_image_detection(
                            camera_pil_image,
                            current_config,
                        )

                    display_image_results(
                        annotated_image,
                        detected_classes,
                        confidence_scores,
                        current_config,
                    )

        except Exception as error:
            st.error(f"Unable to process the camera image: {error}")

# =========================================================
# 15. Create the upload image tab
# =========================================================

with upload_tab:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">Upload Image Detection</div>
            <div class="card-description">
                Upload a JPG, JPEG, or PNG image and analyze its PPE status.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            uploaded_pil_image = Image.open(
                uploaded_file
            ).convert("RGB")

            original_column, result_column = st.columns(2)

            with original_column:
                st.markdown("### Original Image")
                st.image(
                    uploaded_pil_image,
                    use_container_width=True,
                )

                upload_button = st.button(
                    "🔍 Detect Uploaded Image",
                    type="primary",
                    key="upload_detection",
                )

            if upload_button:
                with result_column:
                    with st.spinner("Running PPE detection..."):
                        (
                            annotated_image,
                            detected_classes,
                            confidence_scores,
                        ) = run_image_detection(
                            uploaded_pil_image,
                            current_config,
                        )

                    display_image_results(
                        annotated_image,
                        detected_classes,
                        confidence_scores,
                        current_config,
                    )

        except Exception as error:
            st.error(f"Unable to process the uploaded image: {error}")

with video_upload_tab:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">Upload Video Detection</div>
            <div class="card-description">
                Upload a video and run PPE detection on every frame.
                The processed video will contain detection annotations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "mpeg",
            "mpg",
        ],
        label_visibility="collapsed",
        key="ppe_video_uploader",
    )

    if uploaded_video is not None:
        st.markdown("### Original Video")

        original_video_bytes = uploaded_video.getvalue()

        st.video(original_video_bytes)

        file_size_mb = len(original_video_bytes) / (
            1024 * 1024
        )

        st.caption(
            f"File: {uploaded_video.name} · "
            f"Size: {file_size_mb:.2f} MB"
        )

        st.info(
            "Video processing may take longer for high-resolution "
            "or long videos. Audio is not retained in the processed video."
        )

        # ---------------------------------------------
        # Validate model and violation configuration
        # ---------------------------------------------

        video_configuration_valid = True

        if not model_ready:
            st.error(
                "The model is not available. Check the model "
                "path in the sidebar."
            )
            video_configuration_valid = False

        elif (
            strategy == "Explicit violation classes"
            and not violation_classes
        ):
            st.warning(
                "Select at least one violation class "
                "in the sidebar."
            )
            video_configuration_valid = False

        elif (
            strategy == "Scene-level missing PPE"
            and not required_ppe_classes
        ):
            st.warning(
                "Select at least one required PPE class "
                "in the sidebar."
            )
            video_configuration_valid = False

        # ---------------------------------------------
        # Process the uploaded video
        # ---------------------------------------------

        if video_configuration_valid:
            process_video_button = st.button(
                "🎬 Process Uploaded Video",
                type="primary",
                key="process_uploaded_video",
                use_container_width=True,
            )

            if process_video_button:
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                try:
                    with st.spinner(
                        "Running PPE detection on the video..."
                    ):
                        (
                            processed_video_bytes,
                            video_detection_counts,
                            processed_frame_count,
                        ) = process_uploaded_video(
                            uploaded_video=uploaded_video,
                            config=current_config,
                            progress_bar=progress_bar,
                            status_placeholder=status_placeholder,
                        )

                    # Store the result so it survives Streamlit reruns
                    st.session_state[
                        "processed_video_bytes"
                    ] = processed_video_bytes

                    st.session_state[
                        "processed_video_detection_counts"
                    ] = video_detection_counts

                    st.session_state[
                        "processed_video_frame_count"
                    ] = processed_frame_count

                    st.session_state[
                        "processed_video_source_name"
                    ] = uploaded_video.name

                    status_placeholder.success(
                        f"Video processing completed. "
                        f"{processed_frame_count:,} frames were analyzed."
                    )

                except Exception as error:
                    status_placeholder.empty()

                    st.error(
                        f"Unable to process the uploaded video: "
                        f"{error}"
                    )

        # ---------------------------------------------
        # Display the previously processed result
        # ---------------------------------------------

        result_available = (
            "processed_video_bytes" in st.session_state
            and st.session_state.get(
                "processed_video_source_name"
            )
            == uploaded_video.name
        )

        if result_available:
            st.divider()
            st.markdown("### Processed Video")

            processed_video_bytes = st.session_state[
                "processed_video_bytes"
            ]

            st.video(
                    processed_video_bytes,
                    format="video/mp4",
                )

            processed_frame_count = st.session_state.get(
                "processed_video_frame_count",
                0,
            )

            detection_counts = st.session_state.get(
                "processed_video_detection_counts",
                {},
            )

            metric_column_1, metric_column_2 = st.columns(2)

            with metric_column_1:
                st.metric(
                    "Frames Processed",
                    f"{processed_frame_count:,}",
                )

            with metric_column_2:
                st.metric(
                    "Total Detections",
                    f"{sum(detection_counts.values()):,}",
                )

            if detection_counts:
                st.markdown("### Detection Summary")

                detection_summary = [
                    {
                        "Class": class_name,
                        "Detections Across Frames": count,
                    }
                    for class_name, count in sorted(
                        detection_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ]

                st.dataframe(
                    detection_summary,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    "No PPE classes were detected in the video."
                )

            output_filename = (
                f"{Path(uploaded_video.name).stem}"
                f"_ppe_detected.mp4"
            )

            st.download_button(
                label="⬇️ Download Processed Video",
                data=processed_video_bytes,
                file_name=output_filename,
                mime="video/mp4",
                type="primary",
                use_container_width=True,
                key="download_processed_video",
            )

# =========================================================
# 17. Create the About tab
# =========================================================

with about_tab:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">How Violation Monitoring Works</div>
            <div class="card-description">
                This application supports two different violation strategies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Explicit Violation Classes

        This is the recommended strategy when the model contains classes such
        as `no_helmet`, `no_vest`, or `without_gloves`.

        ### Scene-Level Missing PPE

        This strategy checks whether a person or worker is detected and whether
        the required PPE classes appear anywhere in the same frame.

        This is only a scene-level approximation. It does not reliably match a
        specific helmet or vest to a specific worker when multiple people are
        present.

        ### Snapshot Storage

        Saved violation snapshots are stored in:

        ```text
        app/violation_captures/
        ```

        ### Model Path

        ```text
        app/models/best.pt
        ```
        """
    )


# =========================================================
# 18. Add the page footer
# =========================================================

st.divider()

st.caption(
    "PPE Detection System · Powered by Streamlit, WebRTC, and YOLO"
)
