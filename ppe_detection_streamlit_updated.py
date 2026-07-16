from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from threading import Lock
import time

import av
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import (
    VideoHTMLAttributes,
    VideoProcessorBase,
    webrtc_streamer,
)
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
MODEL_DIR = APP_DIR / "models"
CAPTURE_DIR = APP_DIR / "violation_captures"
DEFAULT_MODEL_NAME = "best_saad_26l_finetuned.pt"

# Use FP16 (half precision) inference when a CUDA GPU is available.
# This roughly halves inference time on the GPU without hurting accuracy.
USE_HALF = torch.cuda.is_available()


# =========================================================
# 3. Load YOLO models
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    """
    Load and cache a trained YOLO model.

    A string is used as the cache key so every selected path loads its
    corresponding model once, and previously loaded models remain cached.
    """
    return YOLO(model_path)


def get_model_class_names(yolo_model: YOLO | None) -> list[str]:
    """
    Return the class names stored in a YOLO model.
    """
    if yolo_model is None:
        return []

    names = yolo_model.names

    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]

    return [str(name) for name in names]


def get_combined_class_names(yolo_models: list[YOLO]) -> list[str]:
    """
    Return the union of class names across all selected models.

    The original order is preserved and duplicates are removed, so the
    sidebar class selectors work with every selected model at once.
    """
    combined: list[str] = []
    seen: set[str] = set()

    for yolo_model in yolo_models:
        for class_name in get_model_class_names(yolo_model):
            if class_name not in seen:
                seen.add(class_name)
                combined.append(class_name)

    return combined


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
        max-width: 1700px;
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

    /* Enlarge the WebRTC live video feed */
    [data-testid="stCustomComponentV1"] iframe {
        height: 75vh !important;
        min-height: 480px;
        width: 100% !important;
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
    show_banner: bool
    show_banner_timer: bool
    show_banner_reasons: bool


# =========================================================
# 7. Define the live runtime shared with the WebRTC thread
# =========================================================

class LiveRuntime:
    """
    Store the currently selected models and sidebar configuration.

    Streamlit reruns the page when a sidebar widget changes, while the WebRTC
    video processor can continue running in another thread. The processor must
    therefore read the latest models and settings for every frame instead of
    keeping the values captured when the camera first started.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.models: list[YOLO] = []
        self.config: ViolationConfig | None = None
        self.model_paths: list[str] = []

    def update(
        self,
        models: list[YOLO],
        config: ViolationConfig,
        model_paths: list[Path],
    ) -> None:
        """Replace the active models and settings atomically."""
        with self.lock:
            self.models = list(models)
            self.config = config
            self.model_paths = [str(path) for path in model_paths]

    def get(self) -> tuple[list[YOLO], ViolationConfig | None]:
        """Return the latest models and configuration."""
        with self.lock:
            return list(self.models), self.config


# =========================================================
# 8. Define shared violation state
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

    def clear_saved_path_reference(self) -> None:
        """
        Stop displaying an old disk path when disk saving is disabled.

        This does not delete files that were saved before the option was
        disabled.
        """
        with self.lock:
            self.last_snapshot_path = None

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


if (
    "violation_state" not in st.session_state
    or not hasattr(
        st.session_state.violation_state,
        "clear_saved_path_reference",
    )
):
    st.session_state.violation_state = ViolationState()

if (
    "live_runtime" not in st.session_state
    or not hasattr(st.session_state.live_runtime, "get")
    or not hasattr(st.session_state.live_runtime, "update")
    or not hasattr(st.session_state.live_runtime, "models")
):
    st.session_state.live_runtime = LiveRuntime()

if "live_camera_enabled" not in st.session_state:
    st.session_state.live_camera_enabled = False

if "snapshot_camera_enabled" not in st.session_state:
    st.session_state.snapshot_camera_enabled = False

violation_state: ViolationState = st.session_state.violation_state
live_runtime: LiveRuntime = st.session_state.live_runtime


# =========================================================
# 9. Configure the sidebar
# =========================================================

def find_default_classes(
    class_names: list[str],
    keywords: tuple[str, ...],
) -> list[str]:
    """
    Find model classes that contain one of the supplied keywords.
    """
    defaults: list[str] = []

    for class_name in class_names:
        normalized_name = class_name.lower().replace("-", "_").replace(" ", "_")

        if any(keyword in normalized_name for keyword in keywords):
            defaults.append(class_name)

    return defaults


def clear_processed_results() -> None:
    """Remove cached results that belong to an older model or configuration."""
    result_keys = [
        "processed_video_bytes",
        "processed_video_detection_counts",
        "processed_video_frame_count",
        "processed_video_source_name",
        "processed_video_settings_signature",
    ]

    for key in result_keys:
        st.session_state.pop(key, None)


MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_files = sorted(
    (
        path
        for path in MODEL_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pt"
    ),
    key=lambda path: path.name.lower(),
)
model_names = [path.name for path in model_files]

models: list[YOLO] = []
model_ready = False
model_errors: list[str] = []
MODEL_PATHS: list[Path] = []
LOADED_MODEL_NAMES: list[str] = []
CLASS_NAMES: list[str] = []

with st.sidebar:
    st.title("⚙️ Detection Settings")

    st.markdown("### Detection Models")

    if model_names:
        default_model_selection = (
            [DEFAULT_MODEL_NAME]
            if DEFAULT_MODEL_NAME in model_names
            else [model_names[0]]
        )

        selected_model_names = st.multiselect(
            "Select one or more models",
            options=model_names,
            default=default_model_selection,
            key="selected_model_names",
            help=(
                "All .pt files inside the app/models folder are listed here. "
                "When several models are selected, every frame is analyzed "
                "by each model and the detections are combined (ensemble). "
                "Each additional model reduces the live frame rate."
            ),
        )

        if not selected_model_names:
            model_errors.append(
                "Select at least one model to enable detection."
            )

        for selected_name in selected_model_names:
            model_path = MODEL_DIR / selected_name

            try:
                with st.spinner(f"Loading {selected_name}..."):
                    loaded_model = load_model(str(model_path))

                models.append(loaded_model)
                MODEL_PATHS.append(model_path)
                LOADED_MODEL_NAMES.append(selected_name)
            except Exception as error:
                model_errors.append(
                    f"{selected_name}: {error}"
                )

        model_ready = len(models) > 0
        CLASS_NAMES = get_combined_class_names(models)
    else:
        selected_model_names = []
        st.multiselect(
            "Select one or more models",
            options=["No .pt models found"],
            disabled=True,
            key="selected_model_names_empty",
        )
        model_errors.append(
            f"No .pt model files were found in: {MODEL_DIR}"
        )

    # A short, stable suffix derived from the selected models. It keys the
    # class-dependent widgets so their defaults refresh when the model
    # combination changes.
    model_widget_suffix = hashlib.md5(
        "|".join(sorted(LOADED_MODEL_NAMES)).encode("utf-8")
    ).hexdigest()[:10]

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
        ("person", "worker", "human"),
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

    st.divider()
    st.markdown("### Detection Display")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=1.00,
        value=0.01,
        step=0.01,
        key="confidence_threshold",
    )

    image_size = st.selectbox(
        "Input image size",
        options=[320, 480, 640, 800,1280,1920],
        index=2,
        key="input_image_size",
    )

    show_labels = st.toggle(
        "Show class labels",
        value=True,
        key="show_class_labels",
    )

    show_confidence = st.toggle(
        "Show confidence scores",
        value=False,
        key="show_confidence_scores",
    )

    st.divider()
    st.markdown("### Camera Stream")

    camera_resolution = st.selectbox(
        "Requested camera resolution",
        options=[
            "640x480",
            "1280x720",
            "1920x1080",
            "2560x1440",
            "3840x2160",
        ],
        index=2,
        key="camera_resolution",
        help=(
            "The browser requests this resolution from the webcam. The "
            "camera delivers the closest supported resolution. Higher "
            "resolutions do not improve detection accuracy because frames "
            "are resized to the input image size before inference."
        ),
    )

    camera_fps = st.selectbox(
        "Requested camera frame rate",
        options=[15, 30, 60],
        index=1,
        key="camera_fps",
        help=(
            "Requested as an ideal value. If the webcam cannot deliver it, "
            "the browser falls back to the highest supported frame rate. "
            "Restart the live video for changes to take effect."
        ),
    )

    st.divider()
    st.markdown("### Live Banner")

    show_banner = st.toggle(
        "Show status banner",
        value=True,
        key="show_banner",
        help="Overlay a status bar on the live video feed.",
    )

    show_banner_timer = st.toggle(
        "Show violation timer",
        value=True,
        key="show_banner_timer",
        help="Display how long the current violation has been active.",
    )

    show_banner_reasons = st.toggle(
        "Show violation reasons",
        value=True,
        key="show_banner_reasons",
        help="Display the reason text below the main banner label.",
    )

    st.divider()
    st.markdown("### Violation Logic")

    strategy = st.radio(
        "Violation detection strategy",
        options=[
            "Explicit violation classes",
            "Scene-level missing PPE",
        ],
        key="violation_strategy",
        help=(
            "Explicit violation classes are recommended when a model "
            "contains classes such as no_helmet or no_vest."
        ),
    )

    violation_classes = set(
        st.multiselect(
            "Violation classes",
            options=CLASS_NAMES,
            default=default_violation_classes,
            disabled=(
                strategy != "Explicit violation classes"
                or not model_ready
            ),
            key=f"violation_classes_{model_widget_suffix}",
        )
    )

    person_options = CLASS_NAMES or [""]
    person_default_index = (
        CLASS_NAMES.index(default_person_classes[0])
        if default_person_classes
        else 0
    )

    person_class = st.selectbox(
        "Person or worker class",
        options=person_options,
        index=person_default_index,
        disabled=(
            strategy != "Scene-level missing PPE"
            or not model_ready
        ),
        key=f"person_class_{model_widget_suffix}",
    )

    required_ppe_classes = set(
        st.multiselect(
            "Required PPE classes",
            options=CLASS_NAMES,
            default=default_required_ppe,
            disabled=(
                strategy != "Scene-level missing PPE"
                or not model_ready
            ),
            key=f"required_ppe_{model_widget_suffix}",
        )
    )

    grace_period = st.slider(
        "Grace period before flagging",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="grace_period",
        help="A snapshot is taken only after the violation lasts this long.",
    )

    absence_tolerance = st.slider(
        "Detection gap tolerance",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.1,
        key="absence_tolerance",
        help=(
            "Prevents the timer from resetting because of brief "
            "missed detections."
        ),
    )

    snapshot_cooldown = st.slider(
        "Snapshot cooldown",
        min_value=1.0,
        max_value=60.0,
        value=10.0,
        step=1.0,
        key="snapshot_cooldown",
        help="Minimum number of seconds between violation snapshots.",
    )

    save_snapshots = st.toggle(
        "Save snapshots to disk",
        value=True,
        key="save_snapshots_to_disk",
    )

    st.divider()
    st.markdown("### Model Status")

    if model_ready:
        st.success(
            f"{len(models)} model(s) loaded successfully"
        )

        for loaded_name in LOADED_MODEL_NAMES:
            st.caption(f"Model: {loaded_name}")

        st.caption(f"Combined classes: {len(CLASS_NAMES)}")
        st.caption(
            "Inference device: "
            + ("GPU (FP16)" if USE_HALF else "CPU (FP32)")
        )

        if len(models) > 1:
            st.info(
                "Ensemble mode: every frame runs through each selected "
                "model. Overlapping detections from different models are "
                "not deduplicated, so detection counts may include "
                "duplicates."
            )

        with st.expander("View combined model classes"):
            for class_index, class_name in enumerate(CLASS_NAMES):
                st.write(f"{class_index}: {class_name}")
    else:
        st.error("No model could be loaded")

    if model_errors:
        for error_message in model_errors:
            st.code(error_message)

    if st.button(
        "Reset violation history",
        use_container_width=True,
        key="reset_violation_history",
    ):
        violation_state.reset()
        st.success("Violation history has been reset.")


# Parse the requested camera resolution into width and height values.
camera_width, camera_height = (
    int(value) for value in camera_resolution.split("x")
)

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
    show_banner=show_banner,
    show_banner_timer=show_banner_timer,
    show_banner_reasons=show_banner_reasons,
)

CURRENT_SETTINGS_SIGNATURE = (
    tuple(sorted(str(path) for path in MODEL_PATHS)),
    current_config.confidence,
    current_config.image_size,
    current_config.strategy,
    tuple(sorted(current_config.violation_classes)),
    current_config.person_class,
    tuple(sorted(current_config.required_ppe_classes)),
    current_config.grace_period,
    current_config.absence_tolerance,
    current_config.snapshot_cooldown,
    current_config.save_snapshots,
    current_config.show_labels,
    current_config.show_confidence,
    current_config.show_banner,
    current_config.show_banner_timer,
    current_config.show_banner_reasons,
)

previous_model_paths = st.session_state.get("active_model_paths")
current_model_paths = tuple(sorted(str(path) for path in MODEL_PATHS))

if (
    previous_model_paths is not None
    and previous_model_paths != current_model_paths
):
    violation_state.reset()
    clear_processed_results()

st.session_state.active_model_paths = current_model_paths

if not save_snapshots:
    violation_state.clear_saved_path_reference()

# This update is what makes the running WebRTC processor react to sidebar
# changes without needing to stop and recreate the video processor.
live_runtime.update(
    models=models,
    config=current_config,
    model_paths=MODEL_PATHS,
)


# =========================================================
# 10. Create detection helper functions
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


def run_multi_model_detection_bgr(
    image_bgr: np.ndarray,
    active_models: list[YOLO],
    config: ViolationConfig,
) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Run every selected model on one BGR frame and combine the results.

    Each model predicts on the original frame, and its detections are drawn
    on top of the previous model's annotations, so the output frame shows
    the boxes from all models at once.
    """
    annotated_bgr = image_bgr.copy()
    combined_classes: list[str] = []
    combined_scores: list[float] = []

    for active_model in active_models:
        results = active_model.predict(
            source=image_bgr,
            conf=config.confidence,
            imgsz=config.image_size,
            half=USE_HALF,
            verbose=False,
        )

        result = results[0]

        annotated_bgr = result.plot(
            img=annotated_bgr,
            labels=config.show_labels,
            conf=config.show_confidence,
        )

        detected_classes, confidence_scores = extract_detection_data(result)
        combined_classes.extend(detected_classes)
        combined_scores.extend(confidence_scores)

    return annotated_bgr, combined_classes, combined_scores


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
    show_banner: bool = True,
    show_timer: bool = True,
    show_reasons: bool = True,
) -> np.ndarray:
    """
    Add a compact live violation banner and timer to the video frame.
    """
    if not show_banner:
        return frame

    output = frame.copy()
    frame_height, frame_width = output.shape[:2]

    if active:
        confirmed = duration >= grace_period
        if show_timer:
            banner_text = (
                f"PPE VIOLATION  {duration:.1f}s"
                if confirmed
                else f"Checking...  {duration:.1f}s"
            )
        else:
            banner_text = "PPE VIOLATION" if confirmed else "Checking violation"
        banner_color = (0, 0, 200) if confirmed else (0, 140, 255)

        show_reason_row = show_reasons and bool(reasons)
        banner_h = 44 if show_reason_row else 28

        cv2.rectangle(
            output, (0, 0), (frame_width, banner_h), banner_color, thickness=-1,
        )

        cv2.putText(
            output, banner_text, (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        if show_reason_row:
            reason_text = " | ".join(reasons[:2])
            cv2.putText(
                output, reason_text[:100], (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA,
            )
    else:
        cv2.rectangle(
            output, (0, 0), (frame_width, 26), (30, 130, 60), thickness=-1,
        )
        cv2.putText(
            output, "PPE: compliant", (12, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return output


def run_image_detection(
    image: Image.Image,
    config: ViolationConfig,
) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Run every selected YOLO model on one uploaded or captured image.

    Returns an RGB annotated image together with the combined detected
    class names and confidence scores from all selected models.
    """
    if not models:
        raise RuntimeError("No YOLO model is available.")

    # PIL images are RGB while OpenCV and YOLO plotting use BGR.
    image_rgb = np.array(image)
    image_bgr = image_rgb[:, :, ::-1].copy()

    (
        annotated_bgr,
        detected_classes,
        confidence_scores,
    ) = run_multi_model_detection_bgr(
        image_bgr=image_bgr,
        active_models=models,
        config=config,
    )

    annotated_rgb = annotated_bgr[:, :, ::-1]

    return annotated_rgb, detected_classes, confidence_scores


def get_reactive_image_result(
    cache_prefix: str,
    image: Image.Image,
    source_bytes: bytes,
    detect_requested: bool,
    config: ViolationConfig,
) -> tuple[np.ndarray, list[str], list[float]] | None:
    """
    Cache one image result and refresh it when sidebar settings change.

    A new source image still requires the user to select its Detect button.
    After the first detection, changing the selected models or any sidebar
    option automatically recalculates the result using the new settings.
    """
    source_id = hashlib.sha256(source_bytes).hexdigest()
    source_key = f"{cache_prefix}_source_id"
    requested_key = f"{cache_prefix}_detection_requested"
    result_key = f"{cache_prefix}_result"
    signature_key = f"{cache_prefix}_settings_signature"

    if st.session_state.get(source_key) != source_id:
        st.session_state[source_key] = source_id
        st.session_state[requested_key] = False
        st.session_state.pop(result_key, None)
        st.session_state.pop(signature_key, None)

    if detect_requested:
        st.session_state[requested_key] = True

    if not st.session_state.get(requested_key, False):
        return None

    result_is_stale = (
        result_key not in st.session_state
        or st.session_state.get(signature_key)
        != CURRENT_SETTINGS_SIGNATURE
    )

    if result_is_stale:
        st.session_state[result_key] = run_image_detection(
            image,
            config,
        )
        st.session_state[signature_key] = CURRENT_SETTINGS_SIGNATURE

    return st.session_state[result_key]


# =========================================================
# 11. Create the live video processor
# =========================================================

class PPEViolationProcessor(VideoProcessorBase):
    """
    Detect PPE and monitor violations in every live video frame.

    The selected models and settings are retrieved from LiveRuntime for
    every frame, so sidebar changes (including adding or removing models)
    take effect while the stream is running.
    """

    def __init__(
        self,
        state: ViolationState,
        runtime: LiveRuntime,
    ) -> None:
        self.state = state
        self.runtime = runtime

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process one live video frame using the latest sidebar settings.
        """
        image = frame.to_ndarray(format="bgr24")
        active_models, active_config = self.runtime.get()

        if not active_models or active_config is None:
            unavailable_frame = image.copy()
            cv2.putText(
                unavailable_frame,
                "Detection model is not available",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return av.VideoFrame.from_ndarray(
                unavailable_frame,
                format="bgr24",
            )

        (
            annotated_frame,
            detected_classes,
            _,
        ) = run_multi_model_detection_bgr(
            image_bgr=image,
            active_models=active_models,
            config=active_config,
        )

        violation_detected, reasons = evaluate_violation(
            detected_classes,
            active_config,
        )

        active, duration = self.state.update(
            violation_detected=violation_detected,
            reasons=reasons,
            annotated_frame=annotated_frame,
            config=active_config,
        )

        display_frame = add_violation_banner(
            frame=annotated_frame,
            active=active,
            duration=duration,
            reasons=reasons,
            grace_period=active_config.grace_period,
            show_banner=active_config.show_banner,
            show_timer=active_config.show_banner_timer,
            show_reasons=active_config.show_banner_reasons,
        )

        return av.VideoFrame.from_ndarray(
            display_frame,
            format="bgr24",
        )


# =========================================================
# 12. Create result display functions
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

        if current_config.save_snapshots and state_data["last_snapshot_path"]:
            st.code(state_data["last_snapshot_path"])
        elif not current_config.save_snapshots:
            st.caption("Disk saving is disabled; this snapshot is kept in memory only.")


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

        # Use the source video's real frame rate so the processed video
        # plays back at the original speed.
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

            # Run every selected model directly on the BGR frame
            (
                annotated_bgr,
                detected_classes,
                confidence_scores,
            ) = run_multi_model_detection_bgr(
                image_bgr=frame,
                active_models=models,
                config=config,
            )

            # Ensure that the processed frame has three channels
            if annotated_bgr.ndim == 2:
                annotated_bgr = cv2.cvtColor(
                    annotated_bgr,
                    cv2.COLOR_GRAY2BGR,
                )

            if annotated_bgr.shape[2] == 4:
                annotated_bgr = cv2.cvtColor(
                    annotated_bgr,
                    cv2.COLOR_BGRA2BGR,
                )

            # Ensure that every frame has the original dimensions
            if (
                annotated_bgr.shape[1] != frame_width
                or annotated_bgr.shape[0] != frame_height
            ):
                annotated_bgr = cv2.resize(
                    annotated_bgr,
                    (frame_width, frame_height),
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
# 13. Create the page header and tabs
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
# 14. Create the live video tab
# =========================================================

with live_tab:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">Live PPE Violation Monitoring</div>
            <div class="card-description">
                Start the webcam to detect PPE continuously. The camera remains
                off until you explicitly start live monitoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_start_column, live_stop_column = st.columns(2)

    with live_start_column:
        if st.button(
            "▶️ Start Live Video",
            type="primary",
            disabled=st.session_state.live_camera_enabled,
            use_container_width=True,
            key="start_live_camera",
        ):
            st.session_state.live_camera_enabled = True
            st.rerun()

    with live_stop_column:
        if st.button(
            "⏹️ Stop Live Video",
            disabled=not st.session_state.live_camera_enabled,
            use_container_width=True,
            key="stop_live_camera",
        ):
            st.session_state.live_camera_enabled = False
            violation_state.reset()
            st.rerun()

    live_configuration_valid = True

    if not model_ready:
        st.error(
            "No selected model is available. Check the model selection "
            "and model status in the sidebar."
        )
        live_configuration_valid = False
    elif (
        strategy == "Explicit violation classes"
        and not violation_classes
    ):
        st.warning(
            "Select at least one violation class in the sidebar."
        )
        live_configuration_valid = False
    elif (
        strategy == "Scene-level missing PPE"
        and not required_ppe_classes
    ):
        st.warning(
            "Select at least one required PPE class in the sidebar."
        )
        live_configuration_valid = False

    # Always render the WebRTC component so the browser never accumulates
    # stale RTCPeerConnection objects from conditional mount/unmount cycles.
    webrtc_streamer(
        key="ppe-violation-camera",
        video_processor_factory=lambda: PPEViolationProcessor(
            state=violation_state,
            runtime=live_runtime,
        ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": camera_width},
                "height": {"ideal": camera_height},
                "frameRate": {"ideal": camera_fps, "min": 15},
            },
            "audio": False,
        },
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=True,
            muted=True,
            style={
                "width": "100%",
                "height": "100%",
                "objectFit": "contain",
            },
        ),
        async_processing=True,
        desired_playing_state=st.session_state.live_camera_enabled and live_configuration_valid,
    )

    if st.session_state.live_camera_enabled and live_configuration_valid:
        st.success(
            "Live camera is enabled. Sidebar changes now apply to the next "
            "processed frames without restarting the stream."
        )
        st.caption(
            f"Requested stream: {camera_width}x{camera_height} at "
            f"{camera_fps} fps · Active models: "
            f"{', '.join(LOADED_MODEL_NAMES)}. The browser negotiates the "
            "closest resolution and frame rate your webcam supports. "
            "Resolution and frame rate changes apply the next time the "
            "live video is started."
        )
    elif not st.session_state.live_camera_enabled:
        st.info(
            "The live camera is off. Select Start Live Video when you are "
            "ready to begin detection."
        )

    st.markdown("### Live Violation Status")

    if hasattr(st, "fragment"):
        @st.fragment(run_every=1.0)
        def render_live_status() -> None:
            render_live_status_body()

        render_live_status()
    else:
        render_live_status_body()

        if st.button("Refresh live status", key="refresh_live_status"):
            st.rerun()


# =========================================================
# 15. Create the camera tab
# =========================================================

with camera_tab:
    st.markdown(
        """
        <div class="custom-card">
            <div class="card-title">Camera Snapshot Detection</div>
            <div class="card-description">
                Open the camera only when needed, take one picture, and analyze
                its PPE status using the currently selected sidebar settings.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    camera_open_column, camera_close_column = st.columns(2)

    with camera_open_column:
        if st.button(
            "📷 Open Camera",
            type="primary",
            disabled=st.session_state.snapshot_camera_enabled,
            use_container_width=True,
            key="open_snapshot_camera",
        ):
            st.session_state.snapshot_camera_enabled = True
            st.rerun()

    with camera_close_column:
        if st.button(
            "✖️ Close Camera",
            disabled=not st.session_state.snapshot_camera_enabled,
            use_container_width=True,
            key="close_snapshot_camera",
        ):
            st.session_state.snapshot_camera_enabled = False
            st.session_state.pop("ppe_camera_input", None)
            st.rerun()

    if not st.session_state.snapshot_camera_enabled:
        st.info(
            "The snapshot camera is off. Select Open Camera when you want "
            "to take a picture."
        )
    elif not model_ready:
        st.error(
            "Load at least one valid model from the sidebar before taking "
            "a picture."
        )
    else:
        camera_image = st.camera_input(
            "Take a picture",
            label_visibility="collapsed",
            key="ppe_camera_input",
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

                with result_column:
                    with st.spinner(
                        "Running PPE detection..."
                    ):
                        camera_result = get_reactive_image_result(
                            cache_prefix="camera_image",
                            image=camera_pil_image,
                            source_bytes=camera_image.getvalue(),
                            detect_requested=camera_button,
                            config=current_config,
                        )

                    if camera_result is not None:
                        (
                            annotated_image,
                            detected_classes,
                            confidence_scores,
                        ) = camera_result

                        display_image_results(
                            annotated_image,
                            detected_classes,
                            confidence_scores,
                            current_config,
                        )

            except Exception as error:
                st.error(f"Unable to process the camera image: {error}")


# =========================================================
# 16. Create the upload image tab
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

            with result_column:
                with st.spinner(
                    "Running PPE detection..."
                ):
                    upload_result = get_reactive_image_result(
                        cache_prefix="uploaded_image",
                        image=uploaded_pil_image,
                        source_bytes=uploaded_file.getvalue(),
                        detect_requested=upload_button,
                        config=current_config,
                    )

                if upload_result is not None:
                    (
                        annotated_image,
                        detected_classes,
                        confidence_scores,
                    ) = upload_result

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
            "or long videos, and each additional selected model increases "
            "the processing time. Audio is not retained in the processed "
            "video."
        )

        # ---------------------------------------------
        # Validate model and violation configuration
        # ---------------------------------------------

        video_configuration_valid = True

        if not model_ready:
            st.error(
                "No selected model is available. Check the model "
                "selection in the sidebar."
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

                    st.session_state[
                        "processed_video_settings_signature"
                    ] = CURRENT_SETTINGS_SIGNATURE

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
            and st.session_state.get(
                "processed_video_settings_signature"
            )
            == CURRENT_SETTINGS_SIGNATURE
        )

        previous_video_result_is_stale = (
            "processed_video_bytes" in st.session_state
            and st.session_state.get(
                "processed_video_source_name"
            )
            == uploaded_video.name
            and st.session_state.get(
                "processed_video_settings_signature"
            )
            != CURRENT_SETTINGS_SIGNATURE
        )

        if previous_video_result_is_stale:
            st.info(
                "The models or sidebar settings changed. Select Process "
                "Uploaded Video again to generate an updated result."
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
# 18. Create the About tab
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

        This is the recommended strategy when a model contains classes such
        as `no_helmet`, `no_vest`, or `without_gloves`.

        ### Scene-Level Missing PPE

        This strategy checks whether a person or worker is detected and whether
        the required PPE classes appear anywhere in the same frame.

        This is only a scene-level approximation. It does not reliably match a
        specific helmet or vest to a specific worker when multiple people are
        present.

        ### Multi-Model Ensemble

        When several models are selected in the sidebar, every frame is
        analyzed by each model and the detections are combined. The
        annotated frame shows the boxes from all models layered together.

        Keep in mind:

        - Inference time grows linearly with the number of selected models,
          so the live frame rate drops with each additional model.
        - Detections are not deduplicated across models. If two models
          detect the same object, it is counted twice and drawn twice.
        - Violation logic uses the combined class names from all models.

        ### Camera Resolution and Frame Rate

        The sidebar Camera Stream section requests a resolution and frame
        rate from the browser. The webcam delivers the closest supported
        values, so requesting 60 fps on a 30 fps webcam results in 30 fps.
        The requested values apply the next time the live video starts.

        ### Snapshot Storage

        Saved violation snapshots are stored in:

        ```text
        app/violation_captures/
        ```

        ### Model Selection

        Every `.pt` model placed in `app/models/` appears in the sidebar
        model multiselect. The class-dependent violation controls refresh
        for the selected model combination.

        ### Camera Privacy

        Both live video and snapshot camera components remain disabled until
        you explicitly select their start/open buttons.
        """
    )


# =========================================================
# 19. Add the page footer
# =========================================================

st.divider()

st.caption(
    "PPE Detection System · Powered by Streamlit, WebRTC, and YOLO"
)