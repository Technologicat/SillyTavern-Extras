"""THA3 live mode for SillyTavern-extras.

This is the animation engine, running on top of the THA3 posing engine.
This module implements the live animation backend and serves the API. For usage, see `server.py`.

If you want to play around with THA3 expressions in a standalone app, see `manual_poser.py`.
"""

# TODO: talkinghead live mode:
#  - talking animation is broken, seems the client isn't sending us a request to start/stop talking?
#  - improve idle animations
#    - cosine schedule?
#    - or perhaps the current ODE approach is better (define instant rate only, based on target state; then integrate)
#  - PNG sending efficiency?

import atexit
import io
import logging
import os
import random
import sys
import time
import numpy as np
import threading
from typing import Dict, List, NoReturn, Optional, Union

import PIL

import torch

from flask import Flask, Response
from flask_cors import CORS

from tha3.poser.modes.load_poser import load_poser
from tha3.poser.poser import Poser
from tha3.util import (torch_linear_to_srgb, resize_PIL_image,
                       extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image)
from tha3.app.util import posedict_keys, posedict_key_to_index, load_emotion_presets, posedict_to_pose, to_talkinghead_image, FpsStatistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
# TODO: we could move many of these into TalkingheadAnimator, and just keep a reference to that as global.
global_basedir = "talkinghead"
global_animator_instance = None
_animator_output_lock = threading.Lock()
global_reload_image = None

animation_running = False
is_talking = False
current_emotion = "neutral"

# Flask setup
app = Flask(__name__)
CORS(app)

# --------------------------------------------------------------------------------
# API

def setEmotion(_emotion: Dict[str, float]) -> None:
    """Set the current emotion of the character based on sentiment analysis results.

    Currently, we pick the emotion with the highest confidence score.

    _emotion: result of sentiment analysis: {emotion0: confidence0, ...}
    """
    global current_emotion

    highest_score = float("-inf")
    highest_label = None

    for item in _emotion:
        if item["score"] > highest_score:
            highest_score = item["score"]
            highest_label = item["label"]

    logger.debug(f"setEmotion: applying emotion {highest_label}")
    current_emotion = highest_label

def unload() -> str:
    """Stop animation."""
    global animation_running
    animation_running = False
    logger.debug("unload: animation paused")
    return "Animation Paused"

def start_talking() -> str:
    """Start talking animation."""
    global is_talking
    is_talking = True
    logger.debug("start_talking called")
    return "started"

def stop_talking() -> str:
    """Stop talking animation."""
    global is_talking
    is_talking = False
    logger.debug("stop_talking called")
    return "stopped"

def result_feed() -> Response:
    """Return a Flask `Response` that repeatedly yields the current image as 'image/png'."""
    def generate():
        last_update_time = None
        last_report_time = None
        fps_statistics = FpsStatistics()
        image_bytes = None

        while True:
            # Retrieve a new frame from the animator if available.
            have_new_frame = False
            with _animator_output_lock:
                if global_animator_instance.frame_ready:
                    image_rgba = global_animator_instance.result_image
                    try:
                        pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
                        if image_rgba.shape[2] == 4:
                            alpha_channel = image_rgba[:, :, 3]
                            pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))
                        global_animator_instance.frame_ready = False  # Animation frame consumed; tell the animator it can begin rendering the next one.
                        have_new_frame = True
                    except Exception as exc:
                        logger.error(exc)

            # Pack the new animation frame for sending.
            if have_new_frame:
                try:
                    buffer = io.BytesIO()  # Save as PNG with RGBA mode
                    pil_image.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                except Exception as exc:
                    logger.error(f"Cannot write image to buffer: {exc}")
                    raise

            # Send the animation frame.
            if image_bytes is not None:
                # How often should we send?
                #  - Excessive spamming can DoS the SillyTavern GUI, so there needs to be a rate limit.
                #  - OTOH, we must constantly send something, or the GUI will lock up waiting.
                #
                # Thus, if we have a new frame, send it now. Otherwise wait for a bit.
                if have_new_frame:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/png\r\n\r\n" + image_bytes + b"\r\n")

                    # Update the FPS counter, measuring the time between network sends.
                    time_now = time.time_ns()
                    if last_update_time is not None:
                        elapsed_time = time_now - last_update_time
                        fps = 1.0 / (elapsed_time / 10**9)
                        fps_statistics.add_fps(fps)
                    last_update_time = time_now
                else:
                    # Target an acceptable anime frame rate of 25 FPS.
                    # Note the animator runs in a different thread, so it can render while we are waiting.
                    # We don't measure pack/send time, so this is not exact. In practice we get ~24 FPS.
                    time.sleep(0.04)

                # Log the FPS counter in 5-second intervals.
                if last_report_time is None or time_now - last_report_time > 5e9:
                    trimmed_fps = round(fps_statistics.get_average_fps(), 1)
                    logger.info("rate-limited network FPS: {:.1f}".format(trimmed_fps))
                    last_report_time = time_now

            else:  # first frame not yet available, animator still booting
                time.sleep(0.1)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# TODO: the input is a flask.request.file.stream; what's the type of that?
def talkinghead_load_file(stream) -> str:
    """Load image from stream and start animation."""
    global global_reload_image
    global animation_running
    logger.debug("talkinghead_load_file: loading new input image from stream")

    try:
        animation_running = False  # pause animation while loading a new image
        pil_image = PIL.Image.open(stream)  # Load the image using PIL.Image.open
        img_data = io.BytesIO()  # Create a copy of the image data in memory using BytesIO
        pil_image.save(img_data, format="PNG")
        global_reload_image = PIL.Image.open(io.BytesIO(img_data.getvalue()))  # Set the global_reload_image to a copy of the image data
    except PIL.Image.UnidentifiedImageError:
        logger.warning("Could not load input image from stream, loading blank")
        full_path = os.path.join(os.getcwd(), os.path.normpath(os.path.join(global_basedir, "tha3", "images", "inital.png")))
        global_animator_instance.load_image(full_path)
    finally:
        animation_running = True
    return "OK"

def launch(device: str, model: str) -> Union[None, NoReturn]:
    """Launch the talking head plugin (live mode).

    If the plugin fails to load, the process exits.

    device: "cpu" or "cuda"
    model: one of the folder names inside "talkinghead/tha3/models/"
    """
    global global_animator_instance
    global initAMI  # TODO: initAREYOU? See if we still need this - the idea seems to be to stop animation until the first image is loaded.
    initAMI = True

    try:
        poser = load_poser(model, device, modelsdir=os.path.join(global_basedir, "tha3", "models"))
        global_animator_instance = TalkingheadAnimator(poser, device)

        # Load initial blank character image
        full_path = os.path.join(os.getcwd(), os.path.normpath(os.path.join(global_basedir, "tha3", "images", "inital.png")))
        global_animator_instance.load_image(full_path)

        global_animator_instance.start()

    except RuntimeError as exc:
        logger.error(exc)
        sys.exit()

# --------------------------------------------------------------------------------
# Internal stuff

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """RGBA (linear) -> RGBA (SRGB), preserving the alpha channel."""
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

class TalkingheadAnimator:
    """uWu Waifu"""

    def __init__(self, poser: Poser, device: torch.device):
        self.poser = poser
        self.device = device

        self.current_pose = None
        self.last_blink_timestamp = 0  # TODO: Great idea! We should actually use this.
        self.is_blinked = False  # TODO: Maybe we might need this, too, now that the FPS is acceptable enough that we may need to blink over several frames.
        self.targets = {"head_y_index": 0}
        self.progress = {"head_y_index": 0}
        self.direction = {"head_y_index": 1}
        self.originals = {"head_y_index": 0}  # TODO: what was this for; probably for recording the values from the current emotion, before sway animation?
        self.forward = {"head_y_index": True}  # Direction of interpolation
        self.start_values = {"head_y_index": 0}

        self.fps_statistics = FpsStatistics()

        self.source_image: Optional[torch.tensor] = None
        self.result_image: Optional[np.array] = None
        self.frame_ready = False
        self.last_report_time = None

        self.emotions, self.emotion_names = load_emotion_presets(os.path.join("talkinghead", "emotions"))

    def start(self) -> None:
        """Start the animation thread."""
        self._terminated = False
        def animation_update():
            while not self._terminated:
                self.update_result_image_bitmap()
                time.sleep(0.01)  # rate-limit the renderer to 100 FPS maximum (this could be adjusted later)
        self.animation_thread = threading.Thread(target=animation_update, daemon=True)
        self.animation_thread.start()
        atexit.register(self.exit)

    def exit(self) -> None:
        """Terminate the animation thread.

        Called automatically when the process exits.
        """
        self._terminated = True
        self.animation_thread.join()

    def apply_emotion_to_pose(self, emotion_posedict: Dict[str, float], pose: List[float]) -> List[float]:
        """Copy all morphs except breathing from `emotion_posedict` to `pose`.

        If a morph does not exist in `emotion_posedict`, its value is copied from `pose`.

        Return the modified pose.
        """
        new_pose = list(pose)  # copy
        for idx, key in enumerate(posedict_keys):
            if key in emotion_posedict and key != "breathing_index":
                new_pose[idx] = emotion_posedict[key]
        return new_pose

    def animate_blinking(self, pose: List[float]) -> List[float]:
        # TODO: add smoothly animated blink?

        # If there should be a blink, set the wink morphs to 1; otherwise, use the provided value.
        should_blink = (random.random() <= 0.03)
        if not should_blink:
            return pose

        new_pose = list(pose)  # copy
        for morph_name in ["eye_wink_left_index", "eye_wink_right_index"]:
            idx = posedict_key_to_index[morph_name]
            new_pose[idx] = 1.0
        return new_pose

    def animate_talking(self, pose: List[float]) -> List[float]:
        if not is_talking:
            return pose

        new_pose = list(pose)  # copy
        idx = posedict_key_to_index["mouth_aaa_index"]
        x = pose[idx]
        x = abs(1.0 - x) + random.uniform(-2.0, 2.0)
        x = max(0.0, min(x, 1.0))  # clamp (not the manga studio)
        new_pose[idx] = x
        return new_pose

    def animate_sway(self, pose: List[float]) -> List[float]:
        # TODO: add sway for other axes and body

        new_pose = list(pose)  # copy
        MOVEPARTS = ["head_y_index"]
        for key in MOVEPARTS:
            idx = posedict_key_to_index[key]
            current_value = pose[idx]

            # Linearly interpolate between start and target values
            new_value = self.start_values[key] + self.progress[key] * (self.targets[key] - self.start_values[key])
            new_value = min(max(new_value, -1), 1)  # clip to bounds (just in case)

            # Check if we've reached the target or start value
            is_close_to_target = abs(new_value - self.targets[key]) < 0.04
            is_close_to_start = abs(new_value - self.start_values[key]) < 0.04

            if (self.direction[key] == 1 and is_close_to_target) or (self.direction[key] == -1 and is_close_to_start):
                # Reverse direction
                self.direction[key] *= -1

                # If direction is now forward, set a new target and store starting value
                if self.direction[key] == 1:
                    self.start_values[key] = new_value
                    self.targets[key] = current_value + random.uniform(-0.6, 0.6)
                    self.progress[key] = 0  # Reset progress when setting a new target

            # Update progress based on direction
            self.progress[key] += 0.04 * self.direction[key]

            new_pose[idx] = new_value
        return new_pose

    def interpolate_pose(self, pose: List[float], target_pose: List[float], step=0.1) -> List[float]:
        # TODO: ignore sway?
        # TODO: ignore breathing?
        new_pose = list(pose)  # copy
        for idx, key in enumerate(posedict_keys):
            # # We animate blinking *after* interpolating the pose, so when blinking, the eyes close instantly.
            # # This part makes the blink also end instantly.
            # if key in ["eye_wink_left_index", "eye_wink_right_index"]:
            #     new_pose[idx] = new_pose[idx]

            # Note this leads to an exponentially saturating behavior (1 - exp(-x)), because the delta is from the current pose to the final pose.
            delta = target_pose[idx] - pose[idx]
            new_pose[idx] = pose[idx] + step * delta
        return new_pose

    def update_result_image_bitmap(self) -> None:
        """Render an animation frame.

        If the previous rendered frame has not been retrieved yet, do nothing.
        """

        global animation_running
        global initAMI

        if not animation_running:
            return

        # Skip rendering, if no one has retrieved the previous frame yet.
        if self.frame_ready:
            return

        if global_reload_image is not None:
            self.load_image()
            return  # TODO: do we really need to return here, we could just proceed?
        if self.source_image is None:
            return

        time_render_start = time.time_ns()

        if self.current_pose is None:  # initialize character pose at plugin startup
            self.current_pose = posedict_to_pose(self.emotions[current_emotion])

        emotion_posedict = self.emotions[current_emotion]
        target_pose = self.apply_emotion_to_pose(emotion_posedict, self.current_pose)

        self.current_pose = self.interpolate_pose(self.current_pose, target_pose)
        self.current_pose = self.animate_blinking(self.current_pose)
        self.current_pose = self.animate_sway(self.current_pose)
        self.current_pose = self.animate_talking(self.current_pose)
        # TODO: animate breathing

        pose = torch.tensor(self.current_pose, device=self.device, dtype=self.poser.get_dtype())

        with torch.no_grad():
            output_image = self.poser.pose(self.source_image, pose)[0].float()  # [0]: model's output index for the full result image
            output_image = convert_linear_to_srgb((output_image + 1.0) / 2.0)

            c, h, w = output_image.shape
            output_image = (255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1)).reshape(h, w, c).byte()
            output_image_numpy = output_image.detach().cpu().numpy()

        # Update FPS counter, measuring animation frame render time only.
        #
        # This says how fast the renderer *can* run on the current hardware;
        # note we don't actually render more frames than the client consumes.
        time_now = time.time_ns()
        if self.source_image is not None:
            elapsed_time = time_now - time_render_start
            fps = 1.0 / (elapsed_time / 10**9)
            self.fps_statistics.add_fps(fps)

        # Set the new rendered frame as the output image, and mark the frame as ready for the network thread.
        with _animator_output_lock:
            self.result_image = output_image_numpy
            self.frame_ready = True

        if initAMI:  # If the models are just now initalized stop animation to save
            animation_running = False
            initAMI = False

        # Log the FPS counter in 5-second intervals.
        if self.last_report_time is None or time_now - self.last_report_time > 5e9:
            trimmed_fps = round(self.fps_statistics.get_average_fps(), 1)
            logger.info("available render FPS: {:.1f}".format(trimmed_fps))
            self.last_report_time = time_now

    def load_image(self, file_path=None) -> None:
        """Load the image file at `file_path`.

        Except, if `global_reload_image is not None`, use the global reload image data instead.
        """
        global global_reload_image

        if global_reload_image is not None:
            file_path = "global_reload_image"

        try:
            if file_path == "global_reload_image":
                pil_image = global_reload_image
            else:
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(file_path),
                    (self.poser.get_image_size(), self.poser.get_image_size()))

            w, h = pil_image.size

            if pil_image.size != (512, 512):
                logger.info("Resizing Char Card to work")
                pil_image = to_talkinghead_image(pil_image)

            w, h = pil_image.size

            if pil_image.mode != "RGBA":
                logger.error("load_image: image must have alpha channel")
                self.source_image = None
            else:
                self.source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                    .to(self.device).to(self.poser.get_dtype())

        except Exception as exc:
            logger.error(f"load_image: {exc}")

        finally:
            global_reload_image = None
