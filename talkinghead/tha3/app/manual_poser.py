"""THA3 manual poser.

Pose an anime character manually, based on a suitable 512×512 static input image and some neural networks.


**What**:

This app is an alternative to the live plugin mode of `talkinghead`. Given one static input image,
this allows the automatic generation of the 28 emotional expression sprites for your AI character,
for use with distilbert classification in SillyTavern.

There are two motivations:

  - Much faster than inpainting all 28 expressions manually in Stable Diffusion. Enables agile experimentation
    on the look of your character, since you only need to produce one new image to change the look.
  - No CPU or GPU load while running SillyTavern, unlike the live plugin mode, which is cool, but slow.

For best results for generating the static input image in Stable Diffusion, consider the various vtuber checkpoints
available on the internet. These should reduce the amount of work it takes to get SD to render your character in
a pose suitable for use as input.

Results are often not perfect, but serviceable.


**How**:

To run the manual poser, ensure that you have the correct wxPython installed in your "extras" conda venv,
open a terminal in the SillyTavern-extras top-level directory, and do the following:

    cd talkinghead
    conda activate extras
    ./start_manual_poser.sh

Note that installing wxPython needs `libgtk-3-dev` (on Debian based distros),
so `sudo apt install libgtk-3-dev` before trying to `pip install wxPython`.
The install may take a very long time (even half an hour) as it needs to
compile a whole GUI toolkit.


**Who**:

Original code written and neural networks designed and trained by Pramook Khungurn (@pkhungurn):
    https://github.com/pkhungurn/talking-head-anime-3-demo
    https://arxiv.org/abs/2311.17409

This fork maintained by the SillyTavern-extras project.

Manual poser app improved and documented by Juha Jeronen (@Technologicat).
"""

# TODO: manual poser:
#   - Write new README: use case and supported features are different from the original THA3 package.
#   - Refactor stuff needed both by this and the live mode that is served by `app.py`.

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import PIL.Image

import numpy

import torch

import wx

from tha3.poser.modes.load_poser import load_poser
from tha3.poser.poser import Poser, PoseParameterCategory, PoseParameterGroup
from tha3.util import rgba_to_numpy_image, grid_change_to_numpy_image, \
    rgb_to_numpy_image, resize_PIL_image, extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect image file formats supported by the installed Pillow, and format a list for wxPython file open/save dialogs.
# TODO: This is not very useful unless we can filter these to get only formats that support an alpha channel.
#
# https://docs.wxpython.org/wx.FileDialog.html
# https://stackoverflow.com/questions/71112986/retrieve-a-list-of-supported-read-file-extensions-formats
#
# exts = PIL.Image.registered_extensions()
# PIL_supported_input_formats = {ex[1:].lower() for ex, f in exts.items() if f in PIL.Image.OPEN}  # {".png", ".jpg", ...} -> {"png", "jpg", ...}
# PIL_supported_output_formats = {ex[1:].lower() for ex, f in exts.items() if f in PIL.Image.SAVE}
# def format_fileformat_list(supported_formats):
#     return ["All files (*)|*"] + [f"{fmt.upper()} images (*.{fmt})|*.{fmt}" for fmt in sorted(supported_formats)]
# input_index_to_ext = [""] + sorted(PIL_supported_input_formats)  # list index -> file extension
# input_ext_to_index = {ext: idx for idx, ext in enumerate(input_index_to_ext)}  # file extension -> list index
# output_index_to_ext = [""] + sorted(PIL_supported_output_formats)
# output_ext_to_index = {ext: idx for idx, ext in enumerate(output_index_to_ext)}
# input_exts_and_descs_str = "|".join(format_fileformat_list(PIL_supported_input_formats))  # filter-spec accepted by `wx.FileDialog`
# output_exts_and_descs_str = "|".join(format_fileformat_list(PIL_supported_output_formats))

# The keys for a pose in the emotion JSON files.
#
# TODO: "eye_unimpressed" is arity-2, but has only one entry in the JSON. The current implementation smashes both into one,
#       letting the second one (right slider) win. Maybe the two values should be saved separately, but we have to avoid
#       breaking the live mode served by `app.py`.
posedict_keys = ["eyebrow_troubled_left_index", "eyebrow_troubled_right_index",
                 "eyebrow_angry_left_index", "eyebrow_angry_right_index",
                 "eyebrow_lowered_left_index", "eyebrow_lowered_right_index",
                 "eyebrow_raised_left_index", "eyebrow_raised_right_index",
                 "eyebrow_happy_left_index", "eyebrow_happy_right_index",
                 "eyebrow_serious_left_index", "eyebrow_serious_right_index",
                 "eye_wink_left_index", "eye_wink_right_index",
                 "eye_happy_wink_left_index", "eye_happy_wink_right_index",
                 "eye_surprised_left_index", "eye_surprised_right_index",
                 "eye_relaxed_left_index", "eye_relaxed_right_index",
                 "eye_unimpressed", "eye_unimpressed",
                 "eye_raised_lower_eyelid_left_index", "eye_raised_lower_eyelid_right_index",
                 "iris_small_left_index", "iris_small_right_index",
                 "mouth_aaa_index",
                 "mouth_iii_index",
                 "mouth_uuu_index",
                 "mouth_eee_index",
                 "mouth_ooo_index",
                 "mouth_delta",
                 "mouth_lowered_corner_left_index", "mouth_lowered_corner_right_index",
                 "mouth_raised_corner_left_index", "mouth_raised_corner_right_index",
                 "mouth_smirk",
                 "iris_rotation_x_index", "iris_rotation_y_index",
                 "head_x_index", "head_y_index",
                 "neck_z_index",
                 "body_y_index", "body_z_index",
                 "breathing_index"]
assert len(posedict_keys) == 45


def load_emotion_presets() -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Load emotion presets from disk.

    These are JSON files in "talkinghead/emotions".

    Returns the tuple `(emotions, emotion_names)`, where::

        emotions = {emotion0_name: posedict0, ...}
        emotion_names = [emotion0_name, emotion1_name, ...]

    The dict contains the actual pose data. The list is a sorted list of emotion names
    that can be used to map a linear index (e.g. the choice index in a GUI dropdown)
    to the corresponding key of `emotions`.

    The directory "talkinghead/emotions" must also contain a "_defaults.json" file,
    containing factory defaults (as a fallback) for the 28 standard emotions
    (as recognized by distilbert), as well as a hidden "zero" preset that represents
    a neutral pose. (This is separate from the "neutral" emotion, which is allowed
    to be "non-zero".)
    """
    emotion_names = []
    for root, dirs, files in os.walk("emotions", topdown=True):
        for filename in files:
            if filename == "_defaults.json":  # skip the repository containing the default fallbacks
                continue
            if filename.lower().endswith(".json"):
                emotion_names.append(filename[:-5])  # drop the ".json"
    emotion_names.sort()  # the 28 actual emotions

    # TODO: Note that currently, we build the list of emotion names from JSON filenames,
    #       and then check whether each JSON implements the emotion matching its filename.
    #       On second thought, I'm not sure whether that makes much sense. Maybe rethink the design.
    #         - We *do* want custom JSON files to show up in the list, if those are placed in "tha3/emotions". So the list of emotions shouldn't be hardcoded.
    #         - *Having* a fallback repository with factory defaults (and a hidden "zero" preset) is useful.
    #           But we are currently missing a way to reset an emotion to its factory default.
    def load_emotion_with_fallback(emotion_name: str) -> Dict[str, float]:
        try:
            with open(os.path.join("emotions", f"{emotion_name}.json"), "r") as json_file:
                emotions_from_json = json.load(json_file)  # A single json file may contain presets for multiple emotions.
            posedict = emotions_from_json[emotion_name]
        except (FileNotFoundError, KeyError):  # If no separate json exists for the specified emotion, load the default (all 28 emotions have a default).
            with open(os.path.join("emotions", "_defaults.json"), "r") as json_file:
                emotions_from_json = json.load(json_file)
            posedict = emotions_from_json[emotion_name]
        # If still not found, it's an error, so fail-fast: let the app exit with an informative exception message.
        return posedict

    # Dict keeps its keys in insertion order, so define some special states before inserting the actual emotions.
    emotions = {"[custom]": {},  # custom = the user has changed at least one value manually after last loading a preset
                "[reset]": load_emotion_with_fallback("zero")}  # reset = a preset with all sliders in their default positions. Found in "_defaults.json".
    for emotion_name in emotion_names:
        emotions[emotion_name] = load_emotion_with_fallback(emotion_name)

    emotion_names = list(emotions.keys())
    return emotions, emotion_names


class SimpleParamGroupsControlPanel(wx.Panel):
    """A simple control panel for groups of arity-1 continuous parameters (i.e. float value, and no separate left/right controls).

    The panel represents a *category*, such as "body rotation".

    A category may have several *parameter groups*, all of which are active simultaneously. Here "parameter group" is a misnomer,
    since in all use sites for this panel, each group has only one parameter. For example, "body rotation" has the groups ["body_y", "body_z"].
    """

    def __init__(self, parent,
                 pose_param_category: PoseParameterCategory,
                 param_groups: List[PoseParameterGroup]):
        super().__init__(parent, style=wx.SIMPLE_BORDER)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)

        self.param_groups = [group for group in param_groups if group.get_category() == pose_param_category]
        for param_group in self.param_groups:
            assert not param_group.is_discrete()
            assert param_group.get_arity() == 1

        self.sliders = []
        for param_group in self.param_groups:
            title_text = wx.StaticText(self, label=param_group.get_group_name(), style=wx.ALIGN_CENTER)
            title_text.SetFont(title_text.GetFont().Bold())
            self.sizer.Add(title_text, 0, wx.EXPAND)
            # HACK: iris_rotation_*, head_*, body_* have range [-1, 1], but breathing has range [0, 1],
            #       and all of them should default to the *value* 0.
            range = param_group.get_range()
            min_value = int(range[0] * 1000)
            max_value = int(range[1] * 1000)
            slider = wx.Slider(self, minValue=min_value, maxValue=max_value, value=0, style=wx.HORIZONTAL)
            self.sizer.Add(slider, 0, wx.EXPAND)
            self.sliders.append(slider)

        self.sizer.Fit(self)

    def write_to_pose(self, pose: List[float]) -> None:
        """Update `pose` (in-place) by the current value(s) set in this control panel."""
        for param_group, slider in zip(self.param_groups, self.sliders):
            alpha = (slider.GetValue() - slider.GetMin()) / (slider.GetMax() - slider.GetMin())
            param_index = param_group.get_parameter_index()
            param_range = param_group.get_range()
            pose[param_index] = param_range[0] + (param_range[1] - param_range[0]) * alpha

    def read_from_pose(self, pose: List[float]) -> None:
        """Overwrite the current value(s) in this control panel by those taken from `pose`."""
        for param_group, slider in zip(self.param_groups, self.sliders):
            param_range = param_group.get_range()
            param_index = param_group.get_parameter_index()
            value = pose[param_index]  # cherry-pick only relevant values from `pose`
            alpha = (value - param_range[0]) / (param_range[1] - param_range[0])
            slider.SetValue(int(slider.GetMin() + alpha * (slider.GetMax() - slider.GetMin())))


class MorphCategoryControlPanel(wx.Panel):
    """A more complex control panel with grouping semantics.

    The panel represents a *category*, such as "eyebrow".

    A category may have several *parameter groups*, only one of which can be active at any given time.

    For example, the "eyebrow" category has the parameter groups ["eyebrow_troubled", "eyebrow_angry", ...].

    Each parameter group can be:
      - Continuous with arity 1 (one slider),
      - Continuous with arity 2 (two sliders, for separate left/right control), or
      - Discrete (on/off).

    The panel allows the user to select a parameter group within the category, and enables/disables its
    UI controls appropriately. The user can then use the controls to set the values for the selected
    parameter group within the category represented by the panel.
    """
    def __init__(self,
                 parent,
                 category_title: str,
                 pose_param_category: PoseParameterCategory,
                 param_groups: List[PoseParameterGroup]):
        super().__init__(parent, style=wx.SIMPLE_BORDER)
        self.pose_param_category = pose_param_category
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)

        self.title_text = wx.StaticText(self, label=category_title, style=wx.ALIGN_CENTER)
        self.title_text.SetFont(self.title_text.GetFont().Bold())
        self.sizer.Add(self.title_text, 0, wx.EXPAND)

        self.param_groups = [group for group in param_groups if group.get_category() == pose_param_category]
        self.param_group_names = [group.get_group_name() for group in self.param_groups]
        self.choice = wx.Choice(self, choices=self.param_group_names)
        if len(self.param_groups) > 0:
            self.choice.SetSelection(0)
        self.choice.Bind(wx.EVT_CHOICE, self.on_choice_updated)
        self.sizer.Add(self.choice, 0, wx.EXPAND)

        self.left_slider = wx.Slider(self, minValue=-1000, maxValue=1000, value=-1000, style=wx.HORIZONTAL)
        self.sizer.Add(self.left_slider, 0, wx.EXPAND)

        self.right_slider = wx.Slider(self, minValue=-1000, maxValue=1000, value=-1000, style=wx.HORIZONTAL)
        self.sizer.Add(self.right_slider, 0, wx.EXPAND)

        self.checkbox = wx.CheckBox(self, label="Show")
        self.checkbox.SetValue(True)
        self.sizer.Add(self.checkbox, 0, wx.SHAPED | wx.ALIGN_CENTER)

        self.update_ui()

        self.sizer.Fit(self)

    def update_ui(self) -> None:
        """Enable/disable UI controls based on the currently active parameter group."""
        param_group = self.param_groups[self.choice.GetSelection()]
        if param_group.is_discrete():
            self.left_slider.Enable(False)
            self.right_slider.Enable(False)
            self.checkbox.Enable(True)
        elif param_group.get_arity() == 1:
            self.left_slider.Enable(True)
            self.right_slider.Enable(False)
            self.checkbox.Enable(False)
        else:
            self.left_slider.Enable(True)
            self.right_slider.Enable(True)
            self.checkbox.Enable(False)

    def on_choice_updated(self, event: wx.Event) -> None:
        """Automatically optimize usability for the new arity and discrete/continuous state."""
        param_group = self.param_groups[self.choice.GetSelection()]
        if param_group.is_discrete():
            self.checkbox.SetValue(True)  # discrete parameter group: set to "on" when switched into
            self.left_slider.SetValue(self.left_slider.GetMin())
            self.right_slider.SetValue(self.right_slider.GetMin())
        else:
            if param_group.get_arity() == 2:  # make it apparent that both sliders are in use now
                self.right_slider.SetValue(self.left_slider.GetValue())  # ...by copying value left->right
            else:  # arity 1, right slider not in use, so zero it out visually.
                self.right_slider.SetValue(self.right_slider.GetMin())
        self.update_ui()
        event.Skip()  # allow other handlers for the same event to run

    def write_to_pose(self, pose: List[float]) -> None:
        """Update `pose` (in-place) by the current value(s) set in this control panel.

        Only the currently chosen parameter group is applied.
        """
        if len(self.param_groups) == 0:
            return
        selected_morph_index = self.choice.GetSelection()
        param_group = self.param_groups[selected_morph_index]
        param_index = param_group.get_parameter_index()
        if param_group.is_discrete():
            if self.checkbox.GetValue():
                for i in range(param_group.get_arity()):
                    pose[param_index + i] = 1.0
        else:
            param_range = param_group.get_range()
            alpha = (self.left_slider.GetValue() - self.left_slider.GetMin()) * 1.0 / (self.left_slider.GetMax() - self.left_slider.GetMin())  # -> [0, 1]
            pose[param_index] = param_range[0] + (param_range[1] - param_range[0]) * alpha
            if param_group.get_arity() == 2:
                alpha = (self.right_slider.GetValue() - self.right_slider.GetMin()) * 1.0 / (self.right_slider.GetMax() - self.right_slider.GetMin())
                pose[param_index + 1] = param_range[0] + (param_range[1] - param_range[0]) * alpha

    def read_from_pose(self, pose: List[float]) -> None:
        """Overwrite the current value(s) in this control panel by those taken from `pose`.

        All parameter groups in this panel are scanned to find a nonzero value in `pose`.
        The parameter group that first finds a nonzero value wins, selects its morph for this panel,
        and applies the values to the sliders in the panel.

        If nothing matches, the first available morph is selected, and the sliders are set to zero.
        """
        # Find which morph (param group) is active in our category in `pose`.
        for morph_index, param_group in enumerate(self.param_groups):
            param_index = param_group.get_parameter_index()
            value = pose[param_index]
            if value != 0.0:
                break
            # An arity-2 param group is active also when just the right slider is nonzero.
            if param_group.get_arity() == 2:
                value = pose[param_index + 1]
                if value != 0.0:
                    break
        else:  # No param group in this panel's category had a nonzero value in `pose`.
            if len(self.param_groups) > 0:
                logger.debug(f"category {self.title_text.GetLabel()}: no nonzero values, chose default morph {self.param_group_names[0]}")
                self.choice.SetSelection(0)  # choose the first param group
                self.left_slider.SetValue(self.left_slider.GetMin())
                self.right_slider.SetValue(self.right_slider.GetMin())
                self.checkbox.SetValue(False)
                self.update_ui()
                return
        logger.debug(f"category {self.title_text.GetLabel()}: found nonzero values, chose morph {self.param_group_names[morph_index]}")
        self.choice.SetSelection(morph_index)
        if param_group.is_discrete():
            self.left_slider.SetValue(self.left_slider.GetMin())
            self.right_slider.SetValue(self.right_slider.GetMin())
            if pose[param_index]:
                self.checkbox.SetValue(True)
            else:
                self.checkbox.SetValue(False)
        else:
            self.checkbox.SetValue(False)
            param_range = param_group.get_range()
            value = pose[param_index]
            alpha = (value - param_range[0]) / (param_range[1] - param_range[0])
            self.left_slider.SetValue(int(self.left_slider.GetMin() + alpha * (self.left_slider.GetMax() - self.left_slider.GetMin())))
            if param_group.get_arity() == 2:
                value = pose[param_index + 1]
                alpha = (value - param_range[0]) / (param_range[1] - param_range[0])
                self.right_slider.SetValue(int(self.right_slider.GetMin() + alpha * (self.right_slider.GetMax() - self.right_slider.GetMin())))
            else:  # arity 1, right slider not in use, so zero it out visually.
                self.right_slider.SetValue(self.right_slider.GetMin())
        self.update_ui()


def convert_output_image_from_torch_to_numpy(output_image):
    if output_image.shape[2] == 2:
        h, w, c = output_image.shape
        numpy_image = torch.transpose(output_image.reshape(h * w, c), 0, 1).reshape(c, h, w)
    elif output_image.shape[0] == 4:
        numpy_image = rgba_to_numpy_image(output_image)
    elif output_image.shape[0] == 3:
        numpy_image = rgb_to_numpy_image(output_image)
    elif output_image.shape[0] == 1:
        c, h, w = output_image.shape
        alpha_image = torch.cat([output_image.repeat(3, 1, 1) * 2.0 - 1.0, torch.ones(1, h, w)], dim=0)
        numpy_image = rgba_to_numpy_image(alpha_image)
    elif output_image.shape[0] == 2:
        numpy_image = grid_change_to_numpy_image(output_image, num_channels=4)
    else:
        raise RuntimeError(f"Unsupported # image channels: {output_image.shape[0]}")
    numpy_image = numpy.uint8(numpy.rint(numpy_image * 255.0))
    return numpy_image


class FpsStatistics:
    def __init__(self):
        self.count = 100
        self.fps = []

    def add_fps(self, fps: float) -> None:
        self.fps.append(fps)
        while len(self.fps) > self.count:
            del self.fps[0]

    def get_average_fps(self) -> float:
        if len(self.fps) == 0:
            return 0.0
        else:
            return sum(self.fps) / len(self.fps)


class MyFileDropTarget(wx.FileDropTarget):
    def OnDropFiles(self, x, y, filenames):
        if len(filenames) > 1:
            return False
        filename = filenames[0]
        if filename.lower().endswith(".png"):
            logger.info(f"Accepting drop for {filename}")
            main_frame.load_image(filename)
            return True
        elif filename.lower().endswith(".json"):
            logger.info(f"Accepting drop for {filename}")
            main_frame.load_json(filename)
            return True
        logger.info(f"Rejecting drop for {filename}, unsupported file type")
        return False


class MainFrame(wx.Frame):
    """Main app window for THA3 Manual Poser.

    Usage, roughly::

        from tha3.poser.modes.load_poser import load_poser

        model = "separable_float"  # or some other directory containing a model, under "tha3/models"
        device = torch.device("cuda")  # or "cpu", but then will be slow
        poser = load_poser(model, device, modelsdir="tha3/models")

        app = wx.App()
        main_frame = MainFrame(poser, device, model)
        main_frame.Show(True)
        main_frame.timer.Start(30)
        app.MainLoop()
    """
    def __init__(self, poser: Poser, device: torch.device, model: str):
        super().__init__(None, wx.ID_ANY, f"THA3 Manual Poser [{device}] [{model}]")
        self.poser = poser
        self.dtype = self.poser.get_dtype()
        self.device = device
        self.image_size = self.poser.get_image_size()

        self.wx_source_image = None
        self.torch_source_image = None

        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)
        self.init_left_panel()
        self.init_control_panel()
        self.init_right_panel()
        self.main_sizer.Fit(self)

        self.fps_statistics = FpsStatistics()

        self.timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_images, self.timer)

        load_image_id = wx.NewIdRef()
        load_json_id = wx.NewIdRef()
        save_image_id = wx.NewIdRef()
        save_batch_id = wx.NewIdRef()
        focus_preset_id = wx.NewIdRef()
        focus_editor_id = wx.NewIdRef()
        focus_outputindex_id = wx.NewIdRef()
        def focus_presets(event: wx.Event) -> None:
            self.emotion_choice.SetFocus()
        # TODO: Add hotkeys for each morph control group, and for the non-morph control groups.
        def focus_editor(event: wx.Event) -> None:
            if not self.morph_control_panels:
                return
            first_morph_control_panel = list(self.morph_control_panels.values())[0]
            first_morph_control_panel.choice.SetFocus()
        def focus_output_index(event: wx.Event) -> None:
            self.output_index_choice.SetFocus()
        self.Bind(wx.EVT_MENU, self.on_load_image, id=load_image_id)
        self.Bind(wx.EVT_MENU, self.on_load_json, id=load_json_id)
        self.Bind(wx.EVT_MENU, self.on_save_image, id=save_image_id)
        self.Bind(wx.EVT_MENU, self.on_save_all_emotions, id=save_batch_id)
        self.Bind(wx.EVT_MENU, focus_presets, id=focus_preset_id)
        self.Bind(wx.EVT_MENU, focus_editor, id=focus_editor_id)
        self.Bind(wx.EVT_MENU, focus_output_index, id=focus_outputindex_id)
        accelerator_table = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord("O"), load_image_id),
            (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord("O"), load_json_id),
            (wx.ACCEL_CTRL, ord("S"), save_image_id),
            (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord("S"), save_batch_id),
            (wx.ACCEL_CTRL, ord("P"), focus_preset_id),
            (wx.ACCEL_CTRL, ord("E"), focus_editor_id),
            (wx.ACCEL_CTRL, ord("I"), focus_outputindex_id)
        ])
        self.SetAcceleratorTable(accelerator_table)

        self.last_pose = None
        self.last_emotion_index = None
        self.last_output_index = self.output_index_choice.GetSelection()
        self.last_output_numpy_image = None

        self.wx_source_image = None
        self.torch_source_image = None
        self.source_image_bitmap = wx.Bitmap(self.image_size, self.image_size)
        self.result_image_bitmap = wx.Bitmap(self.image_size, self.image_size)
        self.source_image_dirty = True
        self.update_in_progress = False

    def on_erase_background(self, event: wx.Event) -> None:
        pass

    def on_pose_edited(self, event: wx.Event) -> None:
        """Automatically choose the '[custom]' emotion preset (to indicate edited state) when the pose is manually edited."""
        self.emotion_choice.SetSelection(0)
        self.last_emotion_index = 0
        event.Skip()  # allow other handlers for the same event to run

    def init_left_panel(self) -> None:
        """Initialize the input image and emotion preset panel."""
        self.control_panel = wx.Panel(self, style=wx.SIMPLE_BORDER, size=(self.image_size, -1))
        self.left_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        self.left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_panel.SetSizer(self.left_panel_sizer)
        self.left_panel.SetAutoLayout(1)

        self.source_image_panel = wx.Panel(self.left_panel, size=(self.image_size, self.image_size),
                                           style=wx.SIMPLE_BORDER)
        self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
        self.source_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        self.file_drop_target = MyFileDropTarget()
        self.source_image_panel.SetDropTarget(self.file_drop_target)
        self.left_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

        # Emotion picker.
        self.emotions, self.emotion_names = load_emotion_presets()

        # # Horizontal emotion picker layout; looks bad, text label vertical alignment is wrong.
        # self.emotion_panel = wx.Panel(self.left_panel, style=wx.SIMPLE_BORDER, size=(-1, -1))
        # self.emotion_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # self.emotion_panel.SetSizer(self.emotion_panel_sizer)
        # self.emotion_panel.SetAutoLayout(1)
        # self.emotion_panel_sizer.Add(wx.StaticText(self.emotion_panel, label="Emotion presets", style=wx.ALIGN_CENTRE_HORIZONTAL))
        # self.emotion_choice = wx.Choice(self.emotion_panel, choices=self.emotion_names)
        # self.emotion_choice.SetSelection(0)
        # self.emotion_panel_sizer.Add(self.emotion_choice, 0, wx.EXPAND)
        # left_panel_sizer.Add(self.emotion_panel, 0, wx.EXPAND)

        # Vertical emotion picker layout.
        self.left_panel_sizer.Add(wx.StaticText(self.left_panel, label="Emotion preset [Ctrl+P]", style=wx.ALIGN_LEFT))
        self.emotion_choice = wx.Choice(self.left_panel, choices=self.emotion_names)
        self.emotion_choice.SetSelection(0)
        self.left_panel_sizer.Add(self.emotion_choice, 0, wx.EXPAND)

        self.load_image_button = wx.Button(self.left_panel, wx.ID_ANY, "\nLoad image [Ctrl+O]\n\n")
        self.left_panel_sizer.Add(self.load_image_button, 1, wx.EXPAND)
        self.load_image_button.Bind(wx.EVT_BUTTON, self.on_load_image)

        self.load_json_button = wx.Button(self.left_panel, wx.ID_ANY, "\nLoad JSON [Ctrl+Shift+O]\n\n")
        self.left_panel_sizer.Add(self.load_json_button, 1, wx.EXPAND)
        self.load_json_button.Bind(wx.EVT_BUTTON, self.on_load_json)

        self.left_panel_sizer.Fit(self.left_panel)
        self.main_sizer.Add(self.left_panel, 0, wx.FIXED_MINSIZE)

    def init_control_panel(self) -> None:
        """Initialize the pose editor panel."""
        self.control_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.control_panel.SetSizer(self.control_panel_sizer)
        self.control_panel.SetMinSize(wx.Size(256, 1))

        self.control_panel_sizer.Add(wx.StaticText(self.control_panel, label="Editor [Ctrl+E]", style=wx.ALIGN_CENTER),
                                     wx.SizerFlags().Expand())

        morph_categories = [PoseParameterCategory.EYEBROW,
                            PoseParameterCategory.EYE,
                            PoseParameterCategory.MOUTH,
                            PoseParameterCategory.IRIS_MORPH]
        morph_category_titles = {PoseParameterCategory.EYEBROW: "Eyebrow",
                                 PoseParameterCategory.EYE: "Eye",
                                 PoseParameterCategory.MOUTH: "Mouth",
                                 PoseParameterCategory.IRIS_MORPH: "Iris"}
        self.morph_control_panels = {}
        for category in morph_categories:
            param_groups = self.poser.get_pose_parameter_groups()
            filtered_param_groups = [group for group in param_groups if group.get_category() == category]
            if len(filtered_param_groups) == 0:
                continue
            control_panel = MorphCategoryControlPanel(
                self.control_panel,
                morph_category_titles[category],
                category,
                self.poser.get_pose_parameter_groups())
            # Trigger the choice of the "[custom]" emotion preset when the pose is edited in this panel.
            control_panel.choice.Bind(wx.EVT_CHOICE, self.on_pose_edited)
            control_panel.left_slider.Bind(wx.EVT_SLIDER, self.on_pose_edited)
            control_panel.right_slider.Bind(wx.EVT_SLIDER, self.on_pose_edited)
            control_panel.checkbox.Bind(wx.EVT_CHECKBOX, self.on_pose_edited)
            self.morph_control_panels[category] = control_panel
            self.control_panel_sizer.Add(control_panel, 0, wx.EXPAND)

        self.non_morph_control_panels = {}
        non_morph_categories = [PoseParameterCategory.IRIS_ROTATION,
                                PoseParameterCategory.FACE_ROTATION,
                                PoseParameterCategory.BODY_ROTATION,
                                PoseParameterCategory.BREATHING]
        for category in non_morph_categories:
            param_groups = self.poser.get_pose_parameter_groups()
            filtered_param_groups = [group for group in param_groups if group.get_category() == category]
            if len(filtered_param_groups) == 0:
                continue
            control_panel = SimpleParamGroupsControlPanel(self.control_panel,
                                                          category,
                                                          self.poser.get_pose_parameter_groups())
            # Trigger the choice of the "[custom]" emotion preset when the pose is edited in this panel.
            for slider in control_panel.sliders:
                slider.Bind(wx.EVT_SLIDER, self.on_pose_edited)
            self.non_morph_control_panels[category] = control_panel
            self.control_panel_sizer.Add(control_panel, 0, wx.EXPAND)

        self.fps_text = wx.StaticText(self.control_panel, label="FPS counter will appear here")
        self.fps_text.SetForegroundColour((0, 255, 0))
        self.control_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())

        self.control_panel_sizer.Fit(self.control_panel)
        self.main_sizer.Add(self.control_panel, 1, wx.FIXED_MINSIZE)

    def init_right_panel(self) -> None:
        """Initialize the output image and output controls panel."""
        self.right_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        right_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(right_panel_sizer)
        self.right_panel.SetAutoLayout(1)

        self.result_image_panel = wx.Panel(self.right_panel,
                                           size=(self.image_size, self.image_size),
                                           style=wx.SIMPLE_BORDER)
        self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
        self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        self.output_index_choice = wx.Choice(
            self.right_panel,
            choices=[str(i) for i in range(self.poser.get_output_length())])
        self.output_index_choice.SetSelection(0)
        right_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)
        right_panel_sizer.Add(wx.StaticText(self.right_panel, label="Output index [Ctrl+I] [meaning depends on the model]", style=wx.ALIGN_LEFT))
        right_panel_sizer.Add(self.output_index_choice, 0, wx.EXPAND)

        self.save_image_button = wx.Button(self.right_panel, wx.ID_ANY, "\nSave image and JSON [Ctrl+S]\n\n")
        right_panel_sizer.Add(self.save_image_button, 1, wx.EXPAND)
        self.save_image_button.Bind(wx.EVT_BUTTON, self.on_save_image)

        self.save_all_emotions_button = wx.Button(self.right_panel, wx.ID_ANY, "\nBatch save image and JSON from all presets [Ctrl+Shift+S]\n\n")
        right_panel_sizer.Add(self.save_all_emotions_button, 1, wx.EXPAND)
        self.save_all_emotions_button.Bind(wx.EVT_BUTTON, self.on_save_all_emotions)

        right_panel_sizer.Fit(self.right_panel)
        self.main_sizer.Add(self.right_panel, 0, wx.FIXED_MINSIZE)

    def create_param_category_choice(self, param_category: PoseParameterCategory) -> wx.Choice:
        """Create a `wx.Choice` dropdown for the given pose parameter category (eyebrow, eye, ...)."""
        params = []
        for param_group in self.poser.get_pose_parameter_groups():
            if param_group.get_category() == param_category:
                params.append(param_group.get_group_name())
        choice = wx.Choice(self.control_panel, choices=params)
        if len(params) > 0:
            choice.SetSelection(0)
        return choice

    def on_load_image(self, event: wx.Event) -> None:
        """Ask the user for and load an input image."""
        dir_name = "tha3/images"  # This is where `example.png` is.
        file_dialog = wx.FileDialog(self, "Load input image", dir_name, "", "PNG files (*.png)|*.png", wx.FD_OPEN)
        try:
            if file_dialog.ShowModal() == wx.ID_OK:
                image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
                self.load_image(image_file_name)
        finally:
            file_dialog.Destroy()

    def on_load_json(self, event: wx.Event) -> None:
        """Ask the user for and load a custom emotion JSON file."""
        dir_name = "output"  # This is where "Save image and JSON" puts them by default, so...
        file_dialog = wx.FileDialog(self, "Load JSON", dir_name, "", "JSON files (*.json)|*.json", wx.FD_OPEN)
        try:
            if file_dialog.ShowModal() == wx.ID_OK:
                json_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
                self.load_json(json_file_name)
        finally:
            file_dialog.Destroy()

    def load_image(self, image_file_name: str) -> None:
        """Load an input image."""
        try:
            pil_image = resize_PIL_image(extract_PIL_image_from_filelike(image_file_name),
                                         (self.poser.get_image_size(), self.poser.get_image_size()))
            w, h = pil_image.size
            if pil_image.mode != "RGBA":  # input image must have an alpha channel
                self.wx_source_image = None
                self.torch_source_image = None
                logger.warning(f"Incompatible input image (no alpha channel), canceling load: {image_file_name}")
            else:
                logger.info(f"Loaded input image: {image_file_name}")
                self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image)\
                    .to(self.device).to(self.dtype)
            self.source_image_dirty = True
            self.Refresh()
            self.Update()
        except Exception as exc:
            logger.error(f"Could not load image {image_file_name}, reason: {exc}")
            message_dialog = wx.MessageDialog(self, f"Could not load image {image_file_name}, reason: {exc}", "THA3 Manual Poser", wx.OK)
            try:
                message_dialog.ShowModal()
            finally:
                message_dialog.Destroy()

    def load_json(self, json_file_name: str) -> None:
        """Load a custom emotion JSON file."""
        try:
            # Load the emotion JSON file
            with open(json_file_name, "r") as json_file:
                emotions_from_json = json.load(json_file)
            # TODO: Here we just take the first emotion from the file.
            if not emotions_from_json:
                logger.warning(f"No emotions defined in given JSON file, canceling load: {json_file_name}")
                return
            first_emotion_name = list(emotions_from_json.keys())[0]  # first in insertion order, i.e. topmost in file
            if len(emotions_from_json) > 1:
                logger.warning(f"File {json_file_name} contains multiple emotions, loading the first one '{first_emotion_name}'.")
            posedict = emotions_from_json[first_emotion_name]
            pose = self.posedict_to_pose(posedict)

            # Apply loaded emotion
            self.set_current_pose(pose)

            # Auto-select "[custom]"
            self.emotion_choice.SetSelection(0)

            # Do the GUI update after any pending events have processed
            def on_load_json_cont():
                self.Refresh()
                self.Update()
            wx.CallAfter(on_load_json_cont)
        except Exception as exc:
            logger.error(f"Could not load JSON {json_file_name}, reason: {exc}")
            message_dialog = wx.MessageDialog(self, f"Could not load JSON {json_file_name}, reason: {exc}", "THA3 Manual Poser", wx.OK)
            try:
                message_dialog.ShowModal()
            finally:
                message_dialog.Destroy()
        else:
            logger.info(f"Loaded JSON {json_file_name}")

    def paint_source_image_panel(self, event: wx.Event) -> None:
        wx.BufferedPaintDC(self.source_image_panel, self.source_image_bitmap)

    def paint_result_image_panel(self, event: wx.Event) -> None:
        wx.BufferedPaintDC(self.result_image_panel, self.result_image_bitmap)

    def draw_message_to_bitmap(self, bitmap: wx.Bitmap, message: str) -> None:
        """Write (in-place) a placeholder one-line message into a given bitmap. Used when no image is loaded yet."""
        dc = wx.MemoryDC()
        dc.SelectObject(bitmap)

        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent(message)
        dc.DrawText(message, (self.image_size - w) // 2, (self.image_size - - h) // 2)

        del dc

    def get_current_pose(self) -> List[float]:
        """Get the current pose of the character as a list of morph values (in the order the models expect them).

        We do this by reading the values from the UI elements in the control panel.
        """
        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]
        for morph_control_panel in self.morph_control_panels.values():
            morph_control_panel.write_to_pose(current_pose)
        for rotation_control_panel in self.non_morph_control_panels.values():
            rotation_control_panel.write_to_pose(current_pose)
        return current_pose

    def set_current_pose(self, pose: List[float]) -> None:
        """Write `pose` to the UI controls in the editor panel.

        Note that after this, you have to flush the wx event queue for the GUI to update itself correctly.
        So if you call `set_current_pose` and intend to do something immediately, instead do that something
        using `wx.CallAfter`.
        """
        # `update_images` calls us; but if it is not already running (i.e. if we are called by something else),
        # we should not let it run until the pose update is complete.
        old_update_in_progress = self.update_in_progress
        self.update_in_progress = True
        try:
            for panel in self.morph_control_panels.values():
                panel.read_from_pose(pose)
            for panel in self.non_morph_control_panels.values():
                panel.read_from_pose(pose)
        finally:
            self.update_in_progress = old_update_in_progress

    def update_images(self, event: wx.Event) -> None:  # This runs on a timer; keep the code as light as reasonably possible.
        """Update the input and output images.

        The output image is rendered when necessary.
        """
        # Though we're running in a single thread, the `wx.CallAfter` makes this concurrent,
        # so the contents of this function should really be in a critical section.
        #
        # TODO: Atomic locking/mutex.
        if self.update_in_progress:
            return
        self.update_in_progress = True
        last_update_time = time.time_ns()
        actually_rendered = False  # For the FPS counter, to detect if a render actually took place.

        # Apply the currently selected emotion, unless "[custom]" is selected, in which case skip this.
        # Note this may modify the current pose, hence we do this first.
        current_emotion_index = self.emotion_choice.GetSelection()
        if current_emotion_index != 0 and current_emotion_index != self.last_emotion_index:  # not "[custom]"
            self.last_emotion_index = current_emotion_index
            emotion_name = self.emotion_choice.GetString(current_emotion_index)
            logger.info(f"Loading emotion preset {emotion_name}")
            posedict = self.emotions[emotion_name]
            pose = self.posedict_to_pose(posedict)
            self.set_current_pose(pose)
            current_pose = pose
        else:
            current_pose = self.get_current_pose()

        # `wx.Slider.SetValue` needs to handle some events to update the visible thumb position,
        # so we must defer the rest of our processing until currently pending events have been processed.
        #
        #   https://forums.wxwidgets.org/viewtopic.php?t=47723
        #
        # This code looks like JavaScript apps did before promises became a thing, essentially
        # for the same reason. Manually spelling out async continuations is so 1990s, but:
        #
        #   - These classical GUI toolkits were invented before the async/await syntax, so meh.
        #   - In a Lisp, we'd phrase this as something like `(wx-call-after-with (lambda: ...))`
        #     to have a clearer presentation order (we want to "call now the following thing...",
        #     not "here's a lengthy thing and by the way, call it now"), but Python doesn't have
        #     a proper lambda, so meh.
        #
        # Just keep in mind this "function" (technically, closure) is just a block of code
        # to be run slightly later.
        def update_images_cont() -> None:
            try:
                if not self.source_image_dirty \
                        and self.last_pose is not None \
                        and self.last_pose == current_pose \
                        and self.last_output_index == self.output_index_choice.GetSelection():
                    return
                self.last_pose = current_pose
                self.last_output_index = self.output_index_choice.GetSelection()

                if self.torch_source_image is None:
                    self.draw_message_to_bitmap(self.source_image_bitmap, "[No image loaded]")
                    self.draw_message_to_bitmap(self.result_image_bitmap, "[No image loaded]")
                    self.source_image_dirty = False
                    return

                if self.source_image_dirty:
                    dc = wx.MemoryDC()
                    dc.SelectObject(self.source_image_bitmap)
                    dc.Clear()
                    dc.DrawBitmap(self.wx_source_image, 0, 0)
                    self.source_image_dirty = False

                pose = torch.tensor(current_pose, device=self.device, dtype=self.dtype)
                output_index = self.output_index_choice.GetSelection()
                with torch.no_grad():
                    output_image = self.poser.pose(self.torch_source_image, pose, output_index)[0].detach().cpu()

                numpy_image = convert_output_image_from_torch_to_numpy(output_image)
                self.last_output_numpy_image = numpy_image
                wx_image = wx.ImageFromBuffer(
                    numpy_image.shape[0],
                    numpy_image.shape[1],
                    numpy_image[:, :, 0:3].tobytes(),
                    numpy_image[:, :, 3].tobytes())
                wx_bitmap = wx_image.ConvertToBitmap()

                dc = wx.MemoryDC()
                dc.SelectObject(self.result_image_bitmap)
                dc.Clear()
                dc.DrawBitmap(wx_bitmap,
                              (self.image_size - numpy_image.shape[0]) // 2,
                              (self.image_size - numpy_image.shape[1]) // 2,
                              True)
                del dc

                nonlocal actually_rendered
                actually_rendered = True
            finally:
                # Set up another async continuation to finish things up.
                #
                # I have no idea why the final forced Refresh/Update must wait until other pending
                # GUI events have been processed. When `update_images_cont` *starts*, the sliders
                # should have been set to their final positions, and those events processed already.
                #
                # But for whatever reason, this fixes the remaining flakiness with the GUI element
                # not visually updating when using `slider.SetValue`.
                #
                # Either I'm missing something important, or that's just GUI programming for you.
                #
                # Well, to look at the bright side, at least this gives us a place where we can
                # compute the render FPS after the render is actually complete.
                def update_images_cont2() -> None:
                    self.Refresh()
                    self.Update()

                    # Update FPS counter, but only if a render actually took place (we want to measure the render speed only).
                    if actually_rendered:
                        elapsed_time = time.time_ns() - last_update_time
                        fps = 1.0 / (elapsed_time / 10**9)
                        if self.torch_source_image is not None:
                            self.fps_statistics.add_fps(fps)
                        self.fps_text.SetLabelText(f"Render: {self.fps_statistics.get_average_fps():0.2f} FPS")

                    self.update_in_progress = False
                wx.CallAfter(update_images_cont2)
        wx.CallAfter(update_images_cont)

    def current_pose_to_posedict(self) -> Dict[str, float]:
        """Convert the character's current pose into a posedict for saving into an emotion JSON."""
        current_pose_values = self.get_current_pose()
        current_pose_dict = dict(zip(posedict_keys, current_pose_values))
        return current_pose_dict

    def posedict_to_pose(self, posedict: Dict[str, float]) -> List[float]:
        """Convert a posedict (from an emotion JSON) into a list of morph values (in the order the models expect them)."""
        # sanity check
        unrecognized_keys = set(posedict.keys()) - set(posedict_keys)
        if unrecognized_keys:
            logger.warning(f"Ignoring unrecognized keys in posedict: {unrecognized_keys}")
        # Missing keys are fine - keys for zero values can simply be omitted.

        pose = [0.0 for i in range(self.poser.get_num_parameters())]
        for idx, key in enumerate(posedict_keys):
            pose[idx] = posedict.get(key, 0.0)
        return pose

    def on_save_image(self, event: wx.Event) -> None:
        """Ask the user for destination and save the output image.

        The pose is automatically saved into the same directory as the output image, with
        file name determined from the image file name (e.g. "my_emotion.png" -> "my_emotion.json").
        """
        if self.last_output_numpy_image is None:
            logger.info("There is no output image to save.")
            return
        dir_name = "output"
        file_dialog = wx.FileDialog(self, "Save output image", dir_name, "", "PNG images (*.png)|*.png", wx.FD_SAVE)
        # try:  # multi-format support: select PNG save format by default if available
        #     file_dialog.SetFilterIndex(output_ext_to_index["png"])
        # except Exception:
        #     pass
        try:
            if file_dialog.ShowModal() == wx.ID_OK:
                image_file_name = file_dialog.GetFilename()
                # idx = file_dialog.GetFilterIndex()
                # ext = output_index_to_ext[idx]
                # if ext and not image_file_name.lower().endswith(f".{ext}"):  # usability: auto-add selected file extension
                #     image_file_name += f".{ext}"
                if not image_file_name.lower().endswith(".png"):  # usability: auto-add .png file extension
                    image_file_name += ".png"

                image_file_name = os.path.join(file_dialog.GetDirectory(), image_file_name)
                try:
                    if os.path.exists(image_file_name):
                        message_dialog = wx.MessageDialog(self, f"Overwrite {image_file_name}?", "THA3 Manual Poser",
                                                          wx.YES_NO | wx.ICON_QUESTION)
                        try:
                            result = message_dialog.ShowModal()
                            if result == wx.ID_NO:
                                return
                            self.save_numpy_image(self.last_output_numpy_image, image_file_name)
                        finally:
                            message_dialog.Destroy()
                    else:
                        self.save_numpy_image(self.last_output_numpy_image, image_file_name)

                except Exception as exc:
                    logger.error(f"Could not save {image_file_name}, reason: {exc}")
                    message_dialog = wx.MessageDialog(self, f"Could not save {image_file_name}, reason: {exc}", "THA3 Manual Poser", wx.OK)
                    try:
                        message_dialog.ShowModal()
                    finally:
                        message_dialog.Destroy()

                else:  # Since it is possible to save the image and JSON to "tha3/emotions", on a successful save, refresh the emotion presets list.
                    logger.info(f"Saved image {image_file_name}")

                    current_emotion_old_index = self.emotion_choice.GetSelection()
                    current_emotion_name = self.emotion_choice.GetString(current_emotion_old_index)

                    self.emotions, self.emotion_names = load_emotion_presets()
                    self.emotion_choice.SetItems(self.emotion_names)

                    current_emotion_new_index = self.emotion_choice.FindString(current_emotion_name)
                    self.emotion_choice.SetSelection(current_emotion_new_index)
        finally:
            file_dialog.Destroy()

    def on_save_all_emotions(self, event: wx.Event) -> None:
        """Ask the user for a destination directory, and batch save an output image using each of the emotion presets.

        Does not affect the output image displayed in the GUI.
        """
        if self.torch_source_image is None:
            logger.info("No image is loaded, nothing to batch.")
            return

        dir_dialog = wx.DirDialog(self, "Choose directory to save in", "output", wx.DD_DEFAULT_STYLE)
        try:
            if dir_dialog.ShowModal() == wx.ID_OK:
                dir_name = dir_dialog.GetPath()
                if not os.path.exists(dir_name):
                    p = pathlib.Path(dir_name).expanduser().resolve()
                    pathlib.Path.mkdir(p, parents=True, exist_ok=True)
                if os.listdir(dir_name):  # not empty
                    # TODO: provide replace and merge modes
                    message_dialog = wx.MessageDialog(self, f"Directory is not empty: {dir_name}.\nAny files corresponding to emotion presets will be overwritten.\nProceed?", "THA3 Manual Poser",
                                                      wx.YES_NO | wx.ICON_QUESTION)
                    try:
                        result = message_dialog.ShowModal()
                        if result == wx.ID_NO:
                            return
                    finally:
                        message_dialog.Destroy()

                logger.info(f"Batch saving output based on all emotion presets to directory {dir_name}...")
                for emotion_name, posedict in self.emotions.items():
                    if emotion_name.startswith("[") and emotion_name.endswith("]"):
                        continue  # skip "[custom]" and "[reset]"
                    try:
                        pose = self.posedict_to_pose(posedict)

                        posetensor = torch.tensor(pose, device=self.device, dtype=self.dtype)
                        output_index = self.output_index_choice.GetSelection()
                        with torch.no_grad():
                            output_image = self.poser.pose(self.torch_source_image, posetensor, output_index)[0].detach().cpu()
                        numpy_image = convert_output_image_from_torch_to_numpy(output_image)

                        image_file_name = os.path.join(dir_name, f"{emotion_name}.png")
                        self.save_numpy_image(numpy_image, image_file_name)

                        logger.info(f"Saved image {image_file_name}")
                    except Exception as exc:
                        logger.error(f"Could not save {image_file_name}, reason: {exc}")
                logger.info("Batch save finished.")
        finally:
            dir_dialog.Destroy()

    def save_numpy_image(self, numpy_image: numpy.array, image_file_name: str) -> None:
        """Save the output image.

        Output format is determined by file extension (which must be supported by the installed `Pillow`).
        Automatically save also the corresponding settings as JSON.

        The settings are saved into the same directory as the output image, with file name determined
        from the image file name (e.g. "my_emotion.png" -> "my_emotion.json").
        """
        pil_image = PIL.Image.fromarray(numpy_image, mode="RGBA")
        os.makedirs(os.path.dirname(image_file_name), exist_ok=True)
        pil_image.save(image_file_name)

        data_dict = self.current_pose_to_posedict()
        json_file_path = os.path.splitext(image_file_name)[0] + ".json"

        filename_without_extension = os.path.splitext(os.path.basename(image_file_name))[0]
        data_dict_with_filename = {filename_without_extension: data_dict}  # Create a new dict with the filename as the key

        try:
            with open(json_file_path, "w") as file:
                json.dump(data_dict_with_filename, file, indent=4)
        except Exception:
            pass
        else:
            logger.info(f"Saved JSON {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THA 3 Manual Poser. Pose a character image manually. Useful for generating static expression images.")
    parser.add_argument("--model",
                        type=str,
                        required=False,
                        default="separable_float",
                        choices=["standard_float", "separable_float", "standard_half", "separable_half"],
                        help="The model to use. 'float' means fp32, 'half' means fp16.")
    parser.add_argument("--device",
                        type=str,
                        required=False,
                        default="cuda",
                        choices=["cpu", "cuda"],
                        help='The device to use for PyTorch ("cuda" for GPU, "cpu" for CPU).')
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        poser = load_poser(args.model, device, modelsdir="tha3/models")
    except RuntimeError as e:
        logger.error(e)
        sys.exit()

    # Create the "talkinghead/output" directory if it doesn't exist. This is our default save location.
    p = pathlib.Path("output").expanduser().resolve()
    pathlib.Path.mkdir(p, parents=True, exist_ok=True)

    app = wx.App()
    main_frame = MainFrame(poser, device, args.model)
    main_frame.Show(True)
    main_frame.timer.Start(30)
    app.MainLoop()
