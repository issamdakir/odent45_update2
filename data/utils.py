import bpy  # type: ignore
from bpy.app.handlers import persistent  # type: ignore
from bpy.props import (  # type: ignore
    StringProperty,
)
import gpu  # type: ignore
from gpu_extras.batch import batch_for_shader  # type: ignore
import blf  # type: ignore

import os
import math
from os.path import dirname, join, abspath, expanduser, exists, isfile, isdir
from glob import glob
import zipfile
from importlib import import_module
import socket
import webbrowser
import sys
import json
import shutil
import requests # type: ignore
import tempfile
from requests.exceptions import HTTPError, Timeout, RequestException # type: ignore
from time import sleep, perf_counter
from datetime import datetime
import subprocess
import platform
import weakref
import importlib
from . import bl_info
import traceback
import pathlib


############################################
DRAW_HANDLERS = []
WATCH_OBJECT_NAME = "Slices_Pointer"
CHECK_INTERVAL = 0.1  # seconds between checks
# --- Internal State ---
_last_loc = None
_last_rot = None
_callback = None

def get_socket_geonodes(ng, item_name):
    for item in ng.interface.items_tree:
        if item.item_type == 'SOCKET' and item.name == item_name:
            return item.identifier

def set_socket_value_geonodes(mod, item_name, value):
    ng = mod.node_group
    mod[get_socket_geonodes(ng, item_name)] = value

def collapse_all_collections():
    """Collapse all Outliners across all screens."""
    # Find all Outliner regions across all screens in one go
    outliners = [
        (s, a, next(r for r in a.regions if r.type == 'WINDOW'))
        for s in bpy.data.screens
        for a in s.areas if a.type == 'OUTLINER'
    ]

    for screen, area, region in outliners:
        with bpy.context.temp_override(screen=screen, area=area, region=region):
            # Close all levels by calling the operator
            bpy.ops.outliner.show_one_level(open=False)

def odent_override_context(workspace = None, screen=None, area_type="VIEW_3D", region_type="WINDOW"):
    """Override the context area region with the given area and region types"""
    
    try :
        
        ws = workspace
        if isinstance(workspace, bpy.types.WorkSpace):
            ws = workspace
        elif isinstance(workspace, str):
            ws = bpy.data.workspaces[workspace]
        odent_log([f"workspace : {ws.name}"])
        scr = ws.screens[0]
        if isinstance(screen, bpy.types.Screen):
            scr = screen
        elif isinstance(screen, (int, str)):
            scr = bpy.data.screens[screen]
        odent_log([f"screen : {scr.name}"])
        # Find the area and region types
        for i, area in enumerate(scr.areas):
            if area.type == area_type:
                print(f"area {area_type} : {i}")
                space = area.spaces.active
                for region in area.regions:
                    if region.type == region_type:
                        override = {
                            "workspace": ws,
                            "screen": scr,
                            "area": area,
                            "space": space,
                            "region": region,
                        }
                        return override
    except Exception as e :
        odent_log([f"odent_override_context error : {e}"])
        traceback.print_exc()
    # No area/region found, return empty set
    return None

def module_installed(name: str) -> bool:
    """Check if a Python module is installed."""
    return importlib.util.find_spec(name) is not None


def activate_obj(obj):
    if bpy.context.object:
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)


def replace_dir(src, dst):
    dst_parent = dirname(dst)
    if exists(src) and exists(dst_parent):
        # sleep(3)
        if exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst_parent)


def clear_terminal():
    os.system("cls") if os.name == "nt" else os.system("clear")


def get_odent_version(filepath=None):
    if filepath is None:
        return "####"

    try:
        with open(filepath, "r") as rf:
            lines = rf.readlines()
            version = float(lines[0])
            return version
    except Exception as er:
        txt_message = [f""]
        odent_log(txt_message)
        return "####"


def update_is_availible():
    if OdentConstants.ADDON_VER_DATE == "####" or not isConnected():
        return None
    remote_version, success, error_txt_list = get_update_version()

    if not success or remote_version <= OdentConstants.ADDON_VER_DATE:
        return None

    return remote_version  # float casted to string


class AreaTagManager:
    def __init__(self):
        self._data = weakref.WeakKeyDictionary()

    def tag(self, area, key, value):
        """Assign a value to a custom key for a given area."""
        if area not in self._data:
            self._data[area] = {}
        self._data[area][key] = value

    def get(self, area, key, default=None):
        """Retrieve the value of a custom key for a given area."""
        return self._data.get(area, {}).get(key, default)

    def has(self, area, key):
        """Check if a key exists for the given area."""
        return key in self._data.get(area, {})

    def remove(self, area, key):
        """Remove a key for the given area, if it exists."""
        if area in self._data and key in self._data[area]:
            del self._data[area][key]

    def clear(self, area):
        """Remove all tags for a given area."""
        if area in self._data:
            del self._data[area]


class OdentColors:
    white = [0.8, 0.8, 0.8, 1.0]
    black = [0.0, 0.0, 0.0, 1.0]
    trans = [0.8, 0.8, 0.8, 0.0]
    red = [1.0, 0.0, 0.0, 1.0]
    orange = [0.8, 0.258385, 0.041926, 1.0]
    yellow = [0.4, 0.4, 0.1, 1]
    green = [0, 1, 0.2, 0.7]
    blue = [0.0, 0.5, 0.8, 0.1]
    olive = [0.8, 0.6, 0.0, 0.7]
    base = [0.7, 0.7, 0.7, 0.05]  # [0.7,0.65,0.55,0.05]
    default = olive


class OdentConstants:
    """Odent constants"""

    DEBUG = 1
    ODENT_LIB_NAME = "Odent_Library"
    ODENT_LIB_ARCHIVE_NAME = "Odent_Library_Archive"
    ADDON_VER_NAME = "ODENT_Version.txt"
    ODENT_MODULES_NAME = "odent_modules"
    CONFIG_ZIP_NAME = "config.zip"
    STARTUP_FILE_NAME = "startup.blend"

    ADDON_NAME = bl_info["name"]
    ADDON_RELEASE = "release"

    ADDON_DIR = dirname(abspath(__file__))

    BLENDER_ROOT_PATH = dirname(dirname(dirname(ADDON_DIR)))
    RESOURCES = join(ADDON_DIR, "Resources")
    ADDON_VER_PATH = join(RESOURCES, ADDON_VER_NAME)
    ADDON_VER_DATE = get_odent_version(filepath=ADDON_VER_PATH)
    CONFIG_ZIP_PATH = join(RESOURCES, CONFIG_ZIP_NAME)
    STARTUP_FILE_PATH = join(RESOURCES, STARTUP_FILE_NAME)
    ODENT_LIBRARY_PATH = join(ADDON_DIR, ODENT_LIB_NAME)
    ODENT_LIB_IS_OK = False
    ODENT_LIBRARY_ARCHIVE_PATH = join(RESOURCES, ODENT_LIB_ARCHIVE_NAME)

    ODENT_MODULES_PATH = join(ADDON_DIR, ODENT_MODULES_NAME)
    MESH_REG_AUTO_PATH = join(ADDON_DIR, "operators", "MeshRegAuto.exe")
    DATA_BLEND_FILE = join(RESOURCES, "BlendData", "ODENT_BlendData.blend")
    ODENT_APP_TEMPLATE_PATH = join(RESOURCES, "odent_app_template.zip")
    ODENT_ICONS_PATH = join(RESOURCES, "icons")
    
    ODENT_ICONS = None

    BLF_INFO = {
        "fontid": 0,
        "size": 14,
    }
    TEST_URL = "pypi.org"
    ADDON_UPDATE_URL = "https://github.com/issamdakir/odent45_update2/zipball/main"
    ADDON_UPDATE_NAME = "Odent_45_update"
    UPDATE_MAP_JSON = "update_data_map.json"
    UPDATE_VERSION_URL = "https://raw.githubusercontent.com/issamdakir/odent45_update2/main/data/ODENT_Version.txt"
    TELEGRAM_LINK = "https://t.me/odent_blender"
    REQ_DICT = {
        "SimpleITK": "SimpleITK",
        "vtk": "vtk",
        "cv2": "opencv-contrib-python",
        # "itk": "itk",
    }
    AI_MODULES_DICT = {}

    MAIN_WORKSPACE_NAME = "Odent Main"
    SLICER_WORKSPACE_NAME = "Odent Slicer"

    VISUALISATION_MODE_PCD = "Point cloud 3D"
    VISUALISATION_MODE_TEXTURED = "Textured 3D Vizualisation"

    VOXEL_OBJECT_NAME = "Voxel_Vizualization"
    VOXEL_OBJECT_TYPE = "Voxel_Vizualization_Object"
    DICOM_VIZ_COLLECTION_NAME = "Visualization_3D_Collection"

    VOXEL_PLANE_NAME = "Voxel_Plane"
    VOXEL_PLANE_TYPE = "Voxel_Plane_Object"
    VOXEL_PLANE_MAT_NAME = "Voxel_Plane_Mat"

    VOXEL_IMAGE_NAME = "Voxel_Image"
    VOXEL_IMAGE_TYPE = "Voxel_Image_Object"

    VOXEL_GROUPNODE_NAME = "Voxel_GroupNode"

    SLICES_POINTER_NAME = "Slices_Pointer"
    SLICES_POINTER_TYPE = "slices_pointer"
    SLICES_POINTER_COLLECTION_NAME = "Slices_Pointer_Collection"

    SLICES_SHADER_NAME = "VGS_Dakir_Slices"

    SLICE_PLANE_TYPE = "slice_plane"
    SLICES_COLLECTION_NAME = "Slices_Collection"

    AXIAL_SLICE_NAME = "Axial_Slice"
    SAGITTAL_SLICE_NAME = "Sagittal_Slice"
    CORONAL_SLICE_NAME = "Coronal_Slice"
    SLICE_CAM_TYPE = "slice_camera"

    SLICE_IMAGE_TYPE = "slice_image"

    DICOM_MESH_NAME = "DICOM_Mesh"
    DICOM_MESH_TYPE = "dicom_mesh_object"
    SEGMENTS_COLLECTION_NAME = "Segments_Collection"

    # "VGS_Marcos_modified_MinMax"#"VGS_Marcos_modified"  # "VGS_Marcos_01" "VGS_Dakir_01"
    ODENT_VOXEL_SHADER = "VGS_Dakir_(-400-3000)"  # "VGS_Dakir_01"#"VGS_Dakir_MinMax"
    GUIDE_COMPONENTS_COLLECTION_NAME = "Surgical_Guide_Components_Collection"
    BOOL_NODE = "boolean_geonode"
    GUIDE_NAME = "Odent_guide"
    ODENT_IMAGE_METADATA_TAG = "odent_image"
    ODENT_IMAGE_METADATA_KEY = "image_type"
    ODENT_IMAGE_METADATA_UID_KEY = "uid"
    WMIN = -400
    WMAX = 3000
    BONE_UINT_THRESHOLD = 100
    BONE_FLOAT_THRESHOLD = 0.0
    XRAY_ALPHA = 0.9
    CAM_CLIP_OFFSET = 0.75
    CAM_DISTANCE = 100
    COLOR_CURVE_MAP = {1: (0.351, 0.265), 2: (0.721, 0.860)}
    ODENT_VOLUME_NODE_NAME = "odent_volume"
    VIEW_MATRIX = (
        (0.9205, -0.3908, 0.0, -25),
        (0.2772, 0.6528, 0.7050, -90),
        (-0.2755, -0.6489, 0.7092, -25),
        (0.0000, 0.0000, 0.0000, 1.0000),
    )

    # Selected icons :
    RED_ICON = "COLORSET_01_VEC"
    ORANGE_ICON = "COLORSET_02_VEC"
    GREEN_ICON = "COLORSET_03_VEC"
    BLUE_ICON = "COLORSET_04_VEC"
    VIOLET_ICON = "COLORSET_06_VEC"
    YELLOW_ICON = "COLORSET_09_VEC"
    YELLOW_POINT = "KEYTYPE_KEYFRAME_VEC"
    BLUE_POINT = "KEYTYPE_BREAKDOWN_VEC"
    SLICE_SEGMENT_COLOR_RGB = [
        0.28,
        1.0,
        0.0008,
        1.0,
    ]  # [0.62, 1.0, 0.22, 1.0] [0.68, 0.8, 1.0, 1.0]
    SLICE_SEGMENT_COLOR_RATIO = 0.35

    ODENT_TEXT_3D_TYPE = "odent_text_3d"
    ODENT_TEXT_3D_MAT_NAME = "odent_text_3d_mat"
    ODENT_TEXT_3D_MAT_DIFFUSE_COLOR = [0, 0, 1, 1]

    ODENT_IMPLANT_TYPE = "odent_implant"
    ODENT_IMPLANT_NAME_PREFFIX = "Odent_Implant"
    ODENT_IMPLANT_COLLECTION_NAME = "Implant_Collection"
    ODENT_IMPLANT_MODIFIER_NAME = "odent_implant_modifier"
    ODENT_IMPLANT_GEONODE_NAME = "odent_implant_geonode"

    ODENT_IMPLANT_GN_TEETH_ORDER = tuple(
    decade * 10 + unit for decade in range(1, 5) for unit in range(1, 9)
)

    IMPLANT_SAFE_ZONE_TYPE = "odent_implant_safe_zone"
    IMPLANT_SLEEVE_TYPE = "odent_implant_sleeve"
    IMPLANT_PIN_TYPE = "odent_implant_pin"
    FIXING_PIN_TYPE = "odent_fixing_pin"
    FIXING_SLEEVE_TYPE = "odent_fixing_sleeve"

    MHA_PATH_TAG = "mha_path"
    SOURCE_MHA_TAG = "source_mha"
    ODENT_TYPE_TAG = "odent_type"
    IMPLANT_LOCKED_MAT_NAME = "implant_locked_mat"
    PREVIOUS_MAT_NAME_TAG = "odent_previous_mat_name"
    IMPLANT_MAT_TAG = "implant_mat"
    ODENT_IMPLANT_REMOVE_CODE_TAG = "odent_remove_code"
    VOXEL_NODE_NAME_TAG = "voxel_node_name"

    PCD_OBJECT_NAME = "Point_Cloud_Vizualization"
    PCD_OBJECT_TYPE = "Point_Cloud_Vizualization_Object"
    PCD_MAT_NAME = "pcd_mat"
    PCD_INTENSITY_ATTRIBUTE_NAME = "voxel_intensity"
    PCD_THRESHOLD_NODE_NAME = "pcd_threshold"
    PCD_OPACITY_NODE_NAME = "pcd_opacity"
    PCD_POINT_RADIUS_NODE_NAME = "pcd_point_radius"
    PCD_POINT_AUTO_RESIZE_NODE_NAME = "pcd_point_auto_resize"
    PCD_POINT_EMISSION_NODE_NAME = "pcd_emission"
    PCD_GEONODE_NAME = "pcd_geonode"
    PCD_GEONODE_NAME_TAG = "pcd_geonode_name"
    PCD_GEONODE_MODIFIER_NAME = "pcd_geonode_modifier"
    PCD_MAX_POINTS = 2  # millions
    PCD_SAMPLING_METHOD_GRID = "Grid sampling"
    PCD_SAMPLING_METHOD_RANDOM = "Random sampling"

    CUTTERS_COLL_NAME = "Odent Cutters"
    CONNECT_PATH_CUTTER_NAME = "Connected path cutter"
    CONNECT_PATH_CUTTER_TYPE = "connected_path_cutter"
    CONNECT_PATH_CUTTER_MAT = {
        "name": "connected_path_cutter_mat",
        "diffuse_color": [0.1, 0.4, 0.7, 1.0],
        "roughness": 0.3,
    }
    PREVIOUS_MAT_COLOR_TAG = "odent_previous_mat_color"
    LOCKED_TO_POINTER_MAT_NAME = "odent_locked_to_pointer_mat"
    PREVIOUS_ACTIVE_MAT_NAME_TAG = "odent_previous_mat"

    SPLIT_CUTTER_HOOK_POINT = "split_cutter_hook_point"
    CURVE_CUTTER1_TAG = "curvecutter1"
    CURVE_CUTTER2_TAG = "curvecutter2"

    ODENT_TEMP_DIR = join(expanduser("~"), "odent_temp")
    ODENT_SLICES_DIR = join(ODENT_TEMP_DIR, "slices")
    ODENT_VOLUME_TEXTURES_DIR = join(ODENT_TEMP_DIR, "volume_textures")

    SPLINT_MAT_NAME = "mat_odent_splint"
    SPLINT_COLOR_DARK_GREEN = [0.0, 0.23, 0.2, 1.0]
    SPLINT_COLOR_BLUE = [0.0, 0.23, 0.2, 1.0]

    INFO_FOOTER_TEXT_COLOR = OdentColors.black
    INFO_FOOTER_RECT_COLOR = OdentColors.olive
    INFO_FOOTER_BASE_COLOR = OdentColors.base

    REMOTE_VERSION_KEY = "odent_remote_version"

    # using timer to update slices
    SLICES_UPDATE_TIMER_PERIOD = 0.01  # seconds
    # PREVIEW_COLLECTIONS = {}


class ODENT_OT_SupportTelegram(bpy.types.Operator):
    """open telegram odent support link"""

    bl_idname = "wm.odent_support_telegram"
    bl_label = "Odent Support (Telegram)"

    def execute(self, context):
        telegram_url = OdentConstants.TELEGRAM_LINK
        browse(telegram_url)

        return {"FINISHED"}


class ODENT_OT_AddOdentLibrary(bpy.types.Operator):
    """add odent library"""

    bl_idname = "wm.odent_add_odent_library"
    bl_label = "Add Odent Library"

    def execute(self, context):
        with context.temp_override(window=context.window_manager.windows[0]):
            message = ["Adding Odent library..."]
            ODENT_GpuDrawText(message)
            sleep(1)
            message, success = add_odent_libray()
            if not success and message:
                ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.red)
                sleep(3)
                ODENT_GpuDrawText()
                return {"CANCELLED"}
            
            OdentConstants.ODENT_LIB_IS_OK = True
            message = ["FINISHED"]
            ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.green)
            sleep(2)
            ODENT_GpuDrawText()
            return {"FINISHED"}


class ODENT_OT_AddAppTemplate(bpy.types.Operator):
    """add odent application template"""

    bl_idname = "wm.odent_add_app_template"
    bl_label = "Add Odent Template"

    display_message: bpy.props.BoolProperty(default=False)  # type: ignore

    def execute(self, context):
        try:
            bpy.ops.preferences.app_template_install(
                filepath=OdentConstants.ODENT_APP_TEMPLATE_PATH
            )
        except Exception as er:
            repport = {
                "error context": "odent app tmplate install operator",
                "error": er,
            }
            print(f"Handled error : {repport}")
        p = context.preferences
        p.inputs.use_auto_perspective = False
        p.inputs.use_rotate_around_active = True
        p.inputs.use_mouse_depth_navigate = True
        p.inputs.use_zoom_to_mouse = True
        add_odent_libray()
        bpy.ops.wm.save_userpref()

        return {"FINISHED"}


class ODENT_OT_SetConfig(bpy.types.Operator):
    """Set Odent config"""

    bl_idname = "wm.odent_set_config"
    bl_label = "Set Odent Interface"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # context.preferences.use_preferences_save = False
        # bpy.ops.wm.save_userpref()
        success = reset_config_folder()
        print(f"Reset Odent config : Success = {success}")
        if success:
            p = context.preferences
            p.inputs.use_auto_perspective = False
            p.inputs.use_rotate_around_active = True
            p.inputs.use_mouse_depth_navigate = True
            p.inputs.use_zoom_to_mouse = True

            # add_odent_libray()
            bpy.ops.wm.save_userpref()

            with context.temp_override(window=context.window_manager.windows[0]):
                ODENT_GpuDrawText(
                    message_list=["To load Odent workspace restart blender, or open new project."],
                    rect_color=OdentColors.green,
                    sleep_time=3,
                )

        else:
            with context.temp_override(window=context.window_manager.windows[0]):

                ODENT_GpuDrawText(
                    message_list=["Odent Configuration failed."],
                    rect_color=OdentColors.red,
                    sleep_time=2,
                )
        return {"FINISHED"}


class ODENT_OT_checkUpdate(bpy.types.Operator):
    """check addon update"""

    bl_idname = "wm.odent_checkupdate"
    bl_label = "check update"
    bl_options = {"REGISTER", "UNDO"}

    txt = []
    restart = 0

    def modal(self, context, event):
        if not event.type in {"ESC", "RET"}:
            return {"PASS_THROUGH"}

        elif event.type in {"ESC"}:
            ODENT_GpuDrawText(message_list=["Cancelled."], sleep_time=1)
            return {"CANCELLED"}

        elif event.type in {"RET"} and event.value == "PRESS":
            ODENT_GpuDrawText(message_list=["Downloading..."])
            _message, update_root = addon_update_download()
            if _message:
                ODENT_GpuDrawText(
                    messag_liste=_message, rect_color=OdentColors.red, sleep_time=2
                )
                return {"CANCELLED"}
            
            odent_log(f"update zip extracted to : {update_root}")

            ODENT_GpuDrawText(message_list=["Preparing update..."])
            need_restart = addon_update_preinstall(update_root)
            add_odent_libray()
            if need_restart :
                message = ["Please restart blender to finalize Odent update."]
                ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.green)
            else :
                try :
                    bpy.ops.script.reload()
                    message = ["Update completed successfully."]
                    ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.green)
                except Exception as er:
                    odent_log(f"reload scripts error : {er}")
                    message = ["Please restart blender to finalize Odent update."]
                    ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.green)
            
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        addon_version = OdentConstants.ADDON_VER_DATE
        if addon_version == "####":
            info_txt_list = ["Can't get current version!"]
            ODENT_GpuDrawText(
                message_list=info_txt_list, rect_color=OdentColors.red, sleep_time=2
            )
            odent_log(info_txt_list)
            return {"CANCELLED"}

        if not isConnected():
            info_txt_list = ["Odent update error : Please check internet connexion !"]
            odent_log(info_txt_list)
            ODENT_GpuDrawText(
                message_list=info_txt_list, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}

        update_version, success, error_txt_list = get_update_version()

        if not success:
            ODENT_GpuDrawText(
                message_list=error_txt_list, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}

        odent_log(
            [f"Current version = {addon_version}, Remote version = {update_version}"]
        )
        if update_version <= addon_version:
            txt_list = ["Odent is up to date."]

            ODENT_GpuDrawText(
                message_list=txt_list, rect_color=OdentColors.green, sleep_time=2
            )
            return {"FINISHED"}

        txt_list = [
            f"Current version = {addon_version}, New version = {update_version}",
            "<ENTER> : to install last update / <ESC> : to cancel",
        ]
        ODENT_GpuDrawText(message_list=txt_list)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


class TimerLogger:
    def __init__(self, label=""):
        self.label = label
        self.start_time = perf_counter()
        self.last_time = self.start_time
        print(f"[{self.label}] Start")

    def log(self, step_name=""):
        now = perf_counter()
        step_elapsed = now - self.last_time
        total_elapsed = now - self.start_time

        print(
            f"[{self.label}] Step: {step_name} | +{step_elapsed:.6f}s | Total: {total_elapsed:.6f}s"
        )
        self.last_time = now

    def end(self):
        now = perf_counter()
        total_elapsed = now - self.start_time
        print(f"[{self.label}] End | Total elapsed: {total_elapsed:.6f}s")


class ODENT_OT_MessageBox(bpy.types.Operator):
    """Odent popup message"""

    bl_idname = "wm.odent_message_box"
    bl_label = "ODENT INFO"
    bl_options = {"REGISTER"}

    message: StringProperty()  # type: ignore
    icon: StringProperty()  # type: ignore

    def execute(self, context):
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.alert = True
        box.alignment = "EXPAND"
        message = eval(self.message)
        for txt in message:
            row = box.row()
            row.label(text=txt)

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=300)


class ODENT_OT_OdentModulesPipInstaller(bpy.types.Operator):
    """pip install odent modules"""

    bl_idname = "wm.odent_modules_pip_installer"
    bl_label = "Install Odent Modules Online"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not isConnected(debug=True):
            message = ["No internet connection!"]
            odent_log(message)
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        target_dir_path = OdentConstants.ODENT_MODULES_PATH

        with context.temp_override(window=context.window_manager.windows[0]):
            message = pip_install_modules(self.modules_list, target_dir_path)

        if message:
            odent_log(message)
            with context.temp_override(window=context.window_manager.windows[0]):
                ODENT_GpuDrawText(
                    message_list=message, rect_color=OdentColors.red, sleep_time=2
                )
            return {"CANCELLED"}
        else:
            message = [
                f"Modules installed successfully, {OdentConstants.ADDON_NAME} reloading ..."
            ]
            odent_log(message)
            with context.temp_override(window=context.window_manager.windows[0]):
                ODENT_GpuDrawText(message_list=message, sleep_time=2)
            import importlib, sys

            importlib.invalidate_caches()
            for item in self.imports:
                if item in sys.modules:
                    del sys.modules[item]
            bpy.ops.script.reload()

        with context.temp_override(window=context.window_manager.windows[0]):
            ODENT_GpuDrawText()
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        # box.alert = True
        box.alignment = "EXPAND"
        message = [
            "internet access is required to install Odent python modules!",
            "Click <OK> button to start.",
        ]
        g = box.grid_flow(columns=1, align=True)
        g.alert = True
        for txt in message:
            g.label(text=txt)
            
        g = box.grid_flow(columns=1, align=True)
        g.label(text="Modules to install :")
        g.label(text=", ".join(self.modules_list))
        

    def invoke(self, context, event):
        self.modules_list, self.imports = import_required_modules()
        if not self.modules_list:
            message = ["Odent modules already installed!"]
            odent_log(message)
            with context.temp_override(window=context.window_manager.windows[0]):
                ODENT_GpuDrawText(
                    message_list=message, rect_color=OdentColors.green, sleep_time=2
                )
            return {"CANCELLED"}
        
        wm = context.window_manager
        
        return wm.invoke_props_dialog(self, width=600)
    
    ###################################################
    # using modal handler we can't reload addon modules
    ##################################################
    # def modal(self, context, event):
    #     if not event.type in {"ESC", "RET"}:
    #         return {"PASS_THROUGH"}

    #     elif event.type in {"ESC"}:
    #         ODENT_GpuDrawText(message_list=["Cancelled."], sleep_time=1)
    #         return {"CANCELLED"}

    #     elif event.type in {"RET"} and event.value == "PRESS":
    #         self.execute(context)
    # def invoke(self, context, event):
    #     self.modules_list, self.imports = import_required_modules()
    #     if not self.modules_list:
    #         message = ["Odent modules already installed!"]
    #         odent_log(message)
    #         with context.temp_override(window=context.window_manager.windows[0]):
    #             ODENT_GpuDrawText(
    #                 message_list=message, rect_color=OdentColors.green, sleep_time=2
    #             )
    #         return {"CANCELLED"}
    #     message = [
    #         "internet connection needed to install Odent python modules!",
    #         "Press <ESC> to cancel or <RET> to continue",
    #         f"Modules to install : {self.modules_list}",
    #     ]
    #     ODENT_GpuDrawText(message_list=message)
    #     wm = context.window_manager
    #     wm.modal_handler_add(self)
    #     return {"RUNNING_MODAL"}


#########################################################################################
#                                        utils                                          #
#########################################################################################
def reset_config_folder():
    # addon_dir = dirname(abspath(sys.modules.get('Odent-3').__file__))
    blender_root_path = OdentConstants.BLENDER_ROOT_PATH
    # print(f"version_dir : {blender_root_path}")
    config_dir = None
    for e in os.listdir(blender_root_path):
        fullpath = join(blender_root_path, e)
        if isdir(fullpath) and e.lower() == "config":
            config_dir = fullpath
            break
    if not config_dir:
        config_dir = join(blender_root_path, "config")
        os.mkdir(config_dir)
    try:
        shutil.copy2(OdentConstants.STARTUP_FILE_PATH, config_dir)
        success = 1
    except:
        success = 0

    return success


def get_image_diagonal_lenght(size, spacing):
    # Physical dimensions
    dx = size[0] * spacing[0]
    dy = size[1] * spacing[1]
    dz = size[2] * spacing[2]

    # Full 3D diagonal
    diagonal_lenght = math.sqrt(dx**2 + dy**2 + dz**2)
    return diagonal_lenght


def AbsPath(P):
    if P.startswith("//"):
        P = abspath(bpy.path.abspath(P))
    return P


def RelPath(P):
    if not P.startswith("//"):
        P = bpy.path.relpath(abspath(P))
    return P


def pip_install_modules(modules_list, target_dir_path=None):
    """Install required modules using pip"""
    error_message = []

    info_text = f"Updating pip ..."
    ODENT_GpuDrawText(message_list=[info_text])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])
    except subprocess.CalledProcessError as e:
        info_text = f"Updating pip failed!"
        odent_log([f"Updating pip failed! : {e}"])

    if target_dir_path is not None and not exists(target_dir_path):
        os.makedirs(target_dir_path)
    for module in modules_list:
        info_text = f"Installing {module} ..."
        ODENT_GpuDrawText(message_list=[info_text])
        try:
            if target_dir_path is not None:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-U",
                        module,
                        "--target",
                        target_dir_path,
                    ]
                )
            else:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-U", module]
                )
        except subprocess.CalledProcessError as e:
            info_text = f"Installing {module} failled!"
            ODENT_GpuDrawText(message_list=[info_text], rect_color=OdentColors.red)
            error_message.append(module)
    return error_message


def get_incremental_idx(data=None, odent_type=None):

    idx = 0
    if data is not None:
        objects = data
        if odent_type:
            objects = [obj for obj in data if obj.get("odent_type") == odent_type]
            if objects:
                idx = len(objects) + 1
    return idx


def get_incremental_name(name_preffix="", idx1=None, idx2=None, idx3=None):

    name = f"{name_preffix}_{idx1}"
    for idx in [idx2, idx3]:
        if idx is None:
            break
        name += f"_{idx}"
    return name


def get_unique_name(name, data_block):

    i = 1
    unique_name = f"{name}_{i}"
    while True:
        if not data_block.get(unique_name):
            break
        i += 1
        unique_name = f"{name}_{i}"
    return unique_name


def is_linux():
    return platform.system() == "Linux"


def is_wine_installed():
    return shutil.which("wine") is not None


def install_wine():
    message = []
    try:
        print("Installing Wine...")
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "install", "-y", "wine"], check=True)
        subprocess.run(["sudo", "apt", "install", "-y", "wine64"], check=True)
        print("Wine installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error during Wine installation:", e)
        message.append("Wine installation failed.")
    return message


def run_exe_with_wine(exe_path):
    if not os.path.exists(exe_path):
        print(f"File not found: {exe_path}")
        return
    try:
        subprocess.run(["wine", exe_path], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running the exe file with Wine:", e)


# def odent_log(txt_list,header=None,footer=None):
#     _header, _footer = header, footer
#     if _header is None :
#         _header=f"\n{'#'*20} Odent log :  {'#'*20}\n"
#     if _footer is None:
#         _footer=f"\n{'#'*20} End log.\  {'#'*20}\n"

#     print(_header)
#     for line in txt_list :
#         print(line)
#     print(_footer)


def odent_log(txt_list):
    if OdentConstants.DEBUG == 0:
        return
    if isinstance(txt_list, str):
        txt_list = [txt_list]
    str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[ODent {str_time}] " + " | ".join(txt_list)
    print(log_entry)


def get_update_version(filepath=OdentConstants.UPDATE_VERSION_URL):
    success = False
    txt_list = []
    update_version = None
    try:
        r = requests.get(filepath)
        success = r.ok
        if not success:
            odent_log([f"connection to github update server failed! :\n{r.text}"])
            txt_list.append("connection to update server failed!")
        else:
            update_version = float(r.text)

    except Exception as er:
        txt_list.append("connection to update server failed!")
        odent_log([f"connection to github update server failed! :\n{er}"])
        success = False

    return update_version, success, txt_list

def add_odent_libray():
    message = []
    success = 0
    if not exists(OdentConstants.ODENT_LIBRARY_ARCHIVE_PATH):
        message = ["Odent Library archive not found"]
        return message, success
    _libs_collection_group = bpy.context.preferences.filepaths.asset_libraries
    odent_lib = _libs_collection_group.get(OdentConstants.ODENT_LIB_NAME)
    if odent_lib:
        idx = [
            i
            for i, l in enumerate(_libs_collection_group)
            if l.name == OdentConstants.ODENT_LIB_NAME
        ][0]
        bpy.ops.preferences.asset_library_remove(index=idx)

    if exists(OdentConstants.ODENT_LIBRARY_PATH):
        shutil.rmtree(OdentConstants.ODENT_LIBRARY_PATH)
    os.mkdir(OdentConstants.ODENT_LIBRARY_PATH)

    archive_list = os.listdir(OdentConstants.ODENT_LIBRARY_ARCHIVE_PATH)
    for _item in archive_list:
        _item_full_path = join(OdentConstants.ODENT_LIBRARY_ARCHIVE_PATH, _item)
        if _item.endswith(".zip"):
            with zipfile.ZipFile(_item_full_path, "r") as zip_ref:
                zip_ref.extractall(OdentConstants.ODENT_LIBRARY_PATH)
        else:
            shutil.copy2(_item_full_path, OdentConstants.ODENT_LIBRARY_PATH)

    bpy.ops.preferences.asset_library_add(directory=OdentConstants.ODENT_LIBRARY_PATH)
    success = 1
    return message, success

def check_odent_library():
    lib_path = OdentConstants.ODENT_LIBRARY_PATH
    if not exists(lib_path) :
        return False
    lib = bpy.context.preferences.filepaths.asset_libraries.get(OdentConstants.ODENT_LIB_NAME)
    if not lib:
        return False
    lib.path = lib_path 
    return True


def import_required_modules(required_modules_dict=OdentConstants.REQ_DICT):
    missing_modules = []
    imports = []
    for mod, pkg in required_modules_dict.items():
        try:
            import_module(mod)
        except ImportError:
            missing_modules.append(pkg)
            imports.append(mod)

    return missing_modules, imports


def isConnected(test_url=OdentConstants.TEST_URL, port=443, _timeout=5, debug=False):
    result = False
    try:
        sock = socket.create_connection((test_url, port), timeout=_timeout)
        if sock is not None:
            sock.close
            result = True

    except OSError:
        pass

    if debug:
        info = "no connexion!"
        if result:
            info = "connected..."
        odent_log([info])
    return result


def browse(url):
    success = 0
    try:
        webbrowser.open(url)
        success = 1
        return success
    except Exception as er:
        print(f"open telegram link error :\n{er}")
        return success


def start_blender_session():
    # print(f"binary path : {bpy.app.binary_path}")
    os.system(f'"{bpy.app.binary_path}"')


def set_modules_path(modules_path=OdentConstants.ODENT_MODULES_PATH):
    if not modules_path in sys.path:
        sys.path.insert(0, OdentConstants.ODENT_MODULES_PATH)


def addon_update_preinstall(update_root):
    need_restart = False
    update_data_map_json = join(update_root, OdentConstants.UPDATE_MAP_JSON)
    update_data_map_dict = open_json(update_data_map_json)
    update_data_dir = join(update_root, "data")
    items = os.listdir(update_data_dir)
    update_data_dict = {}
    for i in items:
        if update_data_map_dict.get(i):
            update_data_dict.update(
                {
                    join(update_data_dir, i): join(
                        OdentConstants.ADDON_DIR, *update_data_map_dict.get(i)
                    )
                }
            )
        else:
            odent_log([f"Update data {i} not found in update map!"])
            continue

    # update_data_dict = {join(update_data_dir,i) : join(OdentConstants.ADDON_DIR,*update_data_map_dict.get(i)) for i in items}
    for src, dst in update_data_dict.items():

        if OdentConstants.ODENT_MODULES_NAME in src.lower():
            shutil.move(src, OdentConstants.RESOURCES)
            need_restart  = True
        else:
            if not exists(dirname(dst)):
                os.makedirs(dirname(dst))

            if exists(dst):
                os.remove(dst) if isfile(dst) else shutil.rmtree(dst)

            shutil.move(src, dirname(dst))
    
    return need_restart


def addon_update_download():

    message = []
    update_root = None
    try:
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        _update_zip_local = join(temp_dir, f"{OdentConstants.ADDON_UPDATE_NAME}.zip")

        # Download the file
        with requests.get(
            OdentConstants.ADDON_UPDATE_URL, stream=True, timeout=10
        ) as r:
            try:
                r.raise_for_status()
            except HTTPError as http_err:
                txt = "HTTP error occurred"
                odent_log([f"{txt} : {http_err}"])
                message.extend(["Server connection error!"])
                return message, update_root
            except ConnectionError as conn_err:
                txt = "Server connection error!"
                odent_log([f"{txt} : {conn_err}"])
                message.extend([txt])
                return message, update_root
            except Timeout as timeout_err:
                txt = "Timeout error occurred"
                odent_log([f"{txt} : {timeout_err}"])
                message.extend(["Server connection error!"])
                return message, update_root
            except RequestException as req_err:
                txt = f"Error during requests to {OdentConstants.ADDON_UPDATE_URL}"
                odent_log([f"{txt} : {req_err}"])
                message.extend(["Server connection error!"])
                return message, update_root

            with open(_update_zip_local, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    except Exception as err:
        odent_log([f"An unexpected error occurred: {err}"])
        message.extend(["Server connection error!"])
        return message, update_root

    try:
        with zipfile.ZipFile(_update_zip_local, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        src = [abspath(e) for e in os.listdir(temp_dir) if isdir(abspath(e))][0]
        update_root = join(temp_dir, OdentConstants.ADDON_UPDATE_NAME)
        os.rename(src, update_root)
        return message, update_root

    except zipfile.BadZipFile as zip_err:
        txt = "Error occurred while extracting the downloaded addon ZIP file"
        odent_log([f"{txt} : {zip_err}"])
        message.extend([txt])
        return message, update_root

    except Exception as err:
        odent_log(
            [f"Error occurred while extracting the downloaded addon ZIP file: {err}"]
        )
        message.extend(["Update error!"])
        return message, update_root


def write_json(Dict, outPath):
    jsonString = json.dumps(Dict, indent=4)
    with open(outPath, "w") as wf:
        wf.write(jsonString)


def open_json(jsonPath):
    with open(jsonPath, "r") as f:
        dataDict = json.load(f)
    return dataDict


def set_enum_items(items_list):
    return [(item, item, str(item)) for item in items_list]


def HuTo255(Hu, Wmin=OdentConstants.WMIN, Wmax=OdentConstants.WMAX):
    V255 = int(((Hu - Wmin) / (Wmax - Wmin)) * 255)
    return V255


def remove_handlers_by_names(handlers_names_list):
    depsgraph_update_post_handlers = bpy.app.handlers.depsgraph_update_post
    frame_change_post_handlers = bpy.app.handlers.frame_change_post

    # Remove old handlers :
    depsgraph_h = [
        h
        for h in bpy.app.handlers.depsgraph_update_post
        if h.__name__ in handlers_names_list
    ]
    frame_change_h = [
        h
        for h in bpy.app.handlers.frame_change_post
        if h.__name__ in handlers_names_list
    ]

    for h in depsgraph_h:
        bpy.app.handlers.depsgraph_update_post.remove(h)
    for h in frame_change_h:
        bpy.app.handlers.frame_change_post.remove(h)


def add_handlers_from_func_list(func_list):
    for h in func_list:
        bpy.app.handlers.depsgraph_update_post.append(h)
        bpy.app.handlers.frame_change_post.append(h)


def get_active_object_color():

    # Ensure there is an active object and it is selected
    obj = bpy.context.active_object
    is_valid_object = False

    if (
        not obj
        or obj not in bpy.context.selected_objects
        or obj.type not in {"MESH", "CURVE"}
    ):
        return is_valid_object, tuple()

    is_valid_object = True
    # Look for material
    mat = obj.active_material
    if not mat:
        return is_valid_object, tuple()

    # Get current shading mode
    shading = bpy.context.space_data.shading.type if bpy.context.space_data else None

    # If Solid mode → return viewport diffuse color
    if shading == "SOLID":
        if mat.diffuse_color:
            return is_valid_object, (mat, "diffuse_color")
        else:
            return is_valid_object, tuple()

    # If Material Preview or Rendered mode → try BSDF base color
    if shading in {"MATERIAL", "RENDERED"}:
        if mat.node_tree:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Base Color" in bsdf.inputs:
                return is_valid_object, (
                    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"],
                    "default_value",
                )
        else:
            return is_valid_object, tuple()
    return is_valid_object, tuple()


def get_icon_value(icon_name: str) -> int:
    icon_items = (
        bpy.types.UILayout.bl_rna.functions["prop"]
        .parameters["icon"]
        .enum_items.items()
    )
    icon_dict = {tup[1].identifier: tup[1].value for tup in icon_items}

    return icon_dict[icon_name]


class ODENT_GpuDrawText:
    """gpu draw text in active area 3d"""

    global DRAW_HANDLERS

    def __init__(
        self,
        message_list=[],
        remove_handlers=True,
        button=False,
        percentage=100,
        redraw_timer=True,
        rect_color=OdentConstants.INFO_FOOTER_RECT_COLOR,
        base_color=OdentConstants.INFO_FOOTER_BASE_COLOR,
        txt_color=OdentConstants.INFO_FOOTER_TEXT_COLOR,
        txt_size=OdentConstants.BLF_INFO.get("size"),
        btn_txt="OK",
        info_handler=None,
        sleep_time=0,
    ):

        self.message_list = message_list
        self.remove_handlers = remove_handlers
        self.button = button
        self.percentage = percentage
        self.redraw_timer = redraw_timer
        self.rect_color = rect_color
        self.base_color = base_color

        self.txt_color = txt_color
        self.txt_size = txt_size
        self.btn_txt = btn_txt
        self.info_handler = info_handler

        self.rect_height = 15
        self.line_height = 2
        self.offset_vertical = 35
        self.offset_horizontal = 50
        self.sleep_time = sleep_time

        if self.message_list:
            self._cancell_previous()
            self.gpu_info_footer()
            DRAW_HANDLERS.append(self.info_handler)
            self.redraw()
            if self.sleep_time != 0:
                sleep(self.sleep_time)
                self._cancell_previous()
                self.redraw()
        else:
            if self.remove_handlers:
                self._cancell_previous()
                self.redraw()

        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1)

    def redraw(self):
        try:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        except:
            pass

    def _cancell_previous(self):
        for _h in DRAW_HANDLERS:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(_h, "WINDOW")
                DRAW_HANDLERS.remove(_h)
            except Exception as er:
                odent_log([f"remove_handler error : {er}"])

    def gpu_info_footer(self):

        if self.percentage <= 0:
            self.percentage = 1
        if self.percentage > 100:
            self.percentage = 100

        def draw_callback_function():

            w = int(bpy.context.area.width * (self.percentage / 100))
            w_full = bpy.context.area.width

            self.draw_gpu_rect(
                self.offset_horizontal,
                self.offset_vertical,
                w - self.offset_horizontal,
                self.line_height,
                self.rect_color,
            )

            for i, txt in enumerate((reversed(self.message_list))):
                self.draw_gpu_rect(
                    self.offset_horizontal,
                    self.offset_vertical
                    + self.line_height
                    + 2
                    + (self.rect_height * i),
                    w_full - self.offset_horizontal,
                    self.rect_height,
                    self.base_color,
                )
                blf.position(
                    OdentConstants.BLF_INFO.get("fontid"),
                    self.offset_horizontal + 2,
                    self.offset_vertical
                    + self.line_height
                    + 2
                    + 2
                    + (self.rect_height * i),
                    0,
                )
                blf.size(
                    OdentConstants.BLF_INFO.get("fontid"), self.txt_size
                )  # 3.6 api blf.size(0, 40, 30) -> blf.size(fontid, size)
                r, g, b, a = self.txt_color
                blf.color(0, r, g, b, a)
                blf.draw(0, txt)

            if self.button:
                self.draw_gpu_rect(
                    w - 110, 2, 100, self.rect_height - 4, OdentColors.yellow
                )
                blf.position(0, w - 85, 10, 0)
                blf.size(
                    OdentConstants.BLF_INFO.get("fontid"), self.txt_size
                )  # 3.6 api blf.size(0, 40, 30) -> blf.size(fontid, size)
                r, g, b, a = self.txt_color
                blf.color(0, r, g, b, a)
                blf.draw(0, self.btn_txt)

        self.info_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_function, (), "WINDOW", "POST_PIXEL"
        )

        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def draw_gpu_rect(self, x, y, w, h, rect_color):

        vertices = ((x, y), (x, y + h), (x + w, y + h), (x + w, y))

        indices = ((0, 1, 2), (0, 2, 3))

        gpu.state.blend_set("ALPHA")
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")  # 3.6 api '2D_UNIFORM_COLOR'
        batch = batch_for_shader(shader, "TRIS", {"pos": vertices}, indices=indices)
        shader.bind()
        shader.uniform_float("color", rect_color)
        batch.draw(shader)


class ODENT_OT_RemoveInfoFooter(bpy.types.Operator):
    """Remove Info Footer"""

    bl_idname = "wm.odent_remove_info_footer"
    bl_label = "Hide Info"
    global DRAW_HANDLERS

    def execute(self, context):
        ODENT_GpuDrawText()
        return {"FINISHED"}
    

def load_odent_custom_icons():
    """Load all custom icons for the addon.
    
    Returns:
        PreviewCollection: A preview collection containing all loaded icons.
                         Returns None if no icons are found.
    """
    icons_dir = OdentConstants.ODENT_ICONS_PATH
    
    if not os.path.exists(icons_dir):
        odent_log([f"Icons directory not found: {icons_dir}"])
        return None
    
    # Get all image files
    valid_extensions = {".png", ".jpg", ".jpeg", ".tga", ".tif", ".bmp"}
    icon_files = [
        f for f in os.listdir(icons_dir) 
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not icon_files:
        odent_log([f"No valid image files found in: {icons_dir}"])
        return None
    
    # Create ONE preview collection for ALL icons
    preview_collection = bpy.utils.previews.new()
    
    # Load each icon into the collection
    for icon_file in icon_files:
        icon_name = os.path.splitext(icon_file)[0]
        icon_path = os.path.join(icons_dir, icon_file)
        
        try:
            # Load icon into the SAME collection
            preview_collection.load(icon_name, icon_path, 'IMAGE')
            odent_log([f"Successfully loaded icon: {icon_name}"])
        except Exception as e:
            odent_log([f"Failed to load icon '{icon_file}': {e}"])
    
    return preview_collection

def get_odent_icons():
    """Get or load the icon collection (singleton pattern)."""
    
    if OdentConstants.ODENT_ICONS is None:
        OdentConstants.ODENT_ICONS = load_odent_custom_icons()
    return OdentConstants.ODENT_ICONS

def register_odent_icons():
    """Register icons when addon enables."""
    get_odent_icons()

def unregister_odent_icons():
    """Clean up icons when addon disables (CRITICAL!)."""
    if OdentConstants.ODENT_ICONS is not None:
        bpy.utils.previews.remove(OdentConstants.ODENT_ICONS)
        OdentConstants.ODENT_ICONS = None

class ODENT_OT_create_desktop_shortcut(bpy.types.Operator):
    """Create ODENT shortcut on Desktop (auto‑generates ICO from PNG for Windows)."""
    bl_idname = "odent.create_desktop_shortcut"
    bl_label = "Create ODENT Desktop Shortcut"
    bl_description = "Creates ODENT shortcut on the Desktop"


    # -------------------------------------------------------------------------
    # Generate high-quality ICO from PNG (requires Pillow)
    # -------------------------------------------------------------------------
    def generate_ico(self, png_path, ico_path):
        try:
            from PIL import Image # type: ignore
        except ImportError:
            odent_log(["Pillow not installed, cannot generate ICO. Using default Blender icon."])
            return False

        try:
            img = Image.open(png_path)
            # Convert to RGBA if needed (preserve transparency)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            # Standard Windows sizes
            sizes = [16, 24, 32, 48, 64, 96, 128, 192, 256]
            # Filter sizes that fit within image dimensions
            max_size = max(img.size)
            valid_sizes = [s for s in sizes if s <= max_size]
            if not valid_sizes:
                valid_sizes = [max_size]
            # Save as ICO with multiple sizes
            img.save(ico_path, format='ICO', sizes=[(s, s) for s in valid_sizes])
            odent_log([f"Generated ICO with sizes {valid_sizes}: {ico_path}"])
            return True
        except Exception as e:
            odent_log([f"Failed to generate ICO: {e}"])
            return False

    # -------------------------------------------------------------------------
    # Windows shortcut (VBScript) – auto‑generates ICO from PNG
    # -------------------------------------------------------------------------
    def create_windows_shortcut(self, name, exec_path, png_path, desktop_path):
        shortcut_path = desktop_path / f"{name}.lnk"
        target = str(exec_path)
        
        icon_location = f"{target}, 0"   # set the fallback icon_location to Blender's own icon

        # Generate ICO from PNG (if PNG exists)
        ico_path = png_path.with_suffix(".ico")
        if not ico_path.exists():
            if png_path.exists():
                self.generate_ico(png_path, ico_path)
                
        #check if ICO exists
        if ico_path.exists():
            icon_location = str(ico_path)
        else:
            odent_log(["odent_logo.png not found, using Blender default icon"])


        def escape_vbs(s):
            return s.replace("\\", "\\\\").replace('"', '""')

        vbs_script = f'''
                            Set oShell = CreateObject("WScript.Shell")
                            Set oLink = oShell.CreateShortcut("{escape_vbs(str(shortcut_path))}")
                            oLink.TargetPath = "{escape_vbs(target)}"
                            oLink.IconLocation = "{escape_vbs(icon_location)}"
                            oLink.Save
                            '''
        fd, vbs_path = tempfile.mkstemp(suffix=".vbs", text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(vbs_script)
            subprocess.run(["cscript", "//Nologo", vbs_path], check=False, capture_output=True)
        finally:
            if os.path.exists(vbs_path):
                os.remove(vbs_path)
        return shortcut_path

    # -------------------------------------------------------------------------
    # Linux shortcut (.desktop) – uses PNG directly
    # -------------------------------------------------------------------------
    def create_linux_shortcut(self, name, exec_path, png_path, desktop_path):
        shortcut = desktop_path / f"{name}.desktop"
        content = [
            "[Desktop Entry]",
            "Version=1.0",
            "Type=Application",
            f"Name={name}",
            f"Exec={exec_path}",
            "Terminal=false",
            "Categories=Utility;",
        ]
        if png_path.exists():
            content.append(f"Icon={png_path}")
        shortcut.write_text("\n".join(content) + "\n")
        try:
            os.chmod(shortcut, 0o755)
            subprocess.run(["gio", "set", str(shortcut), "metadata::trusted", "true"], check=False)
        except Exception:
            pass
        return shortcut

    # -------------------------------------------------------------------------
    # macOS shortcut (symlink + AppleScript) – uses PNG
    # -------------------------------------------------------------------------
    def create_macos_shortcut(self, name, exec_path, png_path, desktop_path):
        link_path = desktop_path / name
        if link_path.exists():
            link_path.unlink()
        os.symlink(exec_path, link_path)
        if png_path.exists():
            try:
                subprocess.run([
                    "osascript", "-e",
                    f'''tell application "Finder" to set icon of (POSIX file "{link_path}") to POSIX file "{png_path}"'''
                ], check=False)
            except Exception:
                pass
        return link_path

    # -------------------------------------------------------------------------
    # Main execute
    # -------------------------------------------------------------------------
    def execute(self, context):
        name = "ODENT"   
        exec_path = bpy.app.binary_path
        icons_dir = pathlib.Path(OdentConstants.ODENT_ICONS_PATH)
        png_path = icons_dir / "odent_logo.png"
        desktop_path = pathlib.Path.home() / "Desktop"

        if not desktop_path.exists():
            self.report({'ERROR'}, f"Desktop folder not found: {desktop_path}")
            return {'CANCELLED'}

        _system = platform.system()
        try:
            if _system == "Windows":
                shortcut = self.create_windows_shortcut(name, exec_path, png_path, desktop_path)
            elif _system == "Linux":
                shortcut = self.create_linux_shortcut(name, exec_path, png_path, desktop_path)
            elif _system == "Darwin":
                shortcut = self.create_macos_shortcut(name, exec_path, png_path, desktop_path)
            else:
                message = f"Unsupported OS: {_system}"
                self.report({'ERROR'}, message)
                ODENT_GpuDrawText(message_list=message, sleep_time=2)
                return {'CANCELLED'}

            message = [f"Shortcut created!"]
            ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.green, sleep_time=2)
            return {'FINISHED'}
        except Exception as e:
            odent_log([f"Error creating shortcut: {e}"])
            message = ["Failed to create shortcut. Check console for details."]
            ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.red, sleep_time=2)
            
            return {'CANCELLED'}

@persistent
def _watch_transform_timer():
    """Internal timer function to detect movement/rotation and trigger callback."""
    global _last_loc, _last_rot, _callback

    obj = bpy.data.objects.get(WATCH_OBJECT_NAME)
    if not obj:
        print(f"⚠️ Object '{WATCH_OBJECT_NAME}' not found.")
        return 1.0  # retry later

    loc = obj.location.copy()
    rot = obj.rotation_euler.copy()

    moved = _last_loc is not None and (loc - _last_loc).length > 1e-5
    rotated = _last_rot is not None and (rot - _last_rot).length > 1e-5

    if (moved or rotated) and _callback:
        try:
            _callback(obj, moved=moved, rotated=rotated)
        except Exception as e:
            print(f"⚠️ Callback error: {e}")

    _last_loc = loc
    _last_rot = rot
    return CHECK_INTERVAL


def start_object_watcher(callback):
    """Start watching the object and call `callback(obj, moved, rotated)` when changed."""
    global _callback
    _callback = callback
    print(f"⏱️ Starting transform watcher for '{WATCH_OBJECT_NAME}'...")
    bpy.app.timers.register(_watch_transform_timer, persistent=True)


def stop_object_watcher():
    """Stop watching the object."""
    print("⏹️ Stopping transform watcher...")
    bpy.app.timers.unregister(_watch_transform_timer)


# --- Example usage ---
def on_myobject_change(obj, moved, rotated):
    """Your callback function."""
    print(f"🟢 {obj.name} changed:")
    if moved:
        print(f"   ↪ New location: {obj.location}")
    if rotated:
        print(f"   ↪ New rotation: {obj.rotation_euler}")


############################################

##########################################################################################
# Registration :
##########################################################################################

classes = [
    ODENT_OT_RemoveInfoFooter,
    ODENT_OT_SupportTelegram,
    ODENT_OT_AddOdentLibrary,
    ODENT_OT_AddAppTemplate,
    ODENT_OT_SetConfig,
    ODENT_OT_checkUpdate,
    ODENT_OT_MessageBox,
    ODENT_OT_OdentModulesPipInstaller,
    ODENT_OT_create_desktop_shortcut,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
