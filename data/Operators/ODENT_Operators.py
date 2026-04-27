import os
import shutil
import shlex
import math
import threading
from math import radians, pi, sqrt
import numpy as np  # type: ignore
from time import sleep, perf_counter as tpc
from queue import Queue
from os.path import join, dirname, exists, isdir, isfile, split
import subprocess
import time
import cv2 # type: ignore

import blf  # type: ignore

# Blender Imports :
import bpy  # type: ignore
from bpy.app.handlers import persistent  # type: ignore
from mathutils import Matrix, Vector, Euler, kdtree, bvhtree  # type: ignore
from bpy.props import (  # type: ignore
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)
import SimpleITK as sitk  # type: ignore

# import itk # type: ignore
import vtk  # type: ignore

from vtk import vtkCommand  # type: ignore

# Global Variables :

from .ODENT_Utils import (
    AppendObject,
    append_group_nodes,
    exclude_coll,
    finalize_geonodes_make_dup_colls,
    get_selected_odent_assets,
    lock_object,
    mesh_to_volume,
    get_slicer_areas,
    getLocalCollIndex,
    parse_dicom_series,
    unlock_object,
    view3d_utils,
    close_asset_browser,
    open_asset_browser,
    CtxOverride,
    context_override,
    count_non_manifold_verts,
    mesh_count,
    merge_verts,
    delete_loose,
    delete_interior_faces,
    fill_holes,
    ResizeImage,
    Scene_Settings,
    AddNode,
    MoveToCollection,
    hide_collection,
    hide_object,
    tempfile,
    sitkTovtk,
    vtk_MC_Func,
    vtkSmoothMesh,
    vtkWindowedSincPolyDataFilter,
    vtkMeshReduction,
    vtkTransformMesh,
    AddMarkupPoint,
    PointsToRefPlanes,
    CursorToVoxelPoint,
    CheckString,
    rotate_local,
    bmesh,
    VertexPaintCut,
    CuttingCurveAdd2,
    AddCurveSphere,
    ExtrudeCurvePointToCursor,
    DeleteLastCurvePoint,
    SplitSeparator,
    ShortestPath,
    ConnectPath,
    click_is_in_view3d,
    add_collection,
    CuttingCurveAdd,
    PointsToOcclusalPlane,
    load_matrix_from_file,
    VidDictFromPoints,
    RefPointsToTransformMatrix,
    AddRefPoint,
    KdIcpPairs,
    KdIcpPairsToTransformMatrix,
    Metaball_Splint,
    sitk_to_vtk_image,
    vtk_marching_cubes,
    create_blender_mesh_from_marching_cubes_fast,
    remove_pointer_lock,
    translate_local,
    Segmentator_Fast,
    get_odent_workspaces,
    get_fly_to_objects,
)
from ..utils import (
    # classes:
    OdentConstants,
    OdentColors,
    ODENT_GpuDrawText,
    # functions:
    AbsPath,
    RelPath,
    clear_terminal,
    get_incremental_idx,
    get_incremental_name,
    odent_log,
    is_linux,
    is_wine_installed,
    install_wine,
    set_enum_items,
    remove_handlers_by_names,
    add_handlers_from_func_list,
    HuTo255,
    TimerLogger,
    get_image_diagonal_lenght,
    get_unique_name,
    remove_handlers_by_names,
    add_handlers_from_func_list,
    activate_obj,
    set_socket_value_geonodes,
    
    
)

MC = {}
IMAGE3D = None
SLICES_TXT_HANDLER = []
message_queue = Queue()
FLY_IMPLANT_INDEX = None
RESTART = False
_LAST_SLICES_POINTER_LOC = None
_LAST_SLICES_POINTER_ROT = None
FORCE_SLICES_UPDATE = False

# using timer to update slices
SLICES_DIRTY = False
SLICES_UPDATING = False
SLICES_POINTER = None

ProgEvent = vtkCommand.ProgressEvent


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result

    return wrapper


def draw_slices_text_2d():
    text_color_rgba = [0.8, 0.6, 0.0, 1.0]
    text_Thikness = (40, 30)

    def draw_callback_function():
        data = bpy.context.scene.get("odent_implant_data")
        if data:
            loc, txt = eval(data)
            region = bpy.context.region
            region_3d = bpy.context.space_data.region_3d

            locOnScreen = view3d_utils.location_3d_to_region_2d(region, region_3d, loc)
            blf.position(0, locOnScreen[0] + 1, locOnScreen[1] - 1, 0)
            blf.size(0, text_Thikness[0], text_Thikness[1])
            r, g, b, a = text_color_rgba
            blf.color(0, r, g, b, a)
            blf.draw(0, txt)

    slices_text_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback_function, (), "WINDOW", "POST_PIXEL"
    )
    return slices_text_handler


def update_slices_txt(remove_handlers=True):
    global SLICES_TXT_HANDLER

    if remove_handlers:
        for _h in SLICES_TXT_HANDLER:
            bpy.types.SpaceView3D.draw_handler_remove(_h, "WINDOW")
        SLICES_TXT_HANDLER = []

    slices_text_handler = draw_slices_text_2d()
    SLICES_TXT_HANDLER.append(slices_text_handler)
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


class ODENT_OT_AssetBrowserToggle(bpy.types.Operator):
    """Split area 3d and load asset browser"""

    bl_idname = "wm.odent_asset_browser_toggle"
    bl_label = "Odent Library"
    bl_options = {"REGISTER", "UNDO"}
    
    can_update = False

    def defer(self):
        params = self.asset_browser_space.params
        if not params:
            return 0
        try:
            params.asset_library_reference = OdentConstants.ODENT_LIB_NAME

        except TypeError:
            # If the reference doesn't exist.
            params.asset_library_ref = "LOCAL"

        # params.import_type = 'APPEND'
        self.can_update = True
        return None

    def modal(self, context, event):
        if self.can_update:
            return {"FINISHED"}
        return {"PASS_THROUGH"}

    def execute(self, context):

        if not context.workspace.name == "Odent Main":
            txt = ["Cancelled : Please ensure you are in Odent Main workspace !"]
            ODENT_GpuDrawText(
                message_list=txt, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}

        _close = close_asset_browser(context)
        if not _close:
            self.asset_browser_area, self.asset_browser_space = open_asset_browser()
            bpy.app.timers.register(self.defer)
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        return {"FINISHED"}


class ODENT_OT_ImportMesh(bpy.types.Operator):
    """Import mesh Operator"""

    bl_idname = "wm.odent_import_mesh"
    bl_label = " Import Mesh"
    bl_options = {"REGISTER", "UNDO"}

    f_extention: EnumProperty(
        items=set_enum_items(["STL", "OBJ", "PLY"]),
        name="Scan Type",
        description="Scan Extention",
    )  # type: ignore

    def execute(self, context):

        if self.f_extention == "STL":
            bpy.ops.wm.stl_import("INVOKE_DEFAULT")
        if self.f_extention == "OBJ":
            bpy.ops.wm.obj_import("INVOKE_DEFAULT")
        if self.f_extention == "PLY":
            bpy.ops.wm.ply_import("INVOKE_DEFAULT")

        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_ExportMesh(bpy.types.Operator):
    """Export Mesh Operator"""

    bl_idname = "wm.odent_export_mesh"
    bl_label = " Export Mesh"

    f_extention: EnumProperty(
        items=set_enum_items(["STL", "OBJ", "PLY"]),
        name="Scan Type",
        description="Scan Extention",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if context.object.type != "MESH":
            return False
        if not context.object.select_get():
            return False
        return True

    def execute(self, context):
        target = context.object
        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)

        if self.f_extention == "STL":
            bpy.ops.wm.stl_export("INVOKE_DEFAULT", export_selected_objects=True)
        if self.f_extention == "OBJ":
            bpy.ops.wm.obj_export("INVOKE_DEFAULT", export_selected_objects=True)
        if self.f_extention == "PLY":
            bpy.ops.wm.ply_export("INVOKE_DEFAULT", export_selected_objects=True)

        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class ODENT_OT_AlignToActive(bpy.types.Operator):
    """Align object to active object"""

    bl_idname = "wm.odent_align_to_active"
    bl_label = "Align to active"
    bl_options = {"REGISTER", "UNDO"}
    
    invert_z: BoolProperty(
        name="Invert Z", description="Invert Z axis", default=False
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.select_get()
            and len(context.selected_objects) == 2
        )

    def execute(self, context):
        self.selected = context.selected_objects
        self.active_object = context.object
        self.other_object = [
            obj for obj in self.selected if not obj is self.active_object
        ][0]
        bpy.ops.object.select_all(action="DESELECT")
        # Decompose obj1 matrix_world
        loc, rot, _ = self.active_object.matrix_world.decompose()
        # Get obj2's current scale
        _, _, scale = self.other_object.matrix_world.decompose()
        # Recompose new matrix for obj2 with same scale, new location and rotation
        self.other_object.matrix_world = Matrix.LocRotScale(loc, rot, scale)

        # self.invert_z = False
        if (
            self.active_object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.ODENT_IMPLANT_TYPE
        ):
            bpy.ops.object.select_all(action="DESELECT")
            tooth_number = self.active_object[
                OdentConstants.ODENT_IMPLANT_REMOVE_CODE_TAG
            ]
            if tooth_number < 31:
                self.invert_z = True
            else:
                self.invert_z = False

        if self.invert_z:
            self.other_object.rotation_euler.rotate_axis("X", math.pi)

        return {"FINISHED"}


class ODENT_OT_LockObjects(bpy.types.Operator):
    """Lock objects transform"""

    bl_idname = "wm.odent_lock_objects"
    bl_label = "LOCK OBJECT"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.object and context.object.select_get():
            locks = []
            for obj in context.selected_objects:
                locks.extend(list(obj.lock_location))
                locks.extend(list(obj.lock_rotation))
                locks.extend(list(obj.lock_scale))
            if not all(locks):
                return True
        return False

    def execute(self, context):
        for obj in context.selected_objects:
            obj.lock_location = (True, True, True)
            obj.lock_rotation = (True, True, True)
            obj.lock_scale = (True, True, True)
        return {"FINISHED"}


class ODENT_OT_UnlockObjects(bpy.types.Operator):
    """Unock objects transform"""

    bl_idname = "wm.odent_unlock_objects"
    bl_label = "UNLOCK OBJECT"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.object and context.object.select_get():
            locks = []
            for obj in context.selected_objects:
                locks.extend(list(obj.lock_location))
                locks.extend(list(obj.lock_rotation))
                locks.extend(list(obj.lock_scale))
            if any(locks):
                return True
        return False

    def execute(self, context):
        for obj in context.selected_objects:
            obj.lock_location = (False, False, False)
            obj.lock_rotation = (False, False, False)
            obj.lock_scale = (False, False, False)
        return {"FINISHED"}


class ODENT_OT_add_3d_text(bpy.types.Operator):
    """add 3D text"""

    bl_label = "Add 3D Text"
    bl_idname = "wm.odent_add_3d_text"
    bl_options = {"REGISTER", "UNDO"}
    
    
    text_color = [0.0, 0.0, 1.0, 1.0]
    text = "ODent"
    font_size = 3
    text_mode: EnumProperty(
        items=set_enum_items(["Embossed", "Engraved"]),
        name="Text Mode",
        default="Embossed",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if not (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        ):
            return False
        return True

    def invoke(self, context, event):
        self.target = context.object
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        ODENT_Props = context.scene.ODENT_Props

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.text_add(enter_editmode=False, align="CURSOR")
        self.text_ob = context.object
        self.text_ob[OdentConstants.ODENT_TYPE_TAG] = "odent_text"
        self.text_ob.data.body = ODENT_Props.text
        self.text_ob.name = "Text_" + ODENT_Props.text

        self.text_ob.data.align_x = "CENTER"
        self.text_ob.data.align_y = "CENTER"
        self.text_ob.data.size = self.font_size
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.view_axis(type="TOP", align_active=True)

        # change curve settings:
        self.text_ob.data.extrude = 0.5
        self.text_ob.data.bevel_depth = 0.02
        self.text_ob.data.bevel_resolution = 6

        # add SHRINKWRAP modifier :
        shrinkwrap_modif = self.text_ob.modifiers.new("SHRINKWRAP", "SHRINKWRAP")
        shrinkwrap_modif.use_apply_on_spline = True
        shrinkwrap_modif.wrap_method = "PROJECT"
        shrinkwrap_modif.offset = 0
        shrinkwrap_modif.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap_modif.cull_face = "OFF"
        shrinkwrap_modif.use_negative_direction = True
        shrinkwrap_modif.use_positive_direction = True
        shrinkwrap_modif.use_project_z = True
        shrinkwrap_modif.target = self.target

        mat = bpy.data.materials.get("odent_text_mat") or bpy.data.materials.new(
            "odent_text_mat"
        )
        mat.diffuse_color = self.text_color
        mat.roughness = 0.6
        self.text_ob.active_material = mat

        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE_NEAREST"}
        bpy.context.scene.tool_settings.use_snap_align_rotation = True
        bpy.context.scene.tool_settings.use_snap_rotate = True

        ODENT_GpuDrawText(["Press ESC to cancel, ENTER to confirm"])

        # run modal
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not event.type in {"ESC", "RET"}:
            return {"PASS_THROUGH"}
        elif event.type in {"ESC"}:
            try:
                bpy.data.objects.remove(self.text_ob)
            except:
                pass
            ODENT_GpuDrawText(
                message_list=["Cancelled ./"], rect_color=OdentColors.red, sleep_time=1
            )
            return {"CANCELLED"}

        elif event.type in {"RET"} and event.value == "PRESS":
            ODENT_GpuDrawText(["3D Text processing..."])
            self.text_ob.select_set(True)
            self.target.select_set(True)
            remesh_modif = self.text_ob.modifiers.new("REMESH", "REMESH")
            remesh_modif.voxel_size = 0.05
            with context.temp_override(active_object=self.text_ob):
                bpy.ops.object.convert(target="MESH")
                bpy.ops.object.mode_set(mode="OBJECT")

            if self.text_mode == "Embossed":
                self.embosse_text(context)
            else:
                self.engrave_text(context)

            bpy.context.scene.tool_settings.use_snap = False
            bpy.ops.object.select_all(action="DESELECT")
            ODENT_GpuDrawText(
                message_list=["Finished."], rect_color=OdentColors.green, sleep_time=1
            )

            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def engrave_text(self, context):

        bpy.ops.object.material_slot_remove_all()

        difference_modif = self.target.modifiers.new("DIFFERENCE", "BOOLEAN")
        difference_modif.object = self.text_ob
        difference_modif.operation = "DIFFERENCE"
        with context.temp_override(active_object=self.target):
            bpy.ops.object.convert(target="MESH")

        bpy.data.objects.remove(self.text_ob)
        context.scene.ODENT_Props.text = "ODent"

    def embosse_text(self, context):

        bpy.ops.object.material_slot_remove_all()

        union_modif = self.target.modifiers.new("DIFFERENCE", "BOOLEAN")
        union_modif.object = self.text_ob
        union_modif.operation = "UNION"
        with context.temp_override(active_object=self.target):
            bpy.ops.object.convert(target="MESH")

        bpy.data.objects.remove(self.text_ob)
        context.scene.ODENT_Props.text = "ODent"


class ODENT_OT_Text3d(bpy.types.Operator):
    """knife project 3d text on mesh"""

    bl_idname = "wm.odent_text3d"
    bl_label = "Text 3D"
    bl_options = {"REGISTER", "UNDO"}

    text_color = [0.0, 0.0, 1.0, 1.0]
    text_body: StringProperty(
        name="Text",
        description="text content",
        default="Odent",
    )  # type: ignore
    font_size = 3.5
    space_character = 1.3
    text_offset = 5

    text_mode: EnumProperty(
        items=set_enum_items(["Embossed", "Engraved"]),
        name="Text Mode",
        default="Embossed",
    )  # type: ignore

    extrusion_value: FloatProperty(
        name="Text depth",
        description="Value for extrusion along the normal",
        default=0.5,
        min=0.0,
        max=10.0,
        subtype="DISTANCE",
    )  # type: ignore

    @classmethod
    def poll(cls, context):

        target = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        return target

    def add_text(self, context):

        bpy.ops.object.text_add(enter_editmode=False, align="CURSOR")
        self.text_obj = context.object
        text_obj_idx = get_incremental_idx(
            data=bpy.data.objects, odent_type=OdentConstants.ODENT_TEXT_3D_TYPE
        )
        text_obj_name = get_incremental_name("Text3d", text_obj_idx)
        self.text_obj.name = text_obj_name
        self.text_obj[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.ODENT_TEXT_3D_TYPE

        self.text_obj.data.body = self.text_body

        self.text_obj.data.size = self.font_size
        self.text_obj.data.space_character = self.space_character
        self.text_obj.data.align_x = "CENTER"

        mat_name = OdentConstants.ODENT_TEXT_3D_MAT_NAME
        diffuse_color = OdentConstants.ODENT_TEXT_3D_MAT_DIFFUSE_COLOR
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
        mat.diffuse_color = diffuse_color
        self.text_obj.active_material = mat

    def set_text_position(self, context):
        cursor = context.scene.cursor
        self.text_obj.matrix_world[:3] = cursor.matrix[:3]
        translate_local(self.text_obj, self.text_offset, axis="Z")
        self.Z = Vector(cursor.matrix.col[2][:3]).normalized()
        override, area3D, space3D, region3D = context_override(context)
        r3d = space3D.region_3d
        self.view_mtx = r3d.view_matrix

        return

    def finalize(self, context):

        if not context.object:
            context.view_layer.objects.active = self.target

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        self.text_obj.select_set(True)
        context.view_layer.objects.active = self.target

        override, area3D, space3D, region3D = context_override(context)
        r3d = space3D.region_3d
        r3d.view_matrix = self.view_mtx
        bpy.ops.object.mode_set(mode="EDIT")
        with bpy.context.temp_override(**override):
            bpy.ops.mesh.knife_project()

        if self.text_mode == "Embossed":
            self.extrusion_value = abs(self.extrusion_value)
        else:
            self.extrusion_value = -abs(self.extrusion_value)

        cursor = context.scene.cursor
        extrusion_vector = (
            self.target.matrix_world.inverted()
            @ Vector(cursor.matrix.col[2][:3]).normalized()
            * self.extrusion_value
        )

        bpy.ops.mesh.extrude_region_move()

        bpy.ops.transform.translate(value=extrusion_vector)
        print(f"extrusion vector : {extrusion_vector}")

        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            bpy.data.objects.remove(self.text_obj)
        except:
            pass

        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        context.view_layer.objects.active = self.target

        return

    def invoke(self, context, event):
        self.target = context.object
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        context.view_layer.objects.active = self.target

        override, area3D, space3D, region3D = context_override(context)
        with bpy.context.temp_override(**override):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

        txt = [
            "Mouse left : Set text location",
            "TAB : Edit text",
            "ESC : Cancell operation",
            "ENTER : Finalise",
        ]
        ODENT_GpuDrawText(txt)

        self.text_added = False

        wm = context.window_manager
        wm.modal_handler_add(self)

        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not event.type in {"RET", "ESC", "LEFTMOUSE"}:
            return {"PASS_THROUGH"}

        elif event.type in {"ESC"}:
            try:
                bpy.data.objects.remove(self.text_obj)
            except:
                pass
            override, area3D, space3D, region3D = context_override(context)
            with bpy.context.temp_override(**override):
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
            ODENT_GpuDrawText(
                message_list=["Cancelled."], rect_color=OdentColors.red, sleep_time=1
            )
            return {"CANCELLED"}
        elif event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                return {"PASS_THROUGH"}
            elif event.value == "RELEASE":
                if not self.text_added:
                    self.add_text(context)
                    self.text_added = True

                if context.object and context.mode == "OBJECT":
                    self.set_text_position(context)
                    return {"RUNNING_MODAL"}
                else:
                    return {"PASS_THROUGH"}

        elif event.type == "RET" and event.value == "PRESS":
            if context.object and context.mode == "OBJECT":
                ODENT_GpuDrawText(["3D Text processing..."])
                self.finalize(context)
                override, area3D, space3D, region3D = context_override(context)
                with bpy.context.temp_override(**override):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                ODENT_GpuDrawText(
                    message_list=["Finished."],
                    rect_color=OdentColors.green,
                    sleep_time=1,
                )
                return {"FINISHED"}
            else:
                return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


class ODENT_OT_OpenManual(bpy.types.Operator):
    """Open ODENT Manual"""

    bl_idname = "wm.odent_open_manual"
    bl_label = "User Manual"

    def execute(self, context):

        Manual_Path = join(
            OdentConstants.ADDON_DIR, "Resources", "ODENT User Manual.pdf"
        )
        if exists(Manual_Path):
            os.startfile(Manual_Path)
            return {"FINISHED"}
        else:
            message = [" Sorry Manual not found."]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}


class ODENT_OT_ReloadStartup(bpy.types.Operator):
    """reload startup file"""

    bl_idname = "wm.odent_reload_startup"
    bl_label = "Reload Startup"
    save_alert = [
        "Current project is not saved !",
        "Please save it before reloading the startup file.",
    ]

    def draw(self, context):
        layout = self.layout
        layout.alignment = "EXPAND"
        box = layout.box()
        box.alert = True
        for txt in self.save_alert:
            box.label(text=txt)

        row = layout.row(align=True)
        row.alignment = "EXPAND"
        if bpy.data.is_dirty:
            row.alert = True

        row.operator("wm.save_mainfile", text="Save Project")
        row.operator("wm.save_as_mainfile", text="Save Project As...")

    def execute(self, context):
        bpy.ops.wm.read_homefile("INVOKE_DEFAULT")
        return {"FINISHED"}

    def invoke(self, context, event):
        if bpy.data.is_dirty or not bpy.data.filepath:
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=600)
        else:
            return self.execute(context)


####### Dima
class ODENT_OT_NewProject(bpy.types.Operator):
    """Save current project if not saved"""

    bl_idname = "wm.odent_new_project"
    bl_label = "New Project"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    use_folder_name: bpy.props.BoolProperty(
        name="Project as folder name",
        description="Use folder name as project filename",
        default=False,
    )

    _manual_name: str = "Untitled.blend"
    _folder_mode_active: bool = False
    can_resume: bpy.props.BoolProperty(default=False)

    def invoke(self, context, event):
        base_dir = (
            os.path.dirname(bpy.data.filepath)
            if bpy.data.filepath
            else bpy.path.abspath("//")
        )
        self.filepath = os.path.join(base_dir, self._manual_name)
        self._folder_mode_active = False
        self.can_resume = False

        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        self.layout.prop(self, "use_folder_name")

    def _normalize(self, name: str) -> str:
        return name if name.lower().endswith(".blend") else name + ".blend"

    def _get_folder_name(self, dir_path: str) -> str:
        return os.path.basename(dir_path) or "Untitled"

    def check(self, context):
        dir_path = os.path.normpath(os.path.dirname(self.filepath) or "//")
        current_name = self._normalize(os.path.basename(self.filepath))
        folder_name = self._get_folder_name(dir_path)
        auto_name = f"{folder_name}.blend"

        changed = False

        # Folder Name (checkbox ON)
        if self.use_folder_name:
            if not self._folder_mode_active:
                # First activation - Folder Name
                self.filepath = os.path.join(dir_path, auto_name)
                self._folder_mode_active = True
                changed = True
            elif current_name != auto_name:
                # User chaged name - refresh manual name
                self.use_folder_name = False
                self._manual_name = current_name
                self._folder_mode_active = False
                changed = True

        # Manual name (checkbox OFF)
        else:
            if self._folder_mode_active:
                # Inactivation checkbox Folder name - refresh manual name
                self.filepath = os.path.join(dir_path, self._manual_name)
                self._folder_mode_active = False
                changed = True
            elif current_name != self._manual_name:
                # User chaged name - refresh manual name
                self._manual_name = current_name
                changed = True

        return changed

    def execute(self, context):
        self.filepath = self._normalize(self.filepath)
        try:
            bpy.ops.wm.save_as_mainfile(filepath=self.filepath)
            self.report({"INFO"}, f"Saved as: {self.filepath}")

            # can_resume True modal
            self.can_resume = True
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

    def modal(self, context, event):
        if self.can_resume:
            _props = bpy.context.scene.ODENT_Props
            project_dir = os.path.dirname(bpy.path.abspath(bpy.data.filepath))

            try:
                _props.UserProjectDir = bpy.path.relpath(project_dir)
            except:
                pass
            try:
                _props.UserDcmDir = bpy.path.relpath(project_dir)
            except:
                pass
            try:
                _props.UserImageFile = bpy.path.relpath(project_dir)
            except:
                pass

            return {"FINISHED"}
        return {"PASS_THROUGH"}

    ####### Dima end


class ODENT_OT_DrawTester(bpy.types.Operator):
    """test for gpu info footer draw"""

    bl_idname = "wm.odent_draw_tester"
    bl_label = "Draw Test"

    def execute(self, context):

        message1 = ["this is a 1 line message."]
        message2 = ["this is 1rst line,", "and this is the 2nd line."]
        message3 = [
            "this is 1rst line,",
            "this is the 2nd line,",
            "and this is the 3rd line.",
        ]
        for i, message in enumerate([message1, message2, message3]):
            ODENT_GpuDrawText(
                message_list=message,
                rect_color=OdentColors.default,
                sleep_time=3,
                percentage=20 + i * 40,
            )
        return {"FINISHED"}


class ODENT_OT_Dicom_Reader(bpy.types.Operator):
    """Dicom Reader"""

    bl_idname = "wm.odent_dicom_reader"
    bl_label = "Dicom Read"
    bl_options = {"REGISTER", "UNDO"}

    global MC, IMAGE3D
    global VIEW_MATRIX
    add_slices_bool: BoolProperty(
        name="Add slices",
        default=False,
        description="Add slices to existing image",
    )  # type: ignore
    project_name: StringProperty(default="")  # type: ignore
    pointCloudThresholdMin: IntProperty(
        name="PCD threshold",
        description=f"value between {OdentConstants.WMIN} and {OdentConstants.WMAX} for point cloud points extraction",
        default=450,
        min=OdentConstants.WMIN,
        max=OdentConstants.WMAX,
        step=1,
    )  # type: ignore

    center_dicom: BoolProperty(
        name="Center Dicom",
        description="Center Dicom in the scene",
        default=False,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        _props = context.scene.ODENT_Props
        if _props.dicomDataType == "DICOM Series":
            datapath = AbsPath(_props.UserDcmDir)
            is_valid = exists(datapath) and isdir(datapath) and os.listdir(datapath)
        else:
            datapath = AbsPath(_props.UserImageFile)
            is_valid = exists(datapath) and isfile(datapath)

        return is_valid

    def draw(self, context):
        layout = self.layout
        g = layout.grid_flow(columns=1, align=True)
        if self.dicom_data_type == "file_series":
            g.prop(self._props, "Dicom_Series")

        g = layout.grid_flow(columns=2, align=True)
        g.prop(self._props, "scan_resolution")
        # g.prop(self, "center_dicom")
        if self._props.visualisation_mode == OdentConstants.VISUALISATION_MODE_PCD:
            g.prop(self, "pointCloudThresholdMin")

    def get_user_paths(self, context):

        user_proj_path = AbsPath(self._props.UserProjectDir)
        if self._props.dicomDataType == "DICOM Series":
            dicom_data_type = "file_series"
            dicom_data_path = self._props.UserDcmDir
        else:
            dicom_data_type = "single_file"
            dicom_data_path = self._props.UserImageFile

        return user_proj_path, dicom_data_type, dicom_data_path

    def get_dicom_cach_dict(self, context):

        all_dicom_cach_dictionary = eval(self._props.dicom_dictionary)
        current_dicom_cach_dictionary = (
            all_dicom_cach_dictionary.get(self.datapath) or {}
        )

        return all_dicom_cach_dictionary, current_dicom_cach_dictionary

    def update_dicom_dictionary_prop(self):
        self.all_dicom_cach_dictionary[self.uid] = self.current_dicom_cach_dictionary
        self._props.dicom_dictionary = str(self.all_dicom_cach_dictionary)

    def get_sitk_image_from_singlFile(self, context):

        self.sitk_image = None
        error_message = []
        dicom_dictionary = {}
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(AbsPath(self.dicom_data_path))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            self.sitk_image = reader.Execute()
            sp_max = round(max(self.sitk_image.GetSpacing()[:]), 3)
            depth = self.sitk_image.GetDepth()

            if depth <= 2:
                error_message = ["dicom file is not a 3D image"]
                return error_message, dicom_dictionary

            is_odent_image = False
            is_other_valid_image = False

            if self.sitk_image.HasMetaDataKey(OdentConstants.ODENT_IMAGE_METADATA_KEY):
                print(
                    f"odent image pixel id type : {self.sitk_image.GetPixelIDTypeAsString()}"
                )
                image_type = self.sitk_image.GetMetaData(
                    OdentConstants.ODENT_IMAGE_METADATA_KEY
                )  # Image Type
                is_odent_image = image_type == OdentConstants.ODENT_IMAGE_METADATA_TAG
                image_uid = self.sitk_image.GetMetaData(
                    OdentConstants.ODENT_IMAGE_METADATA_UID_KEY
                )

            elif (
                self.dicom_data_path.endswith("_Image3D255.nrrd")
                and self.sitk_image.GetPixelIDTypeAsString() == "8-bit unsigned integer"
            ):
                is_odent_image = True
                image_uid = "Unknown_uid"
                for tag in ["0008|0018", "0020|000d", "0020|000e"]:
                    if self.sitk_image.HasMetaDataKey(tag):
                        image_uid = self.sitk_image.GetMetaData(tag)
                        break

            elif self.sitk_image.GetPixelIDTypeAsString() in [
                "32-bit signed integer",
                "16-bit signed integer",
            ]:
                is_other_valid_image = True
                image_uid = "Unknown_uid"
                for tag in ["0008|0018", "0020|000d", "0020|000e"]:
                    if self.sitk_image.HasMetaDataKey(tag):
                        image_uid = self.sitk_image.GetMetaData(tag)
                        break

            if not (is_odent_image or is_other_valid_image):
                error_message = ["can not read image file"]
                return error_message, dicom_dictionary

            dicom_dictionary.update(
                {
                    "spacing": sp_max,
                    "file": self.dicom_data_path,
                    "uid": image_uid,
                    "is_odent_image": is_odent_image,
                }
            )
            self._props.scan_resolution = sp_max

        except Exception as e:
            log_txt = f"context : read_image | error :\n {e}"
            odent_log([log_txt])
            error_message = ["can not read image file"]

        return error_message, dicom_dictionary

    def get_sitk_image_from_fileSeries(self, s_id, dicom_data_path):
        error_message = []
        self.sitk_image = None
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(AbsPath(dicom_data_path))
            files = reader.GetGDCMSeriesFileNames(AbsPath(dicom_data_path), s_id)
            reader.SetFileNames(files)
            reader.LoadPrivateTagsOn()
            self.sitk_image = reader.Execute()
            pixel_type = self.sitk_image.GetPixelIDTypeAsString()

            if pixel_type not in ["32-bit signed integer", "16-bit signed integer"]:
                #######################################################################
                log_txt = f"insupported 3D image pixel type\npixel type : {pixel_type}"
                odent_log([log_txt])
                ######################################################################################
                error_message = [f"insupported 3D image pixel type!"]
                return error_message
            if not self.sitk_image:
                #######################################################################
                log_txt = "sitk image is None"
                odent_log([log_txt])
                ######################################################################################
                error_message = ["can not read dicom image series"]
                return error_message

            self.current_dicom_cach_dictionary["count"] = len(files)
            # self.fix_sitk_image()

        except Exception as e:
            log_txt = f"context : get_sitk_image_from_fileSeries | error :\n {e}"
            odent_log([log_txt])
            error_message = ["can not read dicom image series"]

        return error_message

    def invoke(self, context, event):

        if not bpy.data.filepath and not self.is_new_project:
            message = ["Please save your project before reading DICOM data."]
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}

        self._props = context.scene.ODENT_Props
        self._props.UserProjectDir = dirname(AbsPath(bpy.data.filepath))
        self.sitk_image = None
        self.windows = context.window_manager.windows

        # info message : Checking dicom data 10% ...
        pourcentage = 10
        message = [f"Checking dicom data {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)
        #################################################################

        self.user_proj_path, self.dicom_data_type, self.dicom_data_path = (
            self.get_user_paths(context)
        )
        self.all_dicom_cach_dictionary = eval(self._props.dicom_dictionary)
        self.current_dicom_cach_dictionary = {}

        if self.dicom_data_type == "file_series":
            # info message : Reading dicom series please wait 40% ...
            pourcentage = 40
            message = [f"Reading dicom series please wait {pourcentage}% ..."]
            ODENT_GpuDrawText(message_list=message, percentage=pourcentage)
            #################################################################

            error_message, self.dicom_series_dictionary = parse_dicom_series(
                self, AbsPath(self.dicom_data_path)
            )

            if error_message:
                ODENT_GpuDrawText(
                    message_list=error_message, rect_color=OdentColors.red, sleep_time=2
                )
                return {"CANCELLED"}

            self._props.dicom_series_dictionary = str(self.dicom_series_dictionary)
            self._props.Dicom_Series = list(self.dicom_series_dictionary.keys())[0]

            if len(self.dicom_series_dictionary) > 1:
                message = [
                    "please select dicom series, optional : set resolution (recommanded = 0.3), and press OK ..."
                ]
            else:
                message = [
                    "optional : set resolution (recommanded = 0.3) and press OK ..."
                ]
            ODENT_GpuDrawText(message_list=message)

        elif self.dicom_data_type == "single_file":

            pourcentage = 40
            message = [f"Reading dicom image please wait {pourcentage}% ..."]
            ODENT_GpuDrawText(message_list=message, percentage=pourcentage)

            error_message, self.current_dicom_cach_dictionary = (
                self.get_sitk_image_from_singlFile(context)
            )
            if error_message:
                ODENT_GpuDrawText(
                    message_list=error_message, rect_color=OdentColors.red, sleep_time=2
                )
                return {"CANCELLED"}
            self._props.current_dicom_dictionary = str(
                self.current_dicom_cach_dictionary
            )
            # self.update_dicom_dictionary_prop()

            message = ["optional : set resolution (recommanded = 0.3) and press OK ..."]
            ODENT_GpuDrawText(message_list=message)

        wm = context.window_manager

        return wm.invoke_props_dialog(self, width=600)

    def fix_sitk_image(self):
        self.sitk_image = sitk.Cast(
            sitk.IntensityWindowing(
                self.sitk_image,
                windowMinimum=OdentConstants.WMIN,
                windowMaximum=OdentConstants.WMAX,
                outputMinimum=0,
                outputMaximum=255,
            ),
            sitk.sitkUInt8,
        )
        if self.center_dicom:
            # center image and resample:
            image_center = self.sitk_image.TransformContinuousIndexToPhysicalPoint(
                np.array(self.sitk_image.GetSize()) / 2.0
            )
            new_transform = sitk.AffineTransform(3)
            new_transform.SetMatrix(np.array(self.sitk_image.GetDirection()).ravel())
            new_transform.SetCenter(image_center)
            new_transform = new_transform.GetInverse()
            self.sitk_image = sitk.Resample(self.sitk_image, new_transform)

            self.sitk_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
            self.sitk_image.SetOrigin([0, 0, 0])

    def get_main_image(self, context):

        global IMAGE3D
        error_message = []

        # ensure correct user_spacing float number :
        self.user_spacing = round(self._props.scan_resolution, 3)

        if self.dicom_data_type == "single_file":

            # everything is already checked in invoke method self.sitk_image is ok
            self.uid = self.current_dicom_cach_dictionary.get("uid")

            # ensure image is isotropic (same spacing in x,y and z),
            # resize if the user changed the native image resolution (spacing)
            self.sitk_image, new_size, new_spacing = ResizeImage(
                self.sitk_image, self.user_spacing
            )

            # if it is not an Odent image :
            #   need to be casted :
            #   (narrow the intensities window <wmin..wmax> and convert to uint8 <0...255>)
            is_odent_image = self.current_dicom_cach_dictionary.get("is_odent_image")
            if not is_odent_image:
                self.sitk_image = sitk.Cast(
                    sitk.IntensityWindowing(
                        self.sitk_image,
                        windowMinimum=OdentConstants.WMIN,
                        windowMaximum=OdentConstants.WMAX,
                        outputMinimum=0,
                        outputMaximum=255,
                    ),
                    sitk.sitkUInt8,
                )

        elif self.dicom_data_type == "file_series":
            print("get_main_image from file series")

            # get the user selected series id in case multiple series or the default one
            self.current_dicom_cach_dictionary = self.dicom_series_dictionary[
                self._props.Dicom_Series
            ]
            self.uid = self.current_dicom_cach_dictionary["s_id"]

            # get the main sitk image :
            error_message = self.get_sitk_image_from_fileSeries(
                self.uid, self.dicom_data_path
            )
            if error_message:
                return error_message

            # ensure image is isotropic (same spacing in x,y and z),
            # resize if the user changed the native image resolution (spacing)
            self.sitk_image, new_size, new_spacing = ResizeImage(
                self.sitk_image, self.user_spacing
            )

            # sitk_image need to be casted :
            # (narrow the intensities window <wmin..wmax> and convert to uint8 <0...255>)
            self.sitk_image = sitk.Cast(
                sitk.IntensityWindowing(
                    self.sitk_image,
                    windowMinimum=OdentConstants.WMIN,
                    windowMaximum=OdentConstants.WMAX,
                    outputMinimum=0,
                    outputMaximum=255,
                ),
                sitk.sitkUInt8,
            )

        # write custom odent metadata to help check if dicom file is an Odent exported image :
        odent_metadata_dict = {
            OdentConstants.ODENT_IMAGE_METADATA_KEY: OdentConstants.ODENT_IMAGE_METADATA_TAG,
            OdentConstants.ODENT_IMAGE_METADATA_UID_KEY: self.uid,
        }
        for key, value in odent_metadata_dict.items():
            self.sitk_image.SetMetaData(key, value)

        # write main image to .mha file to be used for computing slices :
        self.preffix = f"{self.uid}_({self.user_spacing})"
        mha_file_path = join(AbsPath(self.user_proj_path), f"{self.preffix}.mha")
        sitk.WriteImage(self.sitk_image, mha_file_path)

        # set some scene props and odent props :
        context.scene[OdentConstants.MHA_PATH_TAG] = self.mha_path = RelPath(
            mha_file_path
        )
        context.scene["uid"] = self.uid
        self.current_dicom_cach_dictionary["uid"] = self.uid
        self.current_dicom_cach_dictionary["spacing"] = self.user_spacing
        self.current_dicom_cach_dictionary[OdentConstants.MHA_PATH_TAG] = RelPath(
            mha_file_path
        )
        self._props.current_dicom_dictionary = str(self.current_dicom_cach_dictionary)
        self.update_dicom_dictionary_prop()

        # assign main sitk image to global property IMAGE3D
        # IMAGE3D can be used in other operators here in this module
        # or imported in other modules
        IMAGE3D = self.sitk_image

        # expose image params needed for creating visualisation object :
        self.spacing = np.array(self.sitk_image.GetSpacing())  # (dx, dy, dz)
        self.direction = np.array(self.sitk_image.GetDirection()).reshape(3, 3)
        self.origin = np.array(self.sitk_image.GetOrigin())  # LPS coordinates
        self.size = np.array(self.sitk_image.GetSize())  # (X, Y, Z)

        # #logging main sitk image infos:
        # info_text_list = [
        #     f"Image size (voxels) : {self.size}",
        #     f"Image spacing (mm) : {np.round(self.spacing,3)}",
        #     f"Image direction : {np.round(self.direction,3).tolist()}",
        #     f"Image origin (LPS) : {np.round(self.origin,3).tolist()}",
        # ]
        # odent_log(info_text_list)

        return error_message

    def extract_threshold_points(
        self,
        sitk_image,
        max_points,
        sampling_method=OdentConstants.PCD_SAMPLING_METHOD_GRID,
    ):
        """
        Extract physical coordinates and intensities from voxels > threshold.
        Automatically downsamples image to limit number of points.

        Parameters:
            sitk_image (sitk.Image): Input SimpleITK image (should be sitkUInt8 or similar).
            threshold (int): Intensity threshold.
            max_points (int): Max number of points to keep (default: 4,000,000).

        Returns:
            Tuple of two numpy arrays:
                - points: (N, 3) array of physical coordinates
                - values: (N,) array of intensity values at those points
        """
        max_points *= 1_000_000
        # Step 1: Threshold mask and count points
        pcd_threshold_min = self.pointCloudThresholdMin
        pcd_threshold_min_255 = HuTo255(pcd_threshold_min)
        mask = sitk_image > pcd_threshold_min_255
        mask_array = sitk.GetArrayViewFromImage(mask)
        indices = np.argwhere(mask_array > 0)
        point_count = int(mask_array.sum())

        # Step 2: Resize if needed
        if (
            point_count > max_points
            and sampling_method == OdentConstants.PCD_SAMPLING_METHOD_GRID
        ):
            odent_log(["using grid sampling "])
            scale = (max_points / point_count) ** (1 / 3)
            original_size = sitk_image.GetSize()
            new_size = [max(1, int(sz * scale)) for sz in original_size]

            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            new_spacing = [
                sp * osz / nsz
                for sp, osz, nsz in zip(
                    sitk_image.GetSpacing(), original_size, new_size
                )
            ]
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputOrigin(sitk_image.GetOrigin())
            resampler.SetOutputDirection(sitk_image.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)

            sitk_image = resampler.Execute(sitk_image)
            mask = sitk_image > pcd_threshold_min_255
            mask_array = sitk.GetArrayViewFromImage(mask)
            indices = np.argwhere(mask_array > 0)

        # Step 3:: Get intensity values from image array
        image_array = sitk.GetArrayViewFromImage(sitk_image)
        intensities = (
            image_array[indices[:, 0], indices[:, 1], indices[:, 2]]
            .astype(np.int32)
            .ravel()
        )

        # Step 4: Convert to physical coordinates (vectorized)
        spacing = np.array(sitk_image.GetSpacing())
        origin = np.array((0, 0, 0))
        direction = np.eye(3)

        ijk = np.flip(indices, axis=1)  # (N, 3) in (x, y, z)
        ijk_spacing = ijk * spacing  # apply spacing
        coords = ijk_spacing @ direction.T + origin  # apply direction and origin

        if (
            point_count > max_points
            and sampling_method == OdentConstants.PCD_SAMPLING_METHOD_RANDOM
        ):
            odent_log(["using random sampling "])
            random_indices = np.random.choice(point_count, max_points, replace=False)
            coords = coords[random_indices]
            intensities = intensities[random_indices]

        return coords, intensities

    def create_mesh_from_numpy_fast(self, coords: np.ndarray, name="PointCloud"):

        coords = np.asarray(coords, dtype=np.float32)
        num_points = coords.shape[0]

        # Flatten coordinates in x, y, z order
        flat_coords = coords.ravel()  # shape: (N * 3,)

        # Create empty mesh
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(num_points)

        # Bulk assign coordinates using foreach_set
        mesh.vertices.foreach_set("co", flat_coords)
        mesh.update()

        # Create object and link to scene
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj

        return obj

    def point_cloud_visualization(self, max_points):
        """
        Creates a point cloud from SimplITK image, imports it into Blender with an integer intensity attribute,
        copies Geometry Nodes from the Point_cloud_node object, and removes that object.
        """
        # odent_log(["ODENT_OT_Dicom_Reader.point_cloud_visualization()"])
        # start = tpc()
        error_message = []
        pcd_idx = get_incremental_idx(
            data=bpy.data.objects, odent_type=OdentConstants.PCD_OBJECT_TYPE
        )
        pcd_object_name = get_incremental_name(OdentConstants.PCD_OBJECT_NAME, pcd_idx)
        pcd_geonode_name = get_incremental_name(
            OdentConstants.PCD_GEONODE_NAME, pcd_idx
        )
        pcd_geonode = bpy.data.node_groups.get(pcd_geonode_name)
        if not pcd_geonode:
            filepath = join(
                OdentConstants.DATA_BLEND_FILE,
                "NodeTree",
                OdentConstants.PCD_GEONODE_NAME,
            )
            directory = join(OdentConstants.DATA_BLEND_FILE, "NodeTree")
            filename = OdentConstants.PCD_GEONODE_NAME
            bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)
            pcd_geonode = bpy.data.node_groups.get(OdentConstants.PCD_GEONODE_NAME)
            pcd_geonode.name = pcd_geonode_name

        pcd_geonode["uid"] = self.uid

        viz_collection = bpy.data.collections.get(
            OdentConstants.DICOM_VIZ_COLLECTION_NAME
        )
        if viz_collection:
            hide_collection(
                _hide=False, colname=OdentConstants.DICOM_VIZ_COLLECTION_NAME
            )

        Scene_Settings()

        for scr in bpy.data.screens:
            areas = [area for area in scr.areas if area.type == "VIEW_3D"]
            for area in areas:
                spaces = [sp for sp in area.spaces if sp.type == "VIEW_3D"]
                for space in spaces:
                    space.shading.show_xray = False
                    space.shading.type = "MATERIAL"
                    r3d = space.region_3d
                    r3d.view_perspective = "ORTHO"
                    r3d.view_matrix = Matrix(OdentConstants.VIEW_MATRIX)
                    # space.shading.color_type = "TEXTURE"
                    r3d.update()

        # odent_log([f" scene preparation : {tpc() - start} seconds"])
        # step = tpc()
        coords, intensities = self.extract_threshold_points(
            sitk_image=self.sitk_image,
            max_points=max_points,
            sampling_method=self._props.pcd_sampling_method,
        )

        # odent_log([f" extract_threshold_points : {tpc() - step} seconds"])
        # step = tpc()
        if coords.size == 0:
            error_message = ["No points found above the threshold."]
            ODENT_GpuDrawText(
                message_list=error_message, rect_color=OdentColors.red, sleep_time=2
            )
            return error_message

        # Create Blender object

        pcd_obj = self.create_mesh_from_numpy_fast(coords, name=pcd_object_name)
        MoveToCollection(pcd_obj, OdentConstants.DICOM_VIZ_COLLECTION_NAME)

        # odent_log([f" write pcd_mesh from numpy : {tpc() - step} seconds"])
        # step = tpc()

        # Create integer threshold attribute
        attr = pcd_obj.data.attributes.new(
            name=OdentConstants.PCD_INTENSITY_ATTRIBUTE_NAME, type="INT", domain="POINT"
        )
        attr.data.foreach_set("value", intensities)
        # for i, value in enumerate(intensities):
        #     attr.data[i].value = value

        # odent_log([f" set pcd_mesh intensity attribute : {tpc() - step} seconds"])
        # step = tpc()
        # Copy Geometry Nodes modifier
        pcd_modifier = pcd_obj.modifiers.get(
            OdentConstants.PCD_GEONODE_MODIFIER_NAME
        ) or pcd_obj.modifiers.new(
            name=OdentConstants.PCD_GEONODE_MODIFIER_NAME, type="NODES"
        )
        pcd_modifier.node_group = pcd_geonode

        # Update display

        pcd_obj[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.PCD_OBJECT_TYPE
        pcd_obj[OdentConstants.MHA_PATH_TAG] = self.mha_path
        pcd_obj["idx"] = pcd_idx
        pcd_obj["uid"] = self.uid
        pcd_obj[OdentConstants.PCD_GEONODE_NAME_TAG] = pcd_geonode_name

        for i in range(3):
            pcd_obj.lock_location[i] = True
            pcd_obj.lock_rotation[i] = True
            pcd_obj.lock_scale[i] = True

        bpy.ops.object.select_all(action="DESELECT")
        pcd_obj.select_set(True)
        bpy.context.view_layer.objects.active = pcd_obj

        # trigger update pcd options in UI panel:
        current_threshold = self._props.ThresholdMin
        self._props.ThresholdMin = current_threshold

        current_pcd_point_radius = self._props.pcd_point_radius
        self._props.pcd_point_radius = current_pcd_point_radius

        current_pcd_point_auto_resize = self._props.pcd_point_auto_resize
        self._props.pcd_point_auto_resize = current_pcd_point_auto_resize

        current_pcd_points_opacity = self._props.pcd_points_opacity
        self._props.pcd_points_opacity = current_pcd_points_opacity

        current_pcd_points_emission = self._props.pcd_points_emission
        self._props.pcd_points_emission = current_pcd_points_emission

        pcd_obj.data.update()

        # set visualisation object real world position :
        viz_transform = Matrix(self.direction.tolist()).to_4x4()
        tvec = Vector(self.origin)
        viz_transform.translation = tvec
        pcd_obj.matrix_world = viz_transform @ pcd_obj.matrix_world

        bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # odent_log([f" point cloud creation : {tpc() - step} seconds"])
        # odent_log([f" total time : {tpc() - start} seconds"])

        return error_message

    def create_axial_slices(self):
        slice_images_info = {}
        slice_images_names = []
        self.temp_png_directory = OdentConstants.ODENT_VOLUME_TEXTURES_DIR
        if exists(self.temp_png_directory):
            odent_log([f"removing temp png directory : {self.temp_png_directory}"])
            shutil.rmtree(self.temp_png_directory)
        os.makedirs(self.temp_png_directory, exist_ok=True)
        odent_log([f"temp_png_directory : {self.temp_png_directory}"])
        voxel_idx = self.voxel_idx
        arr = sitk.GetArrayFromImage(self.sitk_image)
        num_slices = arr.shape[0]

        def create_axial_image(i):

            axial_slice_array = arr[i, :, :]
            alpha_channel = axial_slice_array
            slice_array_rgba = np.stack(
                [
                    axial_slice_array,
                    axial_slice_array,
                    axial_slice_array,
                    alpha_channel,
                ],
                axis=-1,
            )
            slice_image = sitk.GetImageFromArray(slice_array_rgba, isVector=True)
            slice_image_name = (
                f"{OdentConstants.VOXEL_IMAGE_NAME}_{voxel_idx}_{i:04}.png"
            )
            slice_image_path = join(self.temp_png_directory, slice_image_name)
            sitk.WriteImage(slice_image, slice_image_path)

            slice_images_info[i] = {
                "name": slice_image_name,
                "path": slice_image_path,
            }

        threads = [
            threading.Thread(
                target=create_axial_image,
                args=[
                    i,
                ],
                daemon=True,
            )
            for i in range(num_slices)
        ]
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        for i in range(num_slices):
            slice_image_name = slice_images_info[i]["name"]
            slice_image_path = slice_images_info[i]["path"]
            image = bpy.data.images.load(slice_image_path)
            image.pack()
            image.name = slice_image_name
            slice_images_names.append(slice_image_name)

        return slice_images_names, num_slices

    def add_cross_sectional_plane_mesh(self, x, y, name_preffix):

        verts = [(0, 0, 0), (x, 0, 0), (0, y, 0), (x, y, 0)]
        faces = [(0, 1, 3, 2)]
        mesh_data = bpy.data.meshes.new(f"{name_preffix}_mesh")
        mesh_data.from_pydata(verts, [], faces)
        if not mesh_data.uv_layers:
            mesh_data.uv_layers.new(name="UVMap")
        uv_layer = mesh_data.uv_layers.active.data

        # Simple square UV mapping
        uvs = [(0, 1), (1, 1), (1, 0), (0, 0)]
        for i, loop in enumerate(mesh_data.loops):
            uv_layer[loop.index].uv = uvs[i % 4]

        # Returns True if any invalid geometry was removed.
        corrections = mesh_data.validate(verbose=True, clean_customdata=True)
        mesh_data.update(calc_edges=True, calc_edges_loose=True)
        return mesh_data

    def add_cross_sectional_plane(self, mesh, slice_index, name_preffix):

        plane = bpy.data.objects.new(name_preffix, mesh.copy())
        bpy.context.scene.collection.objects.link(plane)
        z_loc = slice_index * self.spacing[2]
        plane.location = (0, 0, z_loc)
        return plane

    def textured_visualization(self, context):

        txt = ["Scene preparation, please wait..."]
        ODENT_GpuDrawText(message_list=txt)

        error_message = []
        scr = context.screen
        self.voxel_idx = get_incremental_idx(
            data=bpy.data.objects, odent_type=OdentConstants.VOXEL_OBJECT_TYPE
        )
        voxel_object_name = get_incremental_name(
            OdentConstants.VOXEL_OBJECT_NAME, self.voxel_idx
        )
        voxel_collection = bpy.data.collections.get(
            OdentConstants.DICOM_VIZ_COLLECTION_NAME
        )
        if voxel_collection:
            hide_collection(
                _hide=False, colname=OdentConstants.DICOM_VIZ_COLLECTION_NAME
            )

        # get voxel shader node group:
        voxel_node_name = get_incremental_name(
            OdentConstants.VOXEL_GROUPNODE_NAME, self.voxel_idx
        )
        voxel_node = bpy.data.node_groups.get(voxel_node_name)
        if not voxel_node:
            filepath = join(
                OdentConstants.DATA_BLEND_FILE,
                "NodeTree",
                OdentConstants.ODENT_VOXEL_SHADER,
            )
            directory = join(OdentConstants.DATA_BLEND_FILE, "NodeTree")
            filename = OdentConstants.ODENT_VOXEL_SHADER
            bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)
            voxel_node = bpy.data.node_groups.get(OdentConstants.ODENT_VOXEL_SHADER)
            voxel_node.name = voxel_node_name
        voxel_node["uid"] = self.uid
        self._props.GroupNodeName = OdentConstants.ODENT_VOXEL_SHADER

        # calculate plane dimensions
        width = round(self.size[0] * self.spacing[0], 3)
        height = round(self.size[1] * self.spacing[1], 3)

        # prepare scene for voxel visualization :
        Scene_Settings()

        for scr in bpy.data.screens:
            areas = [area for area in scr.areas if area.type == "VIEW_3D"]
            for _area in areas:
                space = _area.spaces.active
                r3d = space.region_3d
                r3d.view_perspective = "ORTHO"
                # r3d.view_distance = 800
                r3d.view_matrix = Matrix(OdentConstants.VIEW_MATRIX)
                space.shading.type = "SOLID"
                space.shading.color_type = "TEXTURE"
                r3d.update()

        areas = [area for area in scr.areas if area.type == "VIEW_3D"]
        area = areas[0]
        space = area.spaces.active

        space.shading.show_xray = False
        space.overlay.show_overlays = False

        # generate axial slices :
        txt = ["Generating cross-sectional images, please wait..."]
        ODENT_GpuDrawText(message_list=txt)

        slice_images_names, num_slices = self.create_axial_slices()

        # create planes with images:
        txt = ["Creating cross-sectional geometry 0% ..."]
        ODENT_GpuDrawText(message_list=txt, percentage=0)
        voxel_plane_names = []

        voxel_plane_mesh = self.add_cross_sectional_plane_mesh(
            width, height, voxel_object_name
        )
        ctx_area, ctx_space, ctx_region = CtxOverride(context)
        for i in range(num_slices):
            for n in voxel_plane_names:
                bpy.data.objects[n].hide_set(True)
            voxel_image_data = bpy.data.images.get(slice_images_names[i])
            voxel_image_data[OdentConstants.ODENT_TYPE_TAG] = (
                OdentConstants.VOXEL_IMAGE_TYPE
            )

            voxel_plane_name = get_incremental_name(
                OdentConstants.VOXEL_PLANE_NAME, self.voxel_idx, i
            )
            voxel_plane_obj = self.add_cross_sectional_plane(
                voxel_plane_mesh, i, voxel_plane_name
            )

            voxel_plane_names.append(voxel_plane_name)
            voxel_plane_obj[OdentConstants.ODENT_TYPE_TAG] = (
                OdentConstants.VOXEL_PLANE_TYPE
            )
            voxel_plane_obj["uid"] = self.uid

            ##########################################
            # Add Material :
            plane_mat_name = get_incremental_name(
                OdentConstants.VOXEL_PLANE_MAT_NAME, self.voxel_idx, i
            )
            plane_mat = bpy.data.materials.get(
                plane_mat_name
            ) or bpy.data.materials.new(plane_mat_name)

            plane_mat.use_nodes = True
            node_tree = plane_mat.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in nodes:
                if node.type == "OUTPUT_MATERIAL":
                    material_output = node
                else:
                    nodes.remove(node)

            image_texture_node = AddNode(
                nodes, type="ShaderNodeTexImage", name="Image Texture"
            )

            image_texture_node.image = voxel_image_data
            image_texture_node.extension = "CLIP"
            voxel_image_data.colorspace_settings.name = "Non-Color"  # "sRGB"

            plane_mat_group_node = nodes.new("ShaderNodeGroup")
            plane_mat_group_node.node_tree = voxel_node

            links.new(
                image_texture_node.outputs["Color"], plane_mat_group_node.inputs[0]
            )
            links.new(
                plane_mat_group_node.outputs[0], material_output.inputs["Surface"]
            )
            ##########################################################
            voxel_plane_obj.active_material = plane_mat

            if not (i % 5):
                bpy.ops.object.select_all(action="DESELECT")
                voxel_plane_obj.select_set(True)
                bpy.context.view_layer.objects.active = voxel_plane_obj
                with bpy.context.temp_override(
                    area=ctx_area, space_data=ctx_space, region=ctx_region
                ):
                    bpy.ops.view3d.view_selected()
                _percentage = int(i * 100 / num_slices)
                txt = [f"Creating cross-sectional geometry {_percentage}% ..."]
                ODENT_GpuDrawText(message_list=txt, percentage=_percentage)
                # sleep(0.05)
        bpy.ops.object.select_all(action="DESELECT")
        voxel_planes = [bpy.data.objects[n] for n in reversed(voxel_plane_names)]
        for vp in voxel_planes:
            vp.hide_set(False)
            vp.select_set(True)
            bpy.context.view_layer.objects.active = vp

        bpy.ops.object.join()
        voxel_vizualization_object = bpy.context.object
        # voxel_vizualization_object.hide_set(False)

        voxel_vizualization_object.select_set(True)
        voxel_vizualization_object.name = voxel_object_name
        MoveToCollection(
            voxel_vizualization_object, OdentConstants.DICOM_VIZ_COLLECTION_NAME
        )

        # we move by half spacing to have cube origin exactely a the center of voxel (0,0,0)
        origin_shift = Vector([-self.user_spacing / 2, -self.user_spacing / 2, 0])
        voxel_vizualization_object.matrix_world.translation += origin_shift

        # set visualisation object real world position :
        viz_transform = Matrix(self.direction.tolist()).to_4x4()
        tvec = Vector(self.origin)
        viz_transform.translation = tvec
        voxel_vizualization_object.matrix_world = (
            viz_transform @ voxel_vizualization_object.matrix_world
        )

        context.scene.cursor.location = (0.0, 0.0, 0.0)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        for scr in bpy.data.screens:
            areas = [area for area in scr.areas if area.type == "VIEW_3D"]
            for _area in areas:

                try:
                    _space = _area.spaces.active
                    _region = [r for r in _area.regions if r.type == "WINDOW"][0]
                    with bpy.context.temp_override(
                        area=_area, space_data=_space, region=_region
                    ):
                        bpy.ops.view3d.view_all(center=True)
                except Exception as e:
                    pass

        voxel_vizualization_object[OdentConstants.ODENT_TYPE_TAG] = (
            OdentConstants.VOXEL_OBJECT_TYPE
        )
        voxel_vizualization_object[OdentConstants.MHA_PATH_TAG] = self.mha_path
        voxel_vizualization_object["idx"] = self.voxel_idx
        voxel_vizualization_object["uid"] = self.uid
        voxel_vizualization_object["voxel_node_name"] = voxel_node_name
        voxel_vizualization_object["center_dicom"] = self.center_dicom

        # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

        for i in range(3):
            voxel_vizualization_object.lock_location[i] = True
            voxel_vizualization_object.lock_rotation[i] = True
            voxel_vizualization_object.lock_scale[i] = True

        space.shading.type = "MATERIAL"
        space.overlay.show_overlays = True
        space.shading.show_xray = True
        space.shading.xray_alpha = OdentConstants.XRAY_ALPHA

        WminNode = voxel_node.nodes["WminNode"].outputs[0]
        WminNode.default_value = OdentConstants.WMIN
        WmaxNode = voxel_node.nodes["WmaxNode"].outputs[0]
        WmaxNode.default_value = OdentConstants.WMAX

        # trigger update pcd options in UI panel:
        current_threshold = self._props.ThresholdMin
        self._props.ThresholdMin = current_threshold

        bpy.ops.object.select_all(action="DESELECT")
        voxel_vizualization_object.select_set(True)
        bpy.context.view_layer.objects.active = voxel_vizualization_object

        return error_message

    def execute(self, context):
        # #get main sitk image :
        ######################## update info bar : percentage
        pourcentage = 20
        message = [f"image processing {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)
        #########################################

        error_message = self.get_main_image(context)

        ######################## update info bar : cancell
        if error_message:
            ODENT_GpuDrawText(
                message_list=error_message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        #########################################

        ######################## debugging
        # odent_log([self.all_dicom_cach_dictionary])
        ######################## debugging

        # #create blender 3d visualisation :

        ######################## update info bar : percentage
        pourcentage = 50
        message = [f"image processing {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)
        #########################################

        # #case textured visualisation stacked cross-sectional planes :
        if self._props.visualisation_mode == OdentConstants.VISUALISATION_MODE_TEXTURED:
            error_message = self.textured_visualization(context)

        # #case point cloud visualisation using geo nodes :
        elif self._props.visualisation_mode == OdentConstants.VISUALISATION_MODE_PCD:
            error_message = self.point_cloud_visualization(
                max_points=self._props.pcd_points_max
            )

        ######################## update info bar : cancell
        if error_message:
            ODENT_GpuDrawText(
                message_list=error_message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        #########################################

        # create mpr slices :

        ######################## update info bar : percentage
        message = [f"computing slices ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)
        #########################################
        bpy.ops.wm.odent_volume_slicer()

        ######################## update info bar : end
        message = ["finished."]
        ODENT_GpuDrawText(message_list=message, sleep_time=2)
        #########################################
        # clear_terminal()

        return {"FINISHED"}


# using timer to update slices :
def check_context():
    global IMAGE3D, _LAST_SLICES_POINTER_LOC, _LAST_SLICES_POINTER_ROT, FORCE_SLICES_UPDATE
    _success = False
    _slices_pointer = None
    message = []
    scene = bpy.context.scene
    obj = bpy.context.object

    """
    case 1 : force update is False
    >>>> check if active object is not None and is selected and == slices_pointer else abort update.
    >>>> check if current workspace is slicer else abort update
    >>>> check if slices_pointer moved or rotated else abort update
    """
    if not FORCE_SLICES_UPDATE:

        if (
            not obj
            or not obj.select_get()
            or obj.get(OdentConstants.ODENT_TYPE_TAG)
            != OdentConstants.SLICES_POINTER_TYPE
        ):
            message.append("slices pointer not found or not selected")
            return _success, _slices_pointer, message

        _slices_pointer = obj

        current_workspace = bpy.context.workspace
        if (
            not current_workspace
            or not current_workspace.name == OdentConstants.SLICER_WORKSPACE_NAME
        ):
            message.append("current workspace is not odentslicer")
            return _success, _slices_pointer, message

        loc = _slices_pointer.location.copy()
        rot = _slices_pointer.rotation_euler.copy()
        moved = (
            _LAST_SLICES_POINTER_LOC is not None
            and (loc - _LAST_SLICES_POINTER_LOC).length > 1e-5
        )
        rotated = _LAST_SLICES_POINTER_ROT is not None and (
            rot.to_matrix().to_3x3() != _LAST_SLICES_POINTER_ROT.to_matrix().to_3x3()
        )
        _LAST_SLICES_POINTER_LOC = loc
        _LAST_SLICES_POINTER_ROT = rot

        if not (moved or rotated):
            message.append("slices pointer not moved or rotated")
            return _success, _slices_pointer, message

    elif FORCE_SLICES_UPDATE:
        """
        case 2 : force update is True
        >>>> check if slices_pointer exists else abort update
        """
        slices_pointer_candidates = [
            obj
            for obj in scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_candidates:
            message.append("slices pointer not found")
            return _success, _slices_pointer, message

        _slices_pointer = slices_pointer_candidates[0]

    """
    for case1 and case 2 : 
    >>>> check if IMAGE3D is in memory else get image from .mha
    >>>> if for some reason image cannot be got from .mha abort update
    """

    if IMAGE3D is None:
        message.append("IMAGE3D is None getting image from .mha")
        mha_path = _slices_pointer.get(OdentConstants.MHA_PATH_TAG)
        if not mha_path or not isfile(AbsPath(mha_path)):
            message.append("can't get .mha file")
            return _success, _slices_pointer, message

        try:
            IMAGE3D = sitk.ReadImage(AbsPath(mha_path))
            message.append("IMAGE3D : read image from .mha -> ok")
        except Exception as e:
            message.append(f"Failed to read image from .mha:\n\t{e}")
            return _success, _slices_pointer, message

    _success = True
    message.append("check context = ok.")

    return _success, _slices_pointer, message

def extract_slice(image, plane, inverted_transform):
    ###### get out spacing and size ###########
    # Blender plane physical size (scale applied)
    dims = plane.dimensions  # Vector(width, height, depth)
    plane_width, plane_height = dims.x, dims.y

    # Use source image spacing (in-plane)
    sx, sy, sz = image.GetSpacing()
    # Compute output slice size
    size_x = int(round(plane_width / sx))
    size_y = int(round(plane_height / sy))
    # #update sx,sy
    # sx = plane_width/size_x
    # sx = plane_height/size_y

    out_spacing = [sx, sy, 1.0]
    out_size = (size_x, size_y, 1)
    # update plane dimensions :
    plane.dimensions = Vector(out_spacing) * Vector(out_size)

    ###### get out origin and direction ###########

    # get the slice plane local transform with respect to visualisation object
    # if no visualisation object the inverted transform is identity matrix 4x4 (no effect) !
    M = inverted_transform @ plane.matrix_world
    # Take vertex 0 in visualisation object coordinates as slice origin:
    shifted_origin = plane.data.vertices[0].co + Vector([sx / 2, sx / 2, 0])
    out_origin = list(M @ (shifted_origin))

    # get the 3x3 slice plane rotation matrix with respect to visualisation object
    # Convert to numpy
    # Flatten to 1 Dimension
    rot3 = M.to_3x3().normalized()
    rot_np = np.array(rot3)
    out_direction = rot_np.flatten().tolist()

    # --------------------------
    # 4. Build SITK reference slice
    # --------------------------
    slice_img = sitk.Image(out_size, image.GetPixelID())
    slice_img.SetSpacing(out_spacing)
    slice_img.SetOrigin(out_origin)
    slice_img.SetDirection(out_direction)

    # --------------------------
    # 5. Resample 3D image into slice
    # --------------------------
    resampled = sitk.Resample(
        image,
        slice_img,
        sitk.Transform(3, sitk.sitkIdentity),  # no extra transform needed
        sitk.sitkLinear,
        0.0,
        image.GetPixelID(),
    )

    return resampled

def _update_slices(slices_pointer):
    global IMAGE3D
    message = []
    success = False

    try:

        props = bpy.context.scene.ODENT_Props

        """
        normal behaviour : slices should be computed with respect to ,
        the vizualisation object transform
        """

        # initialize the slices planes global transform
        inverted_transform = Matrix()

        # get the vizualisation object and the its inverted transformation
        uid = slices_pointer.get("uid")
        idx = slices_pointer.get("idx")
        vizobj_check = [
            obj
            for obj in bpy.context.scene.objects
            if obj.get("uid") == uid
            and obj.get("idx") == idx
            and obj.get(OdentConstants.ODENT_TYPE_TAG)
            in [OdentConstants.VOXEL_OBJECT_TYPE, OdentConstants.PCD_OBJECT_TYPE]
        ]
        if vizobj_check:
            vizobj = vizobj_check[0]
            inverted_transform = vizobj.matrix_world.inverted()

        # odent_log([f"inverted_transform = {inverted_transform}"])

        """
        Blender slice images data block(axial,coronal,sagittal) is created
        when running the ODENT_OT_VolumeSlicer operator
        """
        # make a map [(slice_plane, blender_image_data)]
        slice_planes = [
            obj
            for obj in bpy.context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_PLANE_TYPE
        ]
        img_data_list = [bpy.data.images.get(p.name) for p in slice_planes]
        slices_map = zip(slice_planes, img_data_list)

        for p, im_data in slices_map:
            if not im_data:
                continue

            # Extract slice & flip
            resampled = extract_slice(IMAGE3D, p, inverted_transform)
            flipped_image = sitk.Flip(resampled, [False, False, False])

            # Convert SITK → uint8 array (H,W)
            arr_u8 = sitk.GetArrayFromImage(flipped_image)

            # SimpleITK returns (z,y,x) or (y,x) - we need to handle both
            if len(arr_u8.shape) == 3:
                # Take first slice if we got a 3D array
                arr_u8 = arr_u8[0]

            h, w = arr_u8.shape

            # =====================================================================
            # THRESHOLD MODE (OpenCV-based color overlay)
            # =====================================================================
            if props.slicesColorThresholdBool:

                # HU threshold → uint8
                threshold255 = HuTo255(
                    props.ThresholdMin, OdentConstants.WMIN, OdentConstants.WMAX
                )
                threshold255 = max(1, min(threshold255, 254))

                # mask
                mask = arr_u8 >= threshold255

                # grayscale → RGB
                rgb = cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)

                # segmentation color (RGB only)
                col = props.SegmentColor
                sr = int(col[0] * 255)
                sg = int(col[1] * 255)
                sb = int(col[2] * 255)

                # apply color to mask
                rgb[mask] = (sr, sg, sb)

                rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
                # add alpha
                alpha = np.full((h, w), 255, dtype=np.uint8)
                rgba = np.dstack((rgb, alpha))  # (H,W,4)

                # float32 → Blender
                rgba_f32 = rgba.astype(np.float32).reshape(-1) * (1.0 / 255.0)
                im_data.pixels.foreach_set(rgba_f32)
                im_data.update()

            # =====================================================================
            # NORMAL MODE
            # =====================================================================
            else:

                flat_u8 = arr_u8.reshape(-1)
                N = h * w

                rgba_u8 = np.empty(N * 4, dtype=np.uint8)
                rgba_u8[0::4] = flat_u8
                rgba_u8[1::4] = flat_u8
                rgba_u8[2::4] = flat_u8
                rgba_u8[3::4] = 255

                rgba_f32 = rgba_u8.astype(np.float32)
                rgba_f32 *= 1.0 / 255.0

                im_data.pixels.foreach_set(rgba_f32)

                im_data.update()

            success = True
            message.append(f"{im_data.name} updated successfully.")
    except Exception as e:
        pass
        message.append(f"failed to update slices :\n\t{e}")

    return success, message

@persistent
def update_slices_handler_callback(depsgraph):
    global SLICES_POINTER, SLICES_DIRTY, SLICES_UPDATING
    if SLICES_UPDATING:
        return
    if SLICES_DIRTY:
        return

    SLICES_DIRTY, SLICES_POINTER, message = check_context()

def force_update_slices():
    global FORCE_SLICES_UPDATE
    FORCE_SLICES_UPDATE = True


class ODENT_OT_VolumeSlicer(bpy.types.Operator):
    """Add Volume Slices"""

    bl_idname = "wm.odent_volume_slicer"
    bl_label = "Generate/Update Dicom slices"
    bl_options = {"REGISTER", "UNDO"}
    global IMAGE3D
    open_slices_view: BoolProperty(default=True)  # type: ignore

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        return context.object.get(OdentConstants.ODENT_TYPE_TAG) in [
            OdentConstants.VOXEL_OBJECT_TYPE,
            OdentConstants.PCD_OBJECT_TYPE,
        ]

    def execute(self, context):
        global IMAGE3D
        self.vis_object = context.object
        self.props = context.scene.ODENT_Props
        self.uid = self.vis_object["uid"]
        self.idx = self.vis_object["idx"]

        self.main_ws = bpy.data.workspaces.get(OdentConstants.MAIN_WORKSPACE_NAME)
        self.ws_slicer = bpy.data.workspaces.get(OdentConstants.SLICER_WORKSPACE_NAME)

        if not (self.main_ws and self.ws_slicer):
            ODENT_GpuDrawText(
                message_list=["Odent Workspaces not found"],
                rect_color=OdentColors.red,
                sleep_time=2,
            )
            return {"CANCELLED"}
        self.source_mha = self.vis_object.get(OdentConstants.MHA_PATH_TAG)
        if not self.source_mha or not exists(AbsPath(self.source_mha)):
            ODENT_GpuDrawText(
                message_list=["Odent MHA file not found"],
                rect_color=OdentColors.red,
                sleep_time=2,
            )
            return {"CANCELLED"}

        context.scene[OdentConstants.MHA_PATH_TAG] = self.source_mha
        IMAGE3D = sitk.ReadImage(AbsPath(self.source_mha))

        # bounding_box is in local coordinates, so transform to world
        world_bbox = [
            self.vis_object.matrix_world @ Vector(corner)
            for corner in self.vis_object.bound_box
        ]
        self.vis_object_bbox_center = sum(world_bbox, Vector()) / 8

        self.remove_old_stuff(context)

        context.scene.render.resolution_x = 512
        context.scene.render.resolution_y = 512

        self.slices_pointer = self.add_slices_pointer(context)
        (
            self.axial_p,
            self.coronal_p,
            self.sagittal_p,
            self.axial_cam,
            self.coronal_cam,
            self.sagittal_cam,
        ) = self.add_slices(context)

        # activate_obj(self.slices_pointer)

        for obj in [
            self.axial_p,
            # self.axial_cam,
            self.coronal_p,
            # self.coronal_cam,
            self.sagittal_p,
            # self.sagittal_cam,
        ]:
            obj.matrix_world = self.slices_pointer.matrix_world @ obj.matrix_world

            child_of = obj.constraints.new("CHILD_OF")
            child_of.target = self.slices_pointer
            child_of.use_scale_x = False
            child_of.use_scale_y = False
            child_of.use_scale_z = False

            for i in range(3):
                obj.lock_location[i] = True
                obj.lock_rotation[i] = True
                obj.lock_scale[i] = True

        ################
        # update slices to get last slices poinre position
        force_update_slices()
        ################

        return self.set_mpr(context)

    def add_cam_to_plane(self, context, p):
        override, a3d, s3d, r3d = context_override(context)
        with context.temp_override(**override):
            bpy.ops.view3d.view_selected()
        bpy.ops.object.camera_add()
        Cam = context.object
        Cam.data.passepartout_alpha = 1
        Cam.data.show_name = True



        child_of = Cam.constraints.new("CHILD_OF")
        child_of.target = p
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        for i in range(3):
            Cam.lock_location[i] = True
            Cam.lock_rotation[i] = True
            Cam.lock_scale[i] = True
        # Cam.parent = p
        Cam.name = f"{p.name}_CAM"
        Cam.data.name = f"{p.name}_CAM_data"
        Cam[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.SLICE_CAM_TYPE
        Cam.data.type = "ORTHO"
        Cam.data.ortho_scale = max(p.dimensions)  # * 1.1
        Cam.data.display_size = 10

        transform = Matrix.Identity(4)
        transform.translation = Vector((0, 0, OdentConstants.CAM_DISTANCE))

        Cam.matrix_world = transform

        Cam.data.clip_start = (
            OdentConstants.CAM_DISTANCE - OdentConstants.CAM_CLIP_OFFSET
        )
        Cam.data.clip_end = OdentConstants.CAM_DISTANCE + OdentConstants.CAM_CLIP_OFFSET

        Cam.hide_set(True)
        MoveToCollection(obj=Cam, CollName=OdentConstants.SLICES_COLLECTION_NAME)

        return Cam

    def set_image3d(self, context):
        global IMAGE3D
        message = []
        scn_mha_path = context.scene.get(OdentConstants.MHA_PATH_TAG)
        if not scn_mha_path or not exists(AbsPath(scn_mha_path)):
            message = ["odent dicom data file not found, please read dicom and retry !"]
        obj = context.object
        if obj and obj.select_get() and obj.get(OdentConstants.MHA_PATH_TAG):
            mha_path_rel = obj.get(OdentConstants.MHA_PATH_TAG)
            if not mha_path_rel == scn_mha_path:
                context.scene[OdentConstants.MHA_PATH_TAG] = mha_path_rel
                IMAGE3D = sitk.ReadImage(AbsPath(mha_path_rel))
                self.source_mha = AbsPath(mha_path_rel)
        else:
            if not IMAGE3D or self.source_mha != scn_mha_path:
                context.scene[OdentConstants.MHA_PATH_TAG] = self.source_mha
                IMAGE3D = sitk.ReadImage(AbsPath(self.source_mha))

        return message

    def add_slices_pointer(self, context):

        # bpy.ops.object.empty_add(
        # type="PLAIN_AXES",
        # scale=(1, 1, 1),
        # )
        # slices_pointer = context.object

        slices_pointer = AppendObject(
            OdentConstants.SLICES_POINTER_NAME,
            OdentConstants.SLICES_POINTER_COLLECTION_NAME,
        )

        transform = Matrix.Identity(4)
        transform.translation = self.vis_object_bbox_center
        slices_pointer.matrix_world = transform

        # slices_pointer.empty_display_size = 20
        slices_pointer.show_in_front = True
        # slices_pointer.name = OdentConstants.SLICES_POINTER_NAME
        slices_pointer[OdentConstants.ODENT_TYPE_TAG] = (
            OdentConstants.SLICES_POINTER_TYPE
        )
        slices_pointer[OdentConstants.MHA_PATH_TAG] = self.source_mha
        # slices_pointer["voxel_node_name"] = self.voxel_node_name
        slices_pointer["uid"] = self.uid
        slices_pointer["idx"] = self.idx
        slices_pointer.lock_scale = [True for _ in slices_pointer.lock_scale]
        # MoveToCollection(slices_pointer, OdentConstants.SLICES_POINTER_COLLECTION_NAME)
        bpy.ops.object.select_all(action="DESELECT")
        slices_pointer.select_set(True)
        # self.vis_object.select_set(True)
        # bpy.context.view_layer.objects.active = self.vis_object
        # bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
        # with context.temp_override(active_object=self.vis_object, selected_objects=[self.vis_object, slices_pointer], selected_editable_objects=[self.vis_object, slices_pointer]):
        #     bpy.ops.wm.odent_parent_object()
        # slices_pointer.parent=self.vis_object
        return slices_pointer

    def add_slices(self, context):

        # scn = context.scene
        # slices_dir = OdentConstants.ODENT_SLICES_DIR
        # if not exists(AbsPath(slices_dir)):
        #     os.makedirs(AbsPath(slices_dir), exist_ok=True)
        # self.props.SlicesDir = AbsPath(slices_dir)

        sp = IMAGE3D.GetSpacing()
        sz = IMAGE3D.GetSize()
        self.max_dim = get_image_diagonal_lenght(sz, sp) / 2

        slice_planes_names = [
            OdentConstants.AXIAL_SLICE_NAME,
            OdentConstants.CORONAL_SLICE_NAME,
            OdentConstants.SAGITTAL_SLICE_NAME,
        ]
        slice_planes_obj = []
        cams = []

        for s_name in slice_planes_names:
            bpy.ops.mesh.primitive_plane_add()
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.subdivide(number_cuts=100)
            bpy.ops.object.mode_set(mode="OBJECT")
            p = bpy.context.object

            p.dimensions = Vector((self.max_dim, self.max_dim, 0.0))
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            p.name = s_name
            p.data.name = f"{s_name}_mesh"
            p[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.SLICE_PLANE_TYPE
            p.rotation_mode = "XYZ"
            MoveToCollection(obj=p, CollName=OdentConstants.SLICES_COLLECTION_NAME)
            p.hide_select = True
            slice_planes_obj.append(p)

        # self.slices_pointer.select_set(True)
        # context.view_layer.objects.active = self.slices_pointer
        ######################################################
        force_update_slices()
        ######################################################
        slices_node_group = bpy.data.node_groups.get(OdentConstants.SLICES_SHADER_NAME)
        if not slices_node_group:
            filepath = join(
                OdentConstants.DATA_BLEND_FILE,
                "NodeTree",
                OdentConstants.SLICES_SHADER_NAME,
            )
            directory = join(OdentConstants.DATA_BLEND_FILE, "NodeTree")
            filename = OdentConstants.SLICES_SHADER_NAME
            bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)
            slices_node_group = bpy.data.node_groups.get(
                OdentConstants.SLICES_SHADER_NAME
            )

        for p in slice_planes_obj:

            Cam = self.add_cam_to_plane(context, p)
            Cam.hide_select = True
            cams.append(Cam)

            if "coronal" in p.name.lower():
                rotation_euler = Euler((pi / 2, 0.0, 0.0), "XYZ")
                mat = rotation_euler.to_matrix().to_4x4()
                p.matrix_world = mat @ p.matrix_world
                Cam.matrix_world = mat @ Cam.matrix_world

            elif "sagittal" in p.name.lower():
                rotation_euler = Euler((pi / 2, 0.0, 0.0), "XYZ")
                mat = rotation_euler.to_matrix().to_4x4()
                p.matrix_world = mat @ p.matrix_world
                Cam.matrix_world = mat @ Cam.matrix_world

                rotation_euler = Euler((0.0, 0.0, -pi / 2), "XYZ")
                mat = rotation_euler.to_matrix().to_4x4()
                p.matrix_world = mat @ p.matrix_world
                Cam.matrix_world = mat @ Cam.matrix_world

            # Add Material :
            bpy.ops.object.select_all(action="DESELECT")
            p.select_set(True)
            bpy.context.view_layer.objects.active = p
            bpy.ops.object.material_slot_remove_all()

            mat = bpy.data.materials.get(f"{p.name}_mat") or bpy.data.materials.new(
                f"{p.name}_mat"
            )
            p.active_material = mat
            mat.use_nodes = True
            node_tree = mat.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in nodes:
                if node.type == "OUTPUT_MATERIAL":
                    materialOutput = node
                else:
                    nodes.remove(node)

            ImageName = p.name
            BlenderImage = bpy.data.images.get(ImageName)
            if not BlenderImage:
                img_size = int(round(self.max_dim / sp[0]))
                BlenderImage = bpy.data.images.new(
                    name=ImageName,
                    # needed to check this part!
                    width=img_size,
                    height=img_size,
                    # needed to check this part!
                    alpha=True,
                    float_buffer=True,
                )
                BlenderImage.pack()
                BlenderImage.colorspace_settings.name = "sRGB"
                BlenderImage[OdentConstants.ODENT_TYPE_TAG] = (
                    OdentConstants.SLICE_IMAGE_TYPE
                )

            ImageTexture = AddNode(
                nodes, type="ShaderNodeTexImage", name="Image Texture"
            )

            ImageTexture.image = BlenderImage

            GroupNode = nodes.new("ShaderNodeGroup")
            GroupNode.node_tree = slices_node_group

            links.new(ImageTexture.outputs[0], GroupNode.inputs[0])
            links.new(GroupNode.outputs[0], materialOutput.inputs["Surface"])

        ##########################################################
        context.scene.transform_orientation_slots[0].type = "LOCAL"
        context.scene.transform_orientation_slots[1].type = "LOCAL"

        AxialPlane, CoronalPlane, SagittalPlane = slice_planes_obj
        AxialCam, CoronalCam, SagittalCam = cams
        return (
            AxialPlane,
            CoronalPlane,
            SagittalPlane,
            AxialCam,
            CoronalCam,
            SagittalCam,
        )

    def remove_old_stuff(self, context):
        slices_pointer_checklist = [
            obj
            for obj in bpy.context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if slices_pointer_checklist:
            for sp in slices_pointer_checklist:
                remove_pointer_lock()
                bpy.data.objects.remove(sp)

        coll = bpy.data.collections.get(OdentConstants.SLICES_COLLECTION_NAME)
        if coll:
            for obj in coll.objects:
                odent_log(f"removing old object : {obj.name}")
                data = obj.data
                bpy.data.objects.remove(obj)
                if data:
                    if data.users == 0:
                        if isinstance(data, bpy.types.Camera):
                            bpy.data.cameras.remove(data)
                        elif isinstance(data, bpy.types.Mesh):
                            bpy.data.meshes.remove(data)
                    else:
                        print(
                            f"object {obj.name} has users = {data.users} so data block not removed"
                        )
                else:
                    print(f" {obj.name} dont have data block")

            # bpy.data.collections.remove(coll)

        images = [
            img
            for img in bpy.data.images
            if img.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_IMAGE_TYPE
        ]
        for img in images:
            bpy.data.images.remove(img)

    def set_mpr(self, context):

        success, override_data = get_odent_workspaces()

        if not success:
            ODENT_GpuDrawText(
                message_list=["Error getting Odent Workspaces"],
                rect_color=OdentColors.red,
                sleep_time=2,
            )
            return {"CANCELLED"}

        ws_slicer_mpr_to_keep_sentences = [
            OdentConstants.SLICES_COLLECTION_NAME,
            OdentConstants.SLICES_POINTER_COLLECTION_NAME,
            OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME,
        ]
        ws_slicer_mpr_to_hide_coll_names = [
            col.name
            for col in bpy.data.collections
            if not any(
                col.name == sentence for sentence in ws_slicer_mpr_to_keep_sentences
            )
        ]

        ws_slicer_area_3d_to_hide_coll_names = [OdentConstants.SLICES_COLLECTION_NAME]
        ws_main_area_3d_to_hide_coll_names = [OdentConstants.SLICES_COLLECTION_NAME]

        #####################################################
        # set slices areas cameras and local collections view :
        # NOTE : the collections hiding is needed to avoid 
        # showing slices in main 3D view and slicer 3D view
        # and avoid showing other objects than slices in slices areas (axial, coronal, sagittal) :
        slices_overrides = [
            override_data["ws_slicer_area_axial"],
            override_data["ws_slicer_area_coronal"],
            override_data["ws_slicer_area_sagittal"],
        ]
        cams = [self.axial_cam, self.coronal_cam, self.sagittal_cam]
        for override, cam in zip(slices_overrides, cams):
            space_data = override["space_data"]
            area = override["area"]
            
            space_data.camera = cam
            space_data.use_local_camera = True
            space_data.shading.type = "MATERIAL"
            space_data.shading.studiolight_background_alpha = 0.0
            
            space_data.region_3d.lock_rotation = False 
            area.tag_redraw()
            
            with context.temp_override(**override):
                bpy.ops.view3d.view_camera()
                bpy.ops.view3d.view_center_camera()
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                if not context.scene.get("odent_slicer_is_set") == True:
                    odent_log(f"hiding collections in slices area : {[col_name for col_name in ws_slicer_area_3d_to_hide_coll_names]}")
                    space_data.use_local_collections = True
                    for col_name in ws_slicer_mpr_to_hide_coll_names:
                        index = getLocalCollIndex(col_name)
                        if index:
                            bpy.ops.object.hide_collection(
                                collection_index=index, toggle=True
                            )
            # after setting cam view we can lock rotation to avoid losing cam view by mistake when user rotate in slices area :
            space_data.region_3d.lock_rotation = True
            area.tag_redraw()

        #####################################################
        bpy.ops.object.select_all(action="DESELECT")
        self.vis_object.select_set(True)
        bpy.context.view_layer.objects.active = self.vis_object
        
        
        # set ws_main area 3d :
        
        override = override_data["ws_main_area_3d"]
        space_data = override["space_data"]
        area = override["area"]
        # area.tag_redraw()
        with bpy.context.temp_override(**override):
            bpy.ops.view3d.view_selected(use_all_regions=False)
            if not context.scene.get("odent_slicer_is_set") == True:
                odent_log(f"hiding collections in main area : {[col_name for col_name in ws_main_area_3d_to_hide_coll_names]}")
                space_data.use_local_collections = True
                for col_name in ws_main_area_3d_to_hide_coll_names:
                    index = getLocalCollIndex(col_name)
                    if index:
                        bpy.ops.object.hide_collection(collection_index=index, toggle=True)
        area.tag_redraw()
        
        #####################################################
        # set ws_slicer area 3d :
        override = override_data["ws_slicer_area_3d"]
        space_data = override["space_data"]
        area = override["area"]
        space_data.shading.use_scene_lights_render = True
        space_data.shading.use_scene_world_render = True
        space_data.shading.use_scene_lights = True
        space_data.shading.use_scene_world = False
        space_data.shading.type = "MATERIAL"  # "RENDERED"
        # area.tag_redraw()
        with bpy.context.temp_override(**override):
            bpy.ops.view3d.view_selected(use_all_regions=False)
            if not context.scene.get("odent_slicer_is_set") == True:
                space_data.use_local_collections = True
                odent_log(f"hiding collections in slices area : {[col_name for col_name in ws_slicer_area_3d_to_hide_coll_names]}")
                for col_name in ws_slicer_area_3d_to_hide_coll_names:
                    index = getLocalCollIndex(col_name)
                    if index:
                        bpy.ops.object.hide_collection(collection_index=index, toggle=True)
        area.tag_redraw()
        
        ################################################
        self.props.slices_brightness = 0.0
        self.props.slices_contrast = 0.0

        

        # expand slices collection off in outliner
        # if not context.scene.get("odent_slicer_is_set") == True:
        
        for override in override_data["ws_main_area_outliner"], override_data["ws_slicer_area_outliner"]:
            override["area"].tag_redraw()
            with context.temp_override(**override, active_collection = bpy.data.collections.get(OdentConstants.SLICES_COLLECTION_NAME)):
                bpy.ops.outliner.show_one_level(open=False)
            override["area"].tag_redraw()
        
                
        # for ws in bpy.data.workspaces:
        #     scr = ws.screens[0]
        #     override_outliner = odent_override_context(workspace=ws, screen=scr, area_type="OUTLINER", region_type="WINDOW")
        #     if override_outliner:
        #         odent_log([f"override_outliner : {override_outliner}"])
        #         with bpy.context.temp_override(**override_outliner):
        #             # bpy.ops.outliner.show_one_level(open=False)
        #             for i in range(2):
        #                 bpy.ops.outliner.expanded_toggle()
        #                 override_outliner["area"].tag_redraw()
        #             # bpy.ops.outliner.expanded_toggle()
        #             # override_outliner["area"].tag_redraw()
        bpy.ops.object.select_all(action="DESELECT")
        self.slices_pointer.select_set(True)
        bpy.context.view_layer.objects.active = self.slices_pointer
        
        if self.open_slices_view:
            ws_slicer = bpy.data.workspaces.get(OdentConstants.SLICER_WORKSPACE_NAME)
            try:
                context.window.workspace = ws_slicer
            except Exception as e:
                odent_log([f"open odent slicer workspace error : {e}"])
                pass
        
        context.scene["odent_slicer_is_set"] = True

        return {"FINISHED"}


class ODENT_OT_DicomToMesh(bpy.types.Operator):
    """Dicom to Mesh segmentation"""

    bl_idname = "wm.odent_dicom_to_mesh"
    bl_label = "Dicom To Mesh"
    bl_description = "Create Mesh from DICOM data"
    bl_options = {"REGISTER", "UNDO"}

    debug = OdentConstants.DEBUG

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False

        return context.object.get(OdentConstants.ODENT_TYPE_TAG) in [
            # OdentConstants.SLICES_POINTER_TYPE,
            OdentConstants.VOXEL_OBJECT_TYPE,
            OdentConstants.PCD_OBJECT_TYPE,
        ]

    def set_image3d(self, context):
        global IMAGE3D
        image3d_is_ok = False
        if not IMAGE3D:
            if self.debug:
                print("IMAGE3D is not set")
        elif self.source_mha != context.scene.get(OdentConstants.MHA_PATH_TAG):
            if self.debug:
                print("IMAGE3D dont match context mha path")
            IMAGE3D = None
            image3d_is_ok = False
        if not image3d_is_ok:
            context.scene[OdentConstants.MHA_PATH_TAG] = self.source_mha
            IMAGE3D = sitk.ReadImage(AbsPath(self.source_mha))

    def dicom_to_mesh(self, context):
        global IMAGE3D
        error_message = []
        pourcentage = 50
        message = [f"Dicom to mesh {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)

        vtk_image = sitk_to_vtk_image(IMAGE3D, visobj=self.vis_object)
        threshold255 = HuTo255(
            self.props.ThresholdMin, OdentConstants.WMIN, OdentConstants.WMAX
        )
        if threshold255 == 0:
            threshold255 += 1
        elif threshold255 == 255:
            threshold255 -= 1
        _marching_cubes = vtk_marching_cubes(vtk_image, threshold255)

        # Extract geometry and create Blender mesh
        pourcentage = 80
        message = [f"Dicom to mesh {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)

        error_message, dicom_mesh = create_blender_mesh_from_marching_cubes_fast(
            name=f"{OdentConstants.DICOM_MESH_NAME}#{self.idx}({self.props.ThresholdMin})",
            marching_cubes=_marching_cubes,
            visobj=None,
            transform=None,
        )
        if error_message:
            return error_message, dicom_mesh

        dicom_mesh[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.DICOM_MESH_TYPE
        dicom_mesh["idx"] = self.idx
        dicom_mesh["uid"] = self.uid
        dicom_mesh[OdentConstants.MHA_PATH_TAG] = self.source_mha
        dicom_mesh["threshold"] = self.props.ThresholdMin
        return error_message, dicom_mesh

    def execute(self, context):
        start = time.perf_counter()

        global IMAGE3D
        pourcentage = 20
        message = [f"Dicom to mesh {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)
        self.vis_object = context.object
        self.props = context.scene.ODENT_Props
        self.source_mha = context.object[OdentConstants.MHA_PATH_TAG]
        self.idx = self.vis_object["idx"]
        self.uid = self.vis_object["uid"]

        self.set_image3d(context)

        error_message, dicom_mesh = self.dicom_to_mesh(context)
        if error_message:
            ODENT_GpuDrawText(
                message_list=error_message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        dicom_mesh.matrix_world = self.vis_object.matrix_world
        color = self.props.SegmentColor
        mat = bpy.data.materials.get(
            f"{dicom_mesh.name}_mat"
        ) or bpy.data.materials.new(f"{dicom_mesh.name}_mat")
        dicom_mesh.active_material = mat
        mat.diffuse_color = color
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # change the color of the material
            bsdf.inputs[0].default_value = color

        context.scene["odent_slicer_is_set"] = True

        

        ODENT_GpuDrawText(
            message_list=["Finished"], rect_color=OdentColors.green, sleep_time=2
        )
        return {"FINISHED"}


class ODENT_OT_DicomToMesh2(bpy.types.Operator):
    """Dicom to Mesh segmentation"""

    bl_idname = "wm.odent_dicom_to_mesh2"
    bl_label = "Dicom To Mesh"
    bl_description = "Create Mesh from DICOM data"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False

        return context.object.get(OdentConstants.ODENT_TYPE_TAG) in [
            # OdentConstants.SLICES_POINTER_TYPE,
            OdentConstants.VOXEL_OBJECT_TYPE,
            OdentConstants.PCD_OBJECT_TYPE,
        ]

    def set_image3d(self, context):
        global IMAGE3D
        image3d_is_ok = False
        if not IMAGE3D:
            if self.debug:
                print("IMAGE3D is not set")
        elif self.source_mha != context.scene.get(OdentConstants.MHA_PATH_TAG):
            if self.debug:
                print("IMAGE3D dont match context mha path")
            IMAGE3D = None
            image3d_is_ok = False
        if not image3d_is_ok:
            context.scene[OdentConstants.MHA_PATH_TAG] = self.source_mha
            IMAGE3D = sitk.ReadImage(AbsPath(self.source_mha))

    def dicom_to_mesh(self, context):
        global IMAGE3D
        error_message = []
        pourcentage = 30
        message = [f"Dicom to mesh {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)

        segmentator = Segmentator_Fast(
            sitk_image=IMAGE3D, threshold=self.threshold, viz_object=self.vis_object
        )

        pourcentage = 65
        message = [f"Dicom to mesh {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)

        segment_name = f"{OdentConstants.DICOM_MESH_NAME}#{self.idx}({self.threshold})"
        segmented_mesh = segmentator.generate_segmented_mesh(object_name=segment_name)
        if segmented_mesh:
            pourcentage = 80
            message = [f"Dicom to mesh {pourcentage}% ..."]
            ODENT_GpuDrawText(message_list=message, percentage=pourcentage)

            segmented_mesh[OdentConstants.ODENT_TYPE_TAG] = (
                OdentConstants.DICOM_MESH_TYPE
            )
            segmented_mesh["idx"] = self.idx
            segmented_mesh["uid"] = self.uid
            segmented_mesh[OdentConstants.MHA_PATH_TAG] = self.source_mha
            segmented_mesh["threshold"] = self.props.ThresholdMin
        else:
            error_message = ["Can't generate segmented mesh maybe its just a vtk test!"]
        return error_message, segmented_mesh

    def execute(self, context):
        start = time.perf_counter()
        global IMAGE3D
        pourcentage = 20
        message = [f"Dicom to mesh {pourcentage}% ..."]
        ODENT_GpuDrawText(message_list=message, percentage=pourcentage)
        self.vis_object = context.object
        self.props = context.scene.ODENT_Props
        self.source_mha = context.object[OdentConstants.MHA_PATH_TAG]
        self.idx = self.vis_object["idx"]
        self.uid = self.vis_object["uid"]
        self.threshold = self.props.ThresholdMin
        self.debug = OdentConstants.DEBUG

        self.set_image3d(context)

        error_message, segmented_mesh = self.dicom_to_mesh(context)
        if error_message:
            ODENT_GpuDrawText(
                message_list=error_message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        segmented_mesh.matrix_world = self.vis_object.matrix_world
        color = self.props.SegmentColor
        mat = bpy.data.materials.get(
            f"{segmented_mesh.name}_mat"
        ) or bpy.data.materials.new(f"{segmented_mesh.name}_mat")
        segmented_mesh.active_material = mat
        mat.diffuse_color = color
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # change the color of the material
            bsdf.inputs[0].default_value = color
        end = time.perf_counter()
        print(f"<threshold = {self.threshold}> segmentation time = {end-start} seconds")
        ODENT_GpuDrawText(
            message_list=["Finished"], rect_color=OdentColors.green, sleep_time=2
        )

        return {"FINISHED"}


class ODENT_OT_MultiTreshSegment(bpy.types.Operator):
    """Add a mesh Segmentation using Treshold"""

    bl_idname = "wm.odent_multitresh_segment"
    bl_label = "SEGMENTATION"

    TimingDict = {}
    message_queue = Queue()
    Exported = Queue()

    def ImportMeshStl(self, Segment, SegmentStlPath, SegmentColor):

        # import stl to blender scene :
        bpy.ops.wm.stl_import(filepath=SegmentStlPath)
        obj = bpy.context.object
        obj.name = f"{self.Preffix}_{Segment}_SEGMENTATION"
        obj.data.name = f"{self.Preffix}_{Segment}_mesh"

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

        ############### step 8 : Add material... #########################
        mat = bpy.data.materials.get(obj.name) or bpy.data.materials.new(obj.name)
        mat.diffuse_color = SegmentColor
        obj.data.materials.append(mat)
        MoveToCollection(obj=obj, CollName="SEGMENTS")
        bpy.ops.object.shade_smooth()

        # bpy.ops.object.modifier_add(type="CORRECTIVE_SMOOTH")
        # bpy.context.object.modifiers["CorrectiveSmooth"].iterations = 2
        # bpy.context.object.modifiers["CorrectiveSmooth"].use_only_smooth = True
        # bpy.ops.object.modifier_apply(modifier="CorrectiveSmooth")

        print(f"{Segment} Mesh Import Finished")

        return obj

        # self.q.put(["End"])

    def DicomToStl(self, Segment, Image3D):
        message_queue = self.message_queue
        print(f"{Segment} processing ...")
        message = [f"Extracting {Segment} segment ..."]
        message_queue.put(message)
        # Load Infos :
        #########################################################################
        ODENT_Props = bpy.context.scene.ODENT_Props
        UserProjectDir = AbsPath(ODENT_Props.UserProjectDir)
        DcmInfo = self.DcmInfo
        Origin = DcmInfo["Origin"]
        VtkTransform_4x4 = DcmInfo["VtkTransform_4x4"]
        TransformMatrix = DcmInfo["TransformMatrix"]
        VtkMatrix_4x4 = (
            self.Vol.matrix_world @ TransformMatrix.inverted() @ VtkTransform_4x4
        )

        VtkMatrix = list(np.array(VtkMatrix_4x4).ravel())

        Thikness = 1

        SegmentTreshold = self.SegmentsDict[Segment]["Treshold"]
        SegmentColor = self.SegmentsDict[Segment]["Color"]
        SegmentStlPath = join(UserProjectDir, f"{Segment}_SEGMENTATION.stl")

        # Convert Hu treshold value to 0-255 UINT8 :
        Treshold255 = HuTo255(
            Hu=SegmentTreshold, Wmin=OdentConstants.WMIN, Wmax=OdentConstants.WMAX
        )
        if Treshold255 == 0:
            Treshold255 = 1
        elif Treshold255 == 255:
            Treshold255 = 254

        ############### step 2 : Extracting mesh... #########################
        # print("Extracting mesh...")

        vtkImage = sitkTovtk(sitkImage=Image3D)

        ExtractedMesh = vtk_MC_Func(vtkImage=vtkImage, Treshold=Treshold255)
        Mesh = ExtractedMesh

        self.step2 = tpc()
        self.TimingDict["Mesh Extraction Time"] = self.step2 - self.step1
        print(f"{Segment} Mesh Extraction Finished")

        ############### step 3 : mesh Smoothing 1... #########################
        message = [f"Smoothing {Segment} segment ..."]
        message_queue.put(message)
        SmthIter = 1
        SmoothedMesh1 = vtkSmoothMesh(
            q=None,
            mesh=Mesh,
            Iterations=SmthIter,
            step="Mesh Smoothing 2",
            start=0.79,
            finish=0.82,
        )
        Mesh = SmoothedMesh1

        self.step3 = tpc()
        self.TimingDict["Mesh Smoothing 1 Time"] = self.step3 - self.step1
        print(f"{Segment} Mesh Smoothing 1 Finished")

        # ############### step 4 : mesh Smoothing... #########################

        SmthIter = 10
        SmoothedMesh2 = vtkWindowedSincPolyDataFilter(
            q=None,
            mesh=Mesh,
            Iterations=SmthIter,
            step="Mesh Smoothing 1",
            start=0.76,
            finish=0.78,
        )

        self.step4 = tpc()
        self.TimingDict["Mesh Smoothing 2 Time"] = self.step4 - self.step3
        print(f"{Segment} Mesh Smoothing 2 Finished")
        Mesh = SmoothedMesh2

        ############### step 5 : mesh Reduction... #########################
        message = [f"Decimating {Segment} segment ..."]
        message_queue.put(message)
        polysCount = Mesh.GetNumberOfPolys()
        polysLimit = 1000000
        if polysCount > polysLimit:

            Reduction = round(1 - (polysLimit / polysCount), 2)
            # Reduction = 0.5
            ReductedMesh = vtkMeshReduction(
                q=None,
                mesh=Mesh,
                reduction=Reduction,
                step="Mesh Reduction",
                start=0.11,
                finish=0.75,
            )
            Mesh = ReductedMesh

        self.step5 = tpc()
        self.TimingDict["Mesh Reduction Time"] = self.step5 - self.step4
        print(f"{Segment} Mesh Reduction Finished")

        ############### step 6 : Set mesh orientation... #########################
        TransformedMesh = vtkTransformMesh(
            mesh=Mesh,
            Matrix=VtkMatrix,
        )
        self.step6 = tpc()
        self.TimingDict["Mesh Orientation"] = self.step6 - self.step5
        print(f"{Segment} Mesh Orientation Finished")
        Mesh = TransformedMesh

        ############### step 7 : exporting mesh stl... #########################
        message = [f"Exporting {Segment} segment ..."]
        message_queue.put(message)
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(Mesh)
        writer.SetFileTypeToBinary()
        writer.SetFileName(SegmentStlPath)
        writer.Write()

        self.step7 = tpc()
        self.TimingDict["Mesh Export"] = self.step7 - self.step6
        print(f"{Segment} Mesh Export Finished")
        self.Exported.put([Segment, SegmentStlPath, SegmentColor])

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        return (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.VOXEL_OBJECT_TYPE
        )

    def execute(self, context):
        global message_queue

        self.counter_start = tpc()

        ODENT_Props = bpy.context.scene.ODENT_Props

        self.Soft = ODENT_Props.SoftBool
        self.Bone = ODENT_Props.BoneBool
        self.Teeth = ODENT_Props.TeethBool

        self.SoftTresh = ODENT_Props.SoftTreshold
        self.BoneTresh = ODENT_Props.BoneTreshold
        self.TeethTresh = ODENT_Props.TeethTreshold

        self.SoftSegmentColor = ODENT_Props.SoftSegmentColor
        self.BoneSegmentColor = ODENT_Props.BoneSegmentColor
        self.TeethSegmentColor = ODENT_Props.TeethSegmentColor

        self.SegmentsDict = {
            "Soft": {
                "State": self.Soft,
                "Treshold": self.SoftTresh,
                "Color": self.SoftSegmentColor,
            },
            "Bone": {
                "State": self.Bone,
                "Treshold": self.BoneTresh,
                "Color": self.BoneSegmentColor,
            },
            "Teeth": {
                "State": self.Teeth,
                "Treshold": self.TeethTresh,
                "Color": self.TeethSegmentColor,
            },
        }

        ActiveSegmentsList = [k for k, v in self.SegmentsDict.items() if v["State"]]

        if not ActiveSegmentsList:
            message = [
                " Please check at least 1 segmentation ! ",
                "(Soft - Bone - Teeth)",
            ]
            ODENT_GpuDrawText(message)
            sleep(3)
            ODENT_GpuDrawText()
            return {"CANCELLED"}

        message = [
            "Dicom segmentation processing ...",
            f"Active Segment(s) : {', '.join(ActiveSegmentsList)}",
        ]
        ODENT_GpuDrawText(message)
        sleep(1)

        self.Vol = context.object
        self.Preffix = self.Vol.name[:6]
        DcmInfoDict = eval(ODENT_Props.DcmInfo)
        self.DcmInfo = DcmInfoDict[self.Preffix]
        self.Nrrd255Path = AbsPath(self.DcmInfo["Nrrd255Path"])

        if not exists(self.Nrrd255Path):

            message = [" 3D Image File not Found in Project Folder ! "]
            ODENT_GpuDrawText(message)
            sleep(3)
            ODENT_GpuDrawText()
            return {"CANCELLED"}

        ############### step 1 : Reading DICOM #########################
        self.step1 = tpc()
        self.TimingDict["Read DICOM"] = self.step1 - self.counter_start
        print(f"step 1 : Read DICOM ({self.step1-self.counter_start})")

        Image3D = sitk.ReadImage(self.Nrrd255Path)
        target_spacing = 0.3
        Sp = Image3D.GetSpacing()
        if Sp[0] < target_spacing:
            ResizedImage, _, _ = ResizeImage(
                sitkImage=Image3D, target_spacing=target_spacing
            )
            Image3D = ResizedImage

        ############### step 2 : Dicom To Stl Threads #########################

        self.MeshesCount = len(ActiveSegmentsList)
        Imported_Meshes = []
        Threads = [
            threading.Thread(
                target=self.DicomToStl,
                args=[Segment, Image3D],
                daemon=True,
            )
            for Segment in ActiveSegmentsList
        ]
        print(f"segments list : {ActiveSegmentsList}")
        for t in Threads:
            t.start()
        count = 0
        while count < self.MeshesCount:
            if not self.message_queue.empty():
                message = self.message_queue.get()
                ODENT_GpuDrawText(message_list=message)
                sleep(1)
            if not self.Exported.empty():
                (
                    Segment,
                    SegmentStlPath,
                    SegmentColor,
                ) = self.Exported.get()
                for i in range(10):
                    if not exists(SegmentStlPath):
                        sleep(0.1)
                    else:
                        break
                message = [f"{Segment} Mesh import ..."]
                ODENT_GpuDrawText(message_list=message)
                obj = self.ImportMeshStl(Segment, SegmentStlPath, SegmentColor)
                Imported_Meshes.append(obj)
                # os.remove(SegmentStlPath)
                count += 1
            else:
                sleep(0.1)
        for t in Threads:
            t.join()

        for obj in Imported_Meshes:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            for i in range(3):
                obj.lock_location[i] = True
                obj.lock_rotation[i] = True
                obj.lock_scale[i] = True

        bpy.ops.object.select_all(action="DESELECT")
        for obj in Imported_Meshes:
            child_of = obj.constraints.new("CHILD_OF")
            child_of.target = self.Vol
            child_of.use_scale_x = False
            child_of.use_scale_y = False
            child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")

        self.counter_finish = tpc()
        self.TimingDict["Total Time"] = self.counter_finish - self.counter_start

        print(self.TimingDict)
        area3D, space3D, region_3d = CtxOverride(context)
        space3D.shading.type = "SOLID"
        space3D.shading.show_specular_highlight = False
        # space3D.shading.background_type = 'WORLD'
        space3D.shading.color_type = "TEXTURE"
        space3D.shading.light = "STUDIO"
        space3D.shading.studio_light = "paint.sl"
        if space3D.shading.show_xray:
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):

                bpy.ops.view3d.toggle_xray()

        self.Vol.hide_set(True)
        message = [" Dicom Segmentation Finished ! "]
        ODENT_GpuDrawText(message_list=message)
        sleep(2)
        ODENT_GpuDrawText()
        bpy.ops.wm.save_mainfile()

        return {"FINISHED"}


# implant:
class ODENT_OT_AddImplant(bpy.types.Operator):
    """Add Implant"""

    bl_idname = "wm.odent_add_implant"
    bl_label = "ADD IMPLANT"
    bl_options = {"REGISTER", "UNDO"}
    
    implant_diameter: FloatProperty(
        name="Diameter",
        default=4.0,
        min=0.0,
        max=7.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Implant Diameter",
    )  # type: ignore
    implant_lenght: FloatProperty(
        name="Lenght",
        default=10.0,
        min=0.0,
        max=20.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Implant Lenght",
    )  # type: ignore
    
    # tooth_number: IntProperty(
    #     name="Tooth Number", default=11, min=11, max=48, description="Tooth Number"
    # )  # type: ignore

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        # Check if the active object is a Slices Pointer
        return (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        )

    # def draw(self, context):
    #     layout = self.layout
    #     box = layout.box()
    #     box.alignment = "EXPAND"
    #     row = box.row()
    #     row = box.row()
    #     row.prop(self, "tooth_number")
    #     row = box.row()
    #     row.prop(self, "implant_diameter")
    #     row = box.row()
    #     row.prop(self, "implant_lenght")
    
    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.alignment = "EXPAND"
        box.label(text="Select tooth number :")
        
        upper,lower= self.props.get_upper_and_lower_teeth(context)
        
        row = box.row(align=True)
        for t_up in upper:
            row.prop_enum(self.props, "tooth_number", t_up)

        row = box.row(align=True)
        for t_low in lower:
            row.prop_enum(self.props, "tooth_number", t_low)
            
        row = box.row()
        row.prop(self, "implant_diameter")
        row = box.row()
        row.prop(self, "implant_lenght")
    
    def implant_exists_for_same_tooth_number(self, context, tooth_number=None):
        tooth_number = tooth_number or self.props.tooth_number
        implant_tag = OdentConstants.ODENT_TYPE_TAG
        implant_type = OdentConstants.ODENT_IMPLANT_TYPE
        remove_code_tag = OdentConstants.ODENT_IMPLANT_REMOVE_CODE_TAG
        return any(
            obj
            for obj in context.scene.objects
            if obj.get(implant_tag) == implant_type and obj.get(remove_code_tag) == tooth_number
        )
    
        
    def execute(self, context):
        tooth_number = self.props.tooth_number
        
        # NOTE : check if implant with same tooth number already exists :
        if self.implant_exists_for_same_tooth_number(context, tooth_number):
            message = [f"implant number {tooth_number} already exists!"]
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.orange, sleep_time=2
            )
            return {"CANCELLED"}
        
        # NOTE : append implant object to scene :
        
        preffix = OdentConstants.ODENT_IMPLANT_NAME_PREFFIX
        implant_name = f"{preffix}_{tooth_number}"
        implant_coll_name = OdentConstants.ODENT_IMPLANT_COLLECTION_NAME
        implant = AppendObject("implant")
        implant.name = implant_name
        implant.show_name = True
        implant.dimensions = Vector(
            (self.implant_diameter, self.implant_diameter, self.implant_lenght)
        )
        
        # NOTE : append to implant collection 
        MoveToCollection(implant, implant_coll_name)
        hide_collection(False, implant_coll_name)
        
        # NOTE : set implant custom props, position :
        type_tag = OdentConstants.ODENT_TYPE_TAG
        implant_type = OdentConstants.ODENT_IMPLANT_TYPE
        remove_code_tag = OdentConstants.ODENT_IMPLANT_REMOVE_CODE_TAG
        implant[type_tag] = implant_type
        implant[remove_code_tag] = tooth_number
        
        implant.matrix_world = self.pointer.matrix_world @ implant.matrix_world
           
        # NOTE : Set implant geometry node group parameters :
        
        #    > ensure implant geometry node group exists
        implant_geonode_name = OdentConstants.ODENT_IMPLANT_GEONODE_NAME
        mod_name = OdentConstants.ODENT_IMPLANT_MODIFIER_NAME
        
        implant_geonode_group_data =  bpy.data.node_groups.get(implant_geonode_name)
        if not implant_geonode_group_data:
            implant_geonode_group_data = append_group_nodes(implant_geonode_name)
        
        #   > add geometry node modifier to implant ,
        #   > assign new node group 
        #     and set the implant_ng as node group for the modifier :
        mod = implant.modifiers.new(name=mod_name, type="NODES")
        ng = implant_geonode_group_data.copy()
        ng.name = f"{implant_geonode_name}_{tooth_number}"
        mod.node_group = ng

        

        teeth_order = OdentConstants.ODENT_IMPLANT_GN_TEETH_ORDER

        index = teeth_order.index(int(tooth_number))

        set_socket_value_geonodes(mod, "Tooth number", index)
        set_socket_value_geonodes(mod, "Implant Diameter", self.implant_diameter)
        set_socket_value_geonodes(mod, "Implant  Length", self.implant_lenght)

        mod.show_viewport = False
        mod.show_viewport = True
        
        # NOTE : lock implant to pointer :
        remove_pointer_lock()
        bpy.ops.object.select_all(action="DESELECT")
        implant.select_set(True)
        context.view_layer.objects.active = implant
        bpy.ops.wm.odent_lock_object_to_pointer()

        bpy.ops.object.select_all(action="DESELECT")
        self.pointer.select_set(True)
        context.view_layer.objects.active = self.pointer
        return {"FINISHED"}

    def invoke(self, context, event):
        self.props = context.scene.ODENT_Props
        self.pointer = context.object
        
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=800)


class ODENT_OT_RemoveImplant(bpy.types.Operator):
    """Remove Implant and linked components"""

    bl_idname = "wm.odent_remove_implant"
    bl_label = "REMOVE IMPLANT"

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        return (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.ODENT_IMPLANT_TYPE
        )

    def execute(self, context):

        implant = context.object
        remove_code = implant.get(OdentConstants.ODENT_IMPLANT_REMOVE_CODE_TAG)
        for o in context.scene.objects:
            if o.get(OdentConstants.ODENT_IMPLANT_REMOVE_CODE_TAG) == remove_code:
                bpy.data.objects.remove(o)

        txt = [f"Odent implant {remove_code} removed."]
        ODENT_GpuDrawText(message_list=txt, rect_color=OdentColors.green, sleep_time=1)
        return {"FINISHED"}


# class ODENT_OT_LockObjectToPointer(bpy.types.Operator):
#     """add child constraint to slices pointer"""

#     bl_idname = "wm.odent_lock_object_to_pointer"
#     bl_label = "Lock to Pointer"

#     @classmethod
#     def poll(cls, context):
#         slices_pointer_check_list = [
#             obj for obj in context.scene.objects if  obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICES_POINTER_TYPE]
#         if not slices_pointer_check_list:
#             return False
#         slices_pointer = slices_pointer_check_list[0]
#         if not context.object or not context.object.select_get() or \
#             context.object.get(OdentConstants.ODENT_TYPE_TAG)\
#              in [OdentConstants.SLICE_PLANE_TYPE, OdentConstants.SLICES_POINTER_TYPE] : return False

#         is_already_locked_to_pointer = [c for c in context.object.constraints if c.type == "CHILD_OF" and c.target == slices_pointer]
#         if is_already_locked_to_pointer:
#             return False
#         return True

#     def execute(self, context):
#         obj = context.object
#         mat = obj.active_material
#         if mat :
#             mat.use_fake_user = True
#             obj[OdentConstants.PREVIOUS_ACTIVE_MAT_NAME_TAG] = mat.name
#         odent_locked_to_pointer_mat = bpy.data.materials.get(
#             OdentConstants.LOCKED_TO_POINTER_MAT_NAME) or bpy.data.materials.new(OdentConstants.LOCKED_TO_POINTER_MAT_NAME)
#         odent_locked_to_pointer_mat.use_fake_user = True
#         odent_locked_to_pointer_mat.diffuse_color = (1, 0, 0, 1)  # red color
#         odent_locked_to_pointer_mat.use_nodes = True
#         # mat_odent_locked.node_tree.nodes["Principled BSDF"].inputs[19].default_value = (
#         #     1, 0, 0, 1)  # red color
#         nodes = odent_locked_to_pointer_mat.node_tree.nodes
#         pbsdf_node = [n for n in nodes if n.type =='BSDF_PRINCIPLED'][0]
#         pbsdf_node.inputs[0].default_value = (
#             1, 0, 0, 1)  # red color
#         bpy.ops.object.material_slot_remove_all()
#         obj.active_material = odent_locked_to_pointer_mat

#         slices_pointer = [
#             obj for obj in context.scene.objects if \
#             obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICES_POINTER_TYPE
#             ][0]

#         hide_collection(False, OdentConstants.SLICES_POINTER_COLLECTION_NAME)
#         hide_object(False, slices_pointer)

#         child_constraint = obj.constraints.new("CHILD_OF")
#         child_constraint.target = slices_pointer
#         child_constraint.use_scale_x = False
#         child_constraint.use_scale_y = False
#         child_constraint.use_scale_z = False
#         lock_object(context.object)
#         context.view_layer.objects.active = slices_pointer
#         bpy.ops.object.mode_set(mode="OBJECT")
#         bpy.ops.object.select_all(action="DESELECT")
#         slices_pointer.select_set(True)

#         return {"FINISHED"}


class ODENT_OT_LockObjectToPointer(bpy.types.Operator):
    """add child constraint to slices pointer"""

    bl_idname = "wm.odent_lock_object_to_pointer"
    bl_label = "Lock to Pointer"
    bl_options = {"REGISTER", "UNDO"}


    @classmethod
    def poll(cls, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_check_list:
            return False
        slices_pointer = slices_pointer_check_list[0]
        if not context.object:
            return False
        if not context.object.select_get():
            return False
        if (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICE_PLANE_TYPE
        ):
            return False
        if (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ):
            return False
        is_already_locked_to_pointer = [
            c
            for c in context.object.constraints
            if c.type == "CHILD_OF" and c.target == slices_pointer
        ]
        if is_already_locked_to_pointer:
            return False
        return True

    def execute(self, context):
        print("locking to pointer   ...")
        obj = context.object
        mat_red = bpy.data.materials.get(OdentConstants.LOCKED_TO_POINTER_MAT_NAME)
        if not mat_red:
            mat_red = bpy.data.materials.new(OdentConstants.LOCKED_TO_POINTER_MAT_NAME)
            mat_red.use_fake_user = True
            mat_red.diffuse_color = (1, 0, 0, 1)  # red color
            mat_red.use_nodes = True
            nodes = mat_red.node_tree.nodes
            pbsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
            pbsdf_node.inputs[0].default_value = (1, 0, 0, 1)  # red color
        
        
        if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.ODENT_IMPLANT_TYPE:
            mod = obj.modifiers.get(OdentConstants.ODENT_IMPLANT_MODIFIER_NAME)
            if mod :
                ng = mod.node_group
                mat = ng.nodes["Set Material"].inputs[2].default_value
                obj[OdentConstants.PREVIOUS_MAT_NAME_TAG] = mat.name
                ng.nodes["Set Material"].inputs[2].default_value = mat_red
        else :
            mat = obj.active_material
            if mat :
                obj[OdentConstants.PREVIOUS_MAT_NAME_TAG] = mat.name
                
            obj.active_material = mat_red
       
        
            
           
        slices_pointer = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ][0]

        hide_collection(False, OdentConstants.SLICES_POINTER_COLLECTION_NAME)
        hide_object(False, slices_pointer)

        child_constraint = obj.constraints.new("CHILD_OF")
        child_constraint.target = slices_pointer
        child_constraint.use_scale_x = False
        child_constraint.use_scale_y = False
        child_constraint.use_scale_z = False
        lock_object(context.object)
        context.view_layer.objects.active = slices_pointer
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        slices_pointer.select_set(True)

        return {"FINISHED"}


class ODENT_OT_UnlockObjectFromPointer(bpy.types.Operator):
    """remove child constraint to slices pointer"""

    bl_idname = "wm.odent_unlock_object_from_pointer"
    bl_label = "Unlock from Pointer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_check_list:
            cls.poll_message_set("no slices pointer found in scene !")
            return False
        slices_pointer = slices_pointer_check_list[0]
        if not context.object:
            cls.poll_message_set("no active object !")
            return False
        if not context.object.select_get():
            cls.poll_message_set("object not selected !")
            return False
        if (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICE_PLANE_TYPE
        ):
            cls.poll_message_set("slice plane can't be unlocked from pointer !")
            return False
        if (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ):
            cls.poll_message_set("pointer can't be unlocked from itself !")
            return False

        is_already_locked_to_pointer = [
            c
            for c in context.object.constraints
            if c.type == "CHILD_OF" and c.target == slices_pointer
        ]
        if not is_already_locked_to_pointer:
            cls.poll_message_set("object can't be unlocked from pointer : object is not locked initially!")
            return False
        return True

    def execute(self, context):

        obj = context.object
        previous_mat_name = obj.get(OdentConstants.PREVIOUS_MAT_NAME_TAG)
        if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.ODENT_IMPLANT_TYPE:
            if previous_mat_name and bpy.data.materials.get(previous_mat_name):
                mod = obj.modifiers.get(OdentConstants.ODENT_IMPLANT_MODIFIER_NAME)
                if mod and mod.node_group:
                    ng = mod.node_group
                    ng.nodes["Set Material"].inputs[2].default_value = bpy.data.materials[previous_mat_name]
        else :
            print(f"""
                  unlocking from pointer :
                  obj : {obj.name} , 
                  previous_mat_name : {previous_mat_name} , 
                  previous material exists : {bpy.data.materials.get(previous_mat_name) is not None if previous_mat_name else False} ,
                  current mat : {obj.active_material.name if obj.active_material else None}
                  """)
            if previous_mat_name and bpy.data.materials.get(previous_mat_name):
                obj.active_material = bpy.data.materials[previous_mat_name]
            else :
                for slot in obj.material_slots:
                    bpy.ops.object.material_slot_remove()
            
        slices_pointer = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ][0]
        hide_collection(False, OdentConstants.SLICES_POINTER_COLLECTION_NAME)
        hide_object(False, slices_pointer)

        for c in obj.constraints:
            if c.type == "CHILD_OF" and c.target == slices_pointer:
                bpy.ops.constraint.apply(constraint=c.name)
                
        # unlock_object(context.object)

        # # lock implant scale!
        # obj.lock_scale = (True, True, True)

        bpy.ops.object.select_all(action="DESELECT")
        slices_pointer.select_set(True)
        context.view_layer.objects.active = slices_pointer

        return {"FINISHED"}


class ODENT_OT_ImplantToPointer(bpy.types.Operator):
    """move implant to pointer and add child constraint to slices pointer"""

    bl_idname = "wm.odent_implant_to_pointer"
    bl_label = "Implant to Pointer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if (
            not context.object
            or not context.object.select_get()
            or not context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.ODENT_IMPLANT_TYPE
        ):
            return False

        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_check_list:
            return False

        return True

    def execute(self, context):
        obj = context.object
        slices_pointer = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ][0]

        obj.matrix_world[:3] = slices_pointer.matrix_world[:3]

        return {"FINISHED"}


class ODENT_OT_ObjectToPointer(bpy.types.Operator):
    """move and align object to pointer"""

    bl_idname = "wm.odent_object_to_pointer"
    bl_label = "To Pointer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_check_list:
            return False

        if not (context.object and context.object.select_get()):
            return False

        if context.object.get(OdentConstants.ODENT_TYPE_TAG) in (
            OdentConstants.SLICES_POINTER_TYPE,
            OdentConstants.SLICE_PLANE_TYPE,
        ):
            return False
        return True

    def execute(self, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        
        slices_pointer = slices_pointer_check_list[0]

        context.object.matrix_world[:3] = slices_pointer.matrix_world[:3]
        return {"FINISHED"}


class ODENT_OT_PointerToImplant(bpy.types.Operator):
    """move pointer to implant and add child constraint to slices pointer"""

    bl_idname = "wm.odent_pointer_to_implant"
    bl_label = "Pointer to Implant"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_check_list:
            return False

        if not context.object or not context.object.select_get():
            return False

        if not context.object.get(OdentConstants.ODENT_TYPE_TAG) in [
            OdentConstants.ODENT_IMPLANT_TYPE,
            OdentConstants.FIXING_SLEEVE_TYPE,
        ]:
            return False

        return True

    def execute(self, context):
        obj = context.object
        slices_pointer = [
            o
            for o in context.scene.objects
            if o.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ][0]
        hide_collection(False, OdentConstants.SLICES_POINTER_COLLECTION_NAME)
        hide_object(False, slices_pointer)

        for o in context.scene.objects:
            if o.get(OdentConstants.ODENT_TYPE_TAG) in (
                OdentConstants.SLICE_PLANE_TYPE,
                OdentConstants.SLICES_POINTER_TYPE,
            ):
                continue
            cp = [
                c
                for c in o.constraints
                if c.type == "CHILD_OF" and c.target == slices_pointer
            ]
            if cp:
                try:
                    bpy.ops.object.select_all(action="DESELECT")
                    o.select_set(True)
                    context.view_layer.objects.active = o
                    bpy.ops.wm.odent_unlock_object_from_pointer()
                    break
                except:
                    continue
        # Move pointer to implant
        slices_pointer.matrix_world[:3] = obj.matrix_world[:3]

        # Deselect all but pointer
        context.view_layer.objects.active = slices_pointer
        bpy.ops.object.select_all(action="DESELECT")
        slices_pointer.select_set(True)

        return {"FINISHED"}


class ODENT_OT_PointerToActive(bpy.types.Operator):
    """move pointer to active object"""

    bl_idname = "wm.odent_pointer_to_active"
    bl_label = "Pointer to Active"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_check_list:
            return False
        if not context.object:
            return False
        if not context.object.select_get():
            return False
        if (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ):
            return False
        return True

    def execute(self, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_check_list[0]
        hide_collection(False, OdentConstants.SLICES_POINTER_COLLECTION_NAME)
        hide_object(False, slices_pointer)

        obj = context.object
        # Move pointer to implant
        slices_pointer.matrix_world[:3] = obj.matrix_world[:3]

        # Deselect all but pointer
        context.view_layer.objects.active = slices_pointer
        bpy.ops.object.select_all(action="DESELECT")
        slices_pointer.select_set(True)

        return {"FINISHED"}


class ODENT_OT_FlyToImplantOrFixingSleeve(bpy.types.Operator):
    """move pointer to previous or next implant or fixing sleeve/pin iteratively"""

    bl_idname = "wm.odent_fly_to_implant_or_fixing_sleeve"
    bl_label = "Fly Next/Previous"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(
        items=set_enum_items(["next", "previous"]),
        description="fly to implant direction",
        default="next",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        if not slices_pointer_check_list:
            return False

        target_objects = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            in [OdentConstants.ODENT_IMPLANT_TYPE, OdentConstants.FIXING_SLEEVE_TYPE]
        ]
        if not target_objects:
            return False
        return True

    def execute(self, context):
        global FLY_IMPLANT_INDEX
        #unhide and activate collections that will be used in this operator
        collections = [
            OdentConstants.SLICES_POINTER_COLLECTION_NAME,
            OdentConstants.ODENT_IMPLANT_COLLECTION_NAME,
            OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME,
        ]
        for collname in collections:
            hide_collection(False, collname)
        
        #get target objects (implants and sleeves) unlock from pointer
        sorted_fly_to_objects = sorted(get_fly_to_objects(), key=lambda x: x.name)

        for obj in sorted_fly_to_objects:
            hide_object(False, obj)

        #ensure nothing is locked to pointer before flying to target object
        remove_pointer_lock()

        #unhide slices pointer 
        slices_pointer_check_list = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        slices_pointer = slices_pointer_check_list[0]
        hide_object(False, slices_pointer)
        
        if len(sorted_fly_to_objects) <  2:
            return {"FINISHED"}

        if FLY_IMPLANT_INDEX is None:
            FLY_IMPLANT_INDEX = 1
        else:
            if self.direction == "next":
                FLY_IMPLANT_INDEX = (
                    FLY_IMPLANT_INDEX + 1
                    if FLY_IMPLANT_INDEX + 1 < len(sorted_fly_to_objects)
                    else 0
                )
            elif self.direction == "previous":
                FLY_IMPLANT_INDEX = (
                    FLY_IMPLANT_INDEX - 1
                    if FLY_IMPLANT_INDEX - 1 >= 0
                    else len(sorted_fly_to_objects) - 1
                )
        move_to_target = sorted_fly_to_objects[FLY_IMPLANT_INDEX]

        slices_pointer.matrix_world[:3] = move_to_target.matrix_world[:3]

        # trigger slices update
        bpy.ops.object.select_all(action="DESELECT")
        context.view_layer.objects.active = slices_pointer
        slices_pointer.select_set(True)

        return {"FINISHED"}


class ODENT_OT_AlignObjectsAxes(bpy.types.Operator):
    """Align objects axes to active object or average axes"""

    bl_idname = "wm.odent_align_objects_axes"
    bl_label = "Align Axes"
    bl_options = {"REGISTER", "UNDO"}

    align_mode_list = ["To Active", "Averrage Axes"]
    alin_mode_prop: EnumProperty(
        items=set_enum_items(align_mode_list),
        description="Implant Align Mode",
        default="To Active",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.select_get()
            and len(context.selected_objects) > 1
        )

    def execute(self, context):
        for obj in self.objects:
            try:
                context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.wm.odent_unlock_object_from_pointer()
            except:
                pass

        if self.alin_mode_prop == "Averrage Axes":
            mean_rotation = np.mean(
                np.array([obj.rotation_euler for obj in self.objects]), axis=0
            )
            for obj in self.objects:
                obj.rotation_euler = mean_rotation

        elif self.alin_mode_prop == "To Active":
            for obj in self.other_objects:
                obj.rotation_euler = self.target_object.rotation_euler

        return {"FINISHED"}

    def invoke(self, context, event):
        self.target_object = context.object
        self.objects = context.selected_objects
        self.other_objects = [
            obj for obj in context.selected_objects if obj != self.target_object
        ]
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_GuideSetComponents(bpy.types.Operator):
    """set guide components"""

    bl_idname = "wm.odent_set_guide_components"
    bl_label = "Set Guide Components"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if not context.selected_objects:
            return False
        return (
            [obj.type == "MESH" for obj in context.selected_objects]
            and context.object.mode == "OBJECT"
            and context.object.select_get()
        )

    def execute(self, context):

        context.scene["odent_guide_components"] = [
            obj.name for obj in context.selected_objects
        ]
        return {"FINISHED"}


class ODENT_OT_GuideSetCutters(bpy.types.Operator):
    """set Guide cutters"""

    bl_idname = "wm.odent_guide_set_cutters"
    bl_label = "Set Guide Cutters"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if not context.selected_objects:
            return False
        return (
            [obj.type == "MESH" for obj in context.selected_objects]
            and context.object.mode == "OBJECT"
            and context.object.select_get()
        )

    def execute(self, context):
        context.scene["odent_guide_cutters"] = [
            obj.name for obj in context.selected_objects
        ]

        return {"FINISHED"}


class ODENT_OT_SplintGuide(bpy.types.Operator):
    """Splint Guide"""

    bl_idname = "wm.odent_add_guide_splint"
    bl_label = "Add Guide Splint"
    bl_options = {"REGISTER", "UNDO"}

    guide_thikness: FloatProperty(
        name="Guide Thikness",
        default=2,
        min=0.1,
        max=10.0,
        step=1,
        precision=2,
        description="MASK BASE 3D PRINTING THIKNESS",
    )  # type: ignore
    splint_type: EnumProperty(
        items=set_enum_items(["Splint Up", "Splint Low"]),
        name="Splint Type",
        description="Splint Type (Up,Low)",
    )  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.alignment = "EXPAND"
        row = layout.row()
        row.prop(self, "splint_type")
        row = layout.row()
        row.prop(self, "guide_thikness")

    def add_cutter_point(self):

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.extrude(mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.curve.select_all(action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(mode="OBJECT")

    def del_cutter_point(self):
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(action="SELECT")
                bpy.ops.curve.handle_type_set(type="AUTOMATIC")
                bpy.ops.curve.select_all(action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass

    def cut_mesh(self, context):
        area3D, space3D, region_3d = CtxOverride(bpy.context)

        for obj in [self.cutter, self.base_mesh_duplicate]:
            obj.hide_select = False
            obj.hide_set(False)
            obj.hide_viewport = False
        context.view_layer.objects.active = self.cutter

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.cyclic_toggle()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

        self.cutter.data.bevel_depth = 0
        self.cutter.data.resolution_u = 3
        bpy.ops.object.select_all(action="DESELECT")
        self.cutter.select_set(True)
        bpy.ops.object.convert(target="MESH")
        self.cutter = context.object

        bpy.ops.object.modifier_add(type="SHRINKWRAP")
        self.cutter.modifiers["Shrinkwrap"].target = self.base_mesh_duplicate
        bpy.ops.object.convert(target="MESH")

        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh_duplicate.select_set(True)
        bpy.context.view_layer.objects.active = self.base_mesh_duplicate

        self.base_mesh_duplicate.vertex_groups.clear()
        me = self.base_mesh_duplicate.data

        # initiate a KDTree :
        size = len(me.vertices)
        kd = kdtree.KDTree(size)

        for v_id, v in enumerate(me.vertices):
            kd.insert(v.co, v_id)

        kd.balance()

        CutterCoList = [
            self.base_mesh_duplicate.matrix_world.inverted()
            @ self.cutter.matrix_world
            @ v.co
            for v in self.cutter.data.vertices
        ]
        Closest_VIDs = [kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))]
        CloseState = True
        Loop = ShortestPath(self.base_mesh_duplicate, Closest_VIDs, close=CloseState)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for idx in Loop:
            me.vertices[idx].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.odent.looptools_relax(
            input="selected", interpolation="cubic", iterations="3", regular=True
        )

        # perform cut :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.hide()
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        colist = [
            (self.base_mesh_duplicate.matrix_world @ v.co)[2]
            for v in self.base_mesh_duplicate.data.vertices
        ]
        if "Up" in self.splint_type:
            z = max(colist)
        elif "Low" in self.splint_type:
            z = min(colist)

        id = colist.index(z)
        self.base_mesh_duplicate.data.vertices[id].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.mesh.select_linked()
            # bpy.ops.mesh.select_all(action='INVERT')
            bpy.ops.mesh.delete(type="VERT")
            bpy.ops.mesh.reveal()
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.data.objects.remove(self.cutter)
        col = bpy.data.collections["Odent Cutters"]
        bpy.data.collections.remove(col)

        bpy.context.scene.tool_settings.use_snap = False
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
        space3D.overlay.show_outline_selected = True

    def splint(self, context):
        self.splint = context.object
        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH"
        )
        smooth_corrective.iterations = 10
        smooth_corrective.use_only_smooth = True

        solidify = self.splint.modifiers.new(name="Solidify", type="SOLIDIFY")
        solidify.thickness = 1
        solidify.offset = 0

        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH"
        )
        smooth_corrective.iterations = 3
        smooth_corrective.use_only_smooth = True

        remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.2

        displace = self.splint.modifiers.new(name="Displace", type="DISPLACE")
        displace.strength = self.guide_thikness - 0.5
        displace.mid_level = 0

        remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.2

        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH"
        )
        smooth_corrective.iterations = 3
        smooth_corrective.use_only_smooth = True

        bpy.ops.object.convert(target="MESH", keep_original=False)

        bpy.ops.object.material_slot_remove_all()

        mat = bpy.data.materials.get(
            OdentConstants.SPLINT_MAT_NAME
        ) or bpy.data.materials.new(OdentConstants.SPLINT_MAT_NAME)
        mat.diffuse_color = OdentConstants.SPLINT_COLOR_BLUE
        mat.roughness = 0.3
        self.splint.active_material = mat
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        pbsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
        pbsdf_node.inputs[0].default_value = OdentConstants.SPLINT_COLOR_BLUE

        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.view_all(center=True)
            bpy.ops.view3d.view_axis(type="FRONT")

    def make_boolean(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.splint.select_set(True)
        bpy.context.view_layer.objects.active = self.splint
        remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.1
        bpy.ops.object.convert(target="MESH", keep_original=False)

        bool = self.splint.modifiers.new(name="Bool", type="BOOLEAN")
        bool.operation = "DIFFERENCE"
        bool.object = self.bool_model
        bpy.ops.object.convert(target="MESH", keep_original=False)
        bpy.data.objects.remove(self.bool_model)
        for obj in self.start_visible_objects:
            try:
                obj.hide_set(False)
            except:
                pass
        # bpy.ops.object.select_all(action="DESELECT")

    def add_splint_cutter(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(
            radius=1, enter_editmode=False, align="CURSOR"
        )
        # Set cutting_tool name :
        self.cutter = bpy.context.view_layer.objects.active

        # CurveCutter settings :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.curve.dissolve_verts()
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

        bpy.context.object.data.dimensions = "3D"
        bpy.context.object.data.twist_smooth = 3
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.context.object.data.bevel_depth = 0.1
        bpy.context.object.data.bevel_resolution = 6
        bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat = bpy.data.materials.get(
            "Odent_curve_cutter_mat"
        ) or bpy.data.materials.new("Odent_curve_cutter_mat")
        mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
        mat.roughness = 0.3
        bpy.ops.object.mode_set(mode="OBJECT")
        self.cutter.active_material = mat

        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Odent Cutters")

    @classmethod
    def poll(cls, context):

        base_mesh = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        return base_mesh

    def modal(self, context, event):
        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}

        elif event.type in ["LEFTMOUSE", "DEL"] and not self.counter in (0, 1):
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

        elif event.type == "RET" and self.counter == 2:
            if event.value == ("PRESS"):

                message = ["Guide Splint Remeshing ..."]
                ODENT_GpuDrawText(message)
                context.view_layer.objects.active = self.splint
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                self.splint.select_set(True)
                remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
                bpy.ops.object.convert(target="MESH", keep_original=False)

                message = ["FINISHED ./"]
                ODENT_GpuDrawText(message)
                sleep(1)
                ODENT_GpuDrawText()
                return {"FINISHED"}

        elif event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                self.counter += 1
                message = ["Cutting Mesh..."]
                ODENT_GpuDrawText(message)
                self.cut_mesh(context)

                message = [f"Creating Guide Splint {self.splint_suffix} ..."]
                ODENT_GpuDrawText(message)
                self.splint(context)

                bpy.ops.object.mode_set(mode="SCULPT")

                # Set the active brush to Smooth
                bpy.context.tool_settings.sculpt.brush = bpy.data.brushes["Smooth"]

                # Optionally, set the brush strength (default is 0.5)
                bpy.context.tool_settings.sculpt.brush.strength = 0.8

                message = [
                    f"(Optional) : Please smooth Guide Splint and press ENTER ..."
                ]
                ODENT_GpuDrawText(message)
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_splint_cutter(context)
                self.counter += 1
                return {"RUNNING_MODAL"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.odent.message_box("INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh
        bpy.ops.object.duplicate_move()
        self.base_mesh_duplicate = context.object

        smooth_corrective = self.base_mesh_duplicate.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH"
        )
        smooth_corrective.iterations = 10
        smooth_corrective.use_only_smooth = True
        bpy.ops.object.convert(target="MESH", keep_original=False)
        self.base_mesh_duplicate.name = "Guide Splint"
        self.base_mesh_duplicate[OdentConstants.ODENT_TYPE_TAG] = "odent_splint"
        self.base_mesh_duplicate.hide_set(True)

        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        self.splint_suffix = "Up" if "up" in self.splint_type.lower() else "Low"

        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        message = ["please draw Guide border", "when done press ENTER"]
        ODENT_GpuDrawText(message)
        return {"RUNNING_MODAL"}


class ODENT_OT_SplintGuideGeom(bpy.types.Operator):
    """Splint Guide Geo Nodes"""

    bl_idname = "wm.odent_add_guide_splint_geom"
    bl_label = "Add Guide Splint"
    bl_options = {"REGISTER", "UNDO"}

    guide_thikness: FloatProperty(
        name="Guide Thikness",
        default=2,
        min=0.1,
        max=10.0,
        step=1,
        precision=2,
        description="MASK BASE 3D PRINTING THIKNESS",
    )  # type: ignore
    splint_type: EnumProperty(
        items=set_enum_items(["Splint Up", "Splint Low"]),
        name="Splint Type",
        description="Splint Type (Up,Low)",
    )  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.alignment = "EXPAND"
        row = layout.row()
        row.prop(self, "splint_type")
        row = layout.row()
        row.prop(self, "guide_thikness")

    def add_cutter_point(self):
        area3D, space3D, region_3d = CtxOverride(bpy.context)
        bpy.context.view_layer.objects.active = self.cutter
        bpy.ops.object.mode_set(mode="EDIT")

        with bpy.context.temp_override(
            active_object=self.cutter, area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.curve.extrude(mode="INIT")
            bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.ops.curve.select_all(action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True

        bpy.ops.object.mode_set(mode="OBJECT")

    def del_cutter_point(self):
        try:
            area3D, space3D, region_3d = CtxOverride(bpy.context)
            bpy.context.view_layer.objects.active = self.cutter

            # with bpy.context.temp_override(active_object=self.cutter,area= area3D, space_data=space3D, region = region_3d):

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(action="SELECT")
                bpy.ops.curve.handle_type_set(type="AUTOMATIC")
                bpy.ops.curve.select_all(action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass

    def cut_mesh(self, context):
        area3D, space3D, region_3d = CtxOverride(bpy.context)

        self.cutter.hide_select = False
        self.cutter.hide_set(False)
        self.cutter.hide_viewport = False

        context.view_layer.objects.active = self.cutter

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.cyclic_toggle()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

        self.base_mesh_duplicate.hide_select = False
        self.base_mesh_duplicate.hide_set(False)
        self.base_mesh_duplicate.hide_viewport = False

        self.cutter.data.bevel_depth = 0
        self.cutter.data.resolution_u = 3
        bpy.ops.object.select_all(action="DESELECT")
        self.cutter.select_set(True)
        bpy.ops.object.convert(target="MESH")
        self.cutter = context.object

        bpy.ops.object.modifier_add(type="SHRINKWRAP")
        self.cutter.modifiers["Shrinkwrap"].target = self.base_mesh_duplicate
        bpy.ops.object.convert(target="MESH")

        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh_duplicate.select_set(True)
        bpy.context.view_layer.objects.active = self.base_mesh_duplicate

        self.base_mesh_duplicate.vertex_groups.clear()
        me = self.base_mesh_duplicate.data

        # initiate a KDTree :
        size = len(me.vertices)
        kd = kdtree.KDTree(size)

        for v_id, v in enumerate(me.vertices):
            kd.insert(v.co, v_id)

        kd.balance()

        CutterCoList = [
            self.base_mesh_duplicate.matrix_world.inverted()
            @ self.cutter.matrix_world
            @ v.co
            for v in self.cutter.data.vertices
        ]
        Closest_VIDs = [kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))]
        CloseState = True
        Loop = ShortestPath(self.base_mesh_duplicate, Closest_VIDs, close=CloseState)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for idx in Loop:
            me.vertices[idx].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.odent.looptools_relax(
            input="selected", interpolation="cubic", iterations="3", regular=True
        )

        # perform cut :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.hide()
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        colist = [
            (self.base_mesh_duplicate.matrix_world @ v.co)[2]
            for v in self.base_mesh_duplicate.data.vertices
        ]
        if "Up" in self.splint_type:
            z = max(colist)
        elif "Low" in self.splint_type:
            z = min(colist)

        id = colist.index(z)
        self.base_mesh_duplicate.data.vertices[id].select = True

        bpy.ops.object.mode_set(mode="EDIT")

        bpy.ops.mesh.select_linked()
        # bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.mesh.reveal()
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.data.objects.remove(self.cutter)
        col = bpy.data.collections["Odent Cutters"]
        bpy.data.collections.remove(col)

        bpy.context.scene.tool_settings.use_snap = False
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
        space3D.overlay.show_outline_selected = True

    def splint(self, context):
        area3D, space3D, region_3d = CtxOverride(context)

        self.splint = context.object
        # smooth_corrective = self.splint.modifiers.new(
        #     name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        # smooth_corrective.iterations = 5
        # smooth_corrective.use_only_smooth = True
        # smooth_corrective.use_pin_boundary = True

        gn = append_group_nodes(OdentConstants.ODENT_VOLUME_NODE_NAME)
        mesh_to_volume(self.splint, gn, offset=self.guide_thikness)

        # remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        # remesh.voxel_size = 0.2

        # smooth_corrective = self.splint.modifiers.new(
        #     name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        # smooth_corrective.iterations = 10
        # smooth_corrective.use_only_smooth = True
        # smooth_corrective.use_pin_boundary = True

        bpy.ops.object.select_all(action="DESELECT")
        self.splint.select_set(True)
        bpy.context.view_layer.objects.active = self.splint
        bpy.ops.object.convert(target="MESH", keep_original=False)

        bpy.ops.object.material_slot_remove_all()

        mat = bpy.data.materials.get(
            OdentConstants.SPLINT_MAT_NAME
        ) or bpy.data.materials.new(OdentConstants.SPLINT_MAT_NAME)
        mat.diffuse_color = OdentConstants.SPLINT_COLOR_BLUE
        mat.roughness = 0.3
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        pbsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
        pbsdf_node.inputs[0].default_value = OdentConstants.SPLINT_COLOR_BLUE
        self.splint.active_material = mat

        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):

            bpy.ops.view3d.view_all(center=True)
            bpy.ops.view3d.view_axis(type="FRONT")

    def add_splint_cutter(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):

            # Prepare scene settings :
            bpy.context.scene.tool_settings.use_snap = True
            bpy.context.scene.tool_settings.snap_elements = {"FACE"}

            # ....Add Curve ....... :
            bpy.ops.curve.primitive_bezier_curve_add(
                radius=1, enter_editmode=False, align="CURSOR"
            )
            # Set cutting_tool name :
            self.cutter = bpy.context.view_layer.objects.active

            # CurveCutter settings :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):
                bpy.ops.curve.dissolve_verts()
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

            bpy.context.object.data.dimensions = "3D"
            bpy.context.object.data.twist_smooth = 3
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.context.object.data.bevel_depth = 0.1
            bpy.context.object.data.bevel_resolution = 6
            bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
            bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
            bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
            bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
            bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = (
                True
            )

            # Add color material :
            mat = bpy.data.materials.get(
                "Odent_curve_cutter_mat"
            ) or bpy.data.materials.new("Odent_curve_cutter_mat")
            mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
            mat.roughness = 0.3
            bpy.ops.object.mode_set(mode="OBJECT")
            self.cutter.active_material = mat

            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
            space3D.overlay.show_outline_selected = False

            shrinkwrap = self.cutter.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
            shrinkwrap.target = self.base_mesh
            shrinkwrap.wrap_mode = "ABOVE_SURFACE"
            shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Odent Cutters")

    @classmethod
    def poll(cls, context):

        base_mesh = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        return base_mesh

    def modal(self, context, event):
        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}

        elif event.type in ["LEFTMOUSE", "DEL"] and not self.counter in (0, 1):
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                ODENT_GpuDrawText(
                    message_list=message, rect_color=OdentColors.red, sleep_time=1
                )
                return {"CANCELLED"}

        elif event.type == "RET" and self.counter == 2:
            if event.value == ("PRESS"):

                context.view_layer.objects.active = self.splint
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                self.splint.select_set(True)

                message = ["Guide Splint Remeshing ..."]
                ODENT_GpuDrawText(message)

                remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
                remesh.voxel_size = 0.3
                bpy.ops.object.convert(target="MESH", keep_original=False)

                message = ["FINISHED."]
                ODENT_GpuDrawText(
                    message_list=message, rect_color=OdentColors.green, sleep_time=1
                )
                return {"FINISHED"}

        elif event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                self.counter += 1
                message = ["Cutting Mesh..."]
                ODENT_GpuDrawText(message)
                self.cut_mesh(context)

                message = [f"Creating Guide Splint {self.splint_suffix} ..."]
                ODENT_GpuDrawText(message)
                self.splint(context)
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.object.mode_set(mode="SCULPT")
                    bpy.ops.brush.asset_activate(
                        asset_library_type="ESSENTIALS",
                        asset_library_identifier="",
                        relative_asset_identifier="brushes/essentials_brushes-mesh_sculpt.blend/Brush/Smooth",
                    )
                    brush = bpy.data.brushes["Smooth"]
                    brush.strength = 0.5

                message = [
                    f"(Optional) : Please smooth Guide Splint and press ENTER ..."
                ]
                ODENT_GpuDrawText(message)
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_splint_cutter(context)
                self.counter += 1
                return {"RUNNING_MODAL"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            self.start_objects = bpy.data.objects[:]
            self.start_collections = bpy.data.collections[:]
            self.start_visible_objects = bpy.context.visible_objects[:]
            self.scn = context.scene
            self.counter = 0

            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:
            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.odent.message_box("INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        message = ["Please wait, Preparing base mesh..."]
        ODENT_GpuDrawText(message)
        guide_components_coll_name = OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
        guide_components_coll = add_collection(guide_components_coll_name)
        hide_collection(False, guide_components_coll_name)

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh
        bpy.ops.object.duplicate_move()
        self.base_mesh_duplicate = context.object
        bpy.ops.wm.odent_decimate(decimate_ratio=0.2)  # 0.1
        smooth_corrective = self.base_mesh_duplicate.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH"
        )
        smooth_corrective.iterations = 2  # 10
        smooth_corrective.use_only_smooth = True
        bpy.ops.object.convert(target="MESH", keep_original=False)
        self.base_mesh_duplicate.name = "_ADD_Guide_Splint"
        self.base_mesh_duplicate[OdentConstants.ODENT_TYPE_TAG] = "odent_splint"
        MoveToCollection(self.base_mesh_duplicate, guide_components_coll.name)
        self.base_mesh_duplicate.hide_set(True)

        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        self.splint_suffix = "Up" if "up" in self.splint_type.lower() else "Low"

        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        message = ["please draw Guide border", "when done press ENTER"]
        ODENT_GpuDrawText(message_list=message)
        return {"RUNNING_MODAL"}


class ODENT_OT_GuideFinaliseGeonodes(bpy.types.Operator):
    """Guide finalise using geonodes boolean"""

    bl_idname = "wm.odent_guide_finalise_geonodes"
    bl_label = "Guide Finalization"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        coll = bpy.data.collections.get(OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME)
        return coll and coll.objects

    def execute(self, context):
        start = tpc()
        coll = bpy.data.collections.get(OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME)
        hide_collection(False, coll.name)

        guide_components = coll.objects
        add_components = [
            obj for obj in guide_components if obj.name.startswith("_ADD_")
        ]

        if not add_components:
            message = ["Cancelled : Can't find Guide _ADD_ Components !"]
            ODENT_GpuDrawText(message)
            sleep(3)
            ODENT_GpuDrawText()
            return {"CANCELLED"}

        # message = ["Making guide components duplicates..."]
        # ODENT_GpuDrawText(message)

        add_coll, cut_coll = finalize_geonodes_make_dup_colls(
            context, guide_components, add_components
        )
        suffix = 1
        for obj in bpy.context.scene.objects:
            if OdentConstants.GUIDE_NAME in obj.name:
                suffix += 1
        guide = AppendObject(OdentConstants.GUIDE_NAME, "Guide Collection")
        guide_coll = bpy.data.collections.get("Guide Collection")
        hide_collection(False, guide_coll.name)

        bpy.ops.object.select_all(action="DESELECT")
        context.view_layer.objects.active = guide
        guide.select_set(True)
        guide.name += f"_{suffix}"
        gn = bpy.data.node_groups.get(OdentConstants.BOOL_NODE)
        gn.nodes["collection_add"].inputs[0].default_value = add_coll
        gn.nodes["collection_cut"].inputs[0].default_value = cut_coll
        # ODENT_GpuDrawText(["Applying nodes..."])
        # bpy.ops.object.convert(target="MESH")
        obj = bpy.context.object
        for mod in obj.modifiers:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        mat = bpy.data.materials.get(
            OdentConstants.SPLINT_MAT_NAME
        ) or bpy.data.materials.new(name=OdentConstants.SPLINT_MAT_NAME)
        mat.diffuse_color = OdentConstants.SPLINT_COLOR_BLUE
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        pbsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
        pbsdf_node.inputs[0].default_value = OdentConstants.SPLINT_COLOR_BLUE
        bpy.ops.object.material_slot_remove_all()
        guide.active_material = mat
        exclude_coll(colname=coll.name)

        for coll in [add_coll, cut_coll]:
            for obj in coll.objects:
                bpy.data.objects.remove(obj)
            bpy.data.collections.remove(coll)

        # Remove boolean_geonode to avoid a state conflict, which causes errors during the second execution of the operator.
        node_tree = bpy.data.node_groups
        bnode = bpy.data.node_groups["boolean_geonode"]
        node_tree.remove(bnode)
        end = tpc()
        os.system("cls") if os.name == "nt" else os.system("clear")
        # ODENT_GpuDrawText(message_list=["Guide Finalised !"],rect_color=OdentColors.green,sleep_time=2)
        odent_log([f"Guide Finalised in {round(end-start)} seconds."])

        return {"FINISHED"}


class ODENT_OT_AddGuideCuttersFromSleeves(bpy.types.Operator):
    """add Guide Cutters From Sleeves"""

    bl_idname = "wm.odent_add_guide_cutters_from_sleeves"
    bl_label = "Guide Cutters From Sleeves"
    bl_options = {"REGISTER", "UNDO"}

    guide_cutters = []

    @classmethod
    def poll(cls, context):
        target = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        sleeves_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            and "odent_sleeve" in obj.get(OdentConstants.ODENT_TYPE_TAG)
        ]
        if not target or not sleeves_checklist:
            return False
        return True

    def modal(self, context, event):
        if not event.type in {"ESC", "RET"}:
            return {"PASS_THROUGH"}
        if event.type == "ESC":
            for sleeve in self.sleeves_checklist:
                sleeve.hide_set(False)
            for cutter in self.guide_cutters:
                bpy.data.objects.remove(cutter, do_unlink=True)
            return {"CANCELLED"}

        if event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                message = ["Cutting Apply ..."]
                ODENT_GpuDrawText(message)
                self.target.hide_set(False)
                self.target.hide_viewport = False
                context.view_layer.objects.active = self.target
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                self.target.select_set(True)
                bpy.ops.object.convert(target="MESH", keep_original=False)
                bpy.data.objects.remove(self.joined_cutter, do_unlink=True)
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")

                for sleeve in self.sleeves_checklist:
                    sleeve.hide_set(False)

                message = ["Finished ."]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"FINISHED"}

        elif event.type == "RET" and self.counter == 0:
            if event.value == ("PRESS"):
                self.counter += 1
                message = ["making Boolean ..."]
                ODENT_GpuDrawText(message)
                self.target.hide_set(False)
                self.target.hide_viewport = False
                context.view_layer.objects.active = self.target
                bpy.ops.object.mode_set(mode="OBJECT")
                self.target.select_set(True)

                remesh = self.target.modifiers.new(name="remesh", type="REMESH")
                remesh.voxel_size = 0.1
                bpy.ops.object.convert(target="MESH", keep_original=False)
                bpy.ops.object.select_all(action="DESELECT")

                for cutter in self.guide_cutters:
                    cutter.hide_set(False)
                    cutter.hide_viewport = False
                    cutter.select_set(True)
                    context.view_layer.objects.active = cutter

                bpy.ops.object.join()
                self.joined_cutter = context.object
                self.joined_cutter.display_type = "WIRE"

                bool = self.target.modifiers.new(name="bool", type="BOOLEAN")
                bool.object = self.joined_cutter
                bool.operation = "DIFFERENCE"

                message = ["Please Control cuttings", "when done press ENTER"]
                ODENT_GpuDrawText(message)
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.counter = 0
            self.sleeves_checklist = [
                obj
                for obj in context.scene.objects
                if obj.get(OdentConstants.ODENT_TYPE_TAG)
                and "odent_sleeve" in obj.get(OdentConstants.ODENT_TYPE_TAG)
            ]

            self.target = context.object
            self.target.hide_set(False)
            self.target.hide_viewport = False
            context.view_layer.objects.active = self.target
            bpy.ops.object.mode_set(mode="OBJECT")
            for sleeve in self.sleeves_checklist:
                sleeve.hide_set(False)
                sleeve.hide_viewport = False
                bpy.ops.object.select_all(action="DESELECT")
                sleeve.select_set(True)
                context.view_layer.objects.active = sleeve
                bpy.ops.object.duplicate_move()
                cutter = context.object
                cutter[OdentConstants.ODENT_TYPE_TAG] = "odent_guide_cutter"
                cutter.display_type = "WIRE"

                cutter.scale *= Vector((1.2, 1.2, 1))
                z_loc = cutter.dimensions.z / 2
                # cutter.location += cutter.matrix_world.to_3x3() @ Vector((0,0,z_loc))
                mat = cutter.material_slots[0].material = bpy.data.materials.get(
                    "odent_guide_cutter_material"
                ) or bpy.data.materials.new(name="odent_guide_cutter_material")
                mat.diffuse_color = (0.8, 0, 0, 1)
                self.guide_cutters.append(context.object)
                # sleeve.hide_set(True)
            area3D, space3D, region_3d = CtxOverride(context)
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):
                bpy.ops.wm.tool_set_by_id(name="builtin.transform")
            # modal operator
            context.window_manager.modal_handler_add(self)
            message = ["please set cutters scale and position", "when done press ENTER"]
            ODENT_GpuDrawText(message)
            return {"RUNNING_MODAL"}


class ODENT_OT_guide_3d_text(bpy.types.Operator):
    """add guide 3D text"""

    bl_label = "3D Text"
    bl_idname = "wm.odent_guide_3d_text"
    bl_options = {"REGISTER", "UNDO"}
    
    
    text_color = [0.0, 0.0, 1.0, 1.0]
    font_size = 3
    add: BoolProperty(default=True)  # type: ignore

    @classmethod
    def poll(cls, context):
        is_valid = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )

        if not is_valid:
            return False
        return True

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        ODENT_GpuDrawText(["Press <TAB> to edit text, <ESC> to cancel, <T> to confirm"])
        ODENT_Props = context.scene.ODENT_Props
        self.target = context.object

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.text_add(enter_editmode=False, align="CURSOR")
        self.text_ob = context.object

        self.text_ob[OdentConstants.ODENT_TYPE_TAG] = "odent_text"
        self.text_ob.data.body = "Guide_"
        self.text_ob.name = "Guide_3d_text"
        if self.add:
            self.text_ob.name = "_ADD_" + self.text_ob.name
        guide_components_coll = add_collection(
            OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
        )
        MoveToCollection(self.text_ob, guide_components_coll.name)

        self.text_ob.data.align_x = "CENTER"
        self.text_ob.data.align_y = "CENTER"
        self.text_ob.data.size = self.font_size
        self.text_ob.location = context.scene.cursor.location
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.view_axis(type="TOP", align_active=True)

        # change curve settings:
        self.text_ob.data.extrude = 1
        self.text_ob.data.bevel_depth = 0.1
        self.text_ob.data.bevel_resolution = 3

        # add SHRINKWRAP modifier :
        shrinkwrap_modif = self.text_ob.modifiers.new("SHRINKWRAP", "SHRINKWRAP")
        shrinkwrap_modif.use_apply_on_spline = True
        shrinkwrap_modif.wrap_method = "PROJECT"
        shrinkwrap_modif.offset = 0
        shrinkwrap_modif.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap_modif.cull_face = "OFF"
        shrinkwrap_modif.use_negative_direction = True
        shrinkwrap_modif.use_positive_direction = True
        shrinkwrap_modif.use_project_z = True
        shrinkwrap_modif.target = self.target

        mat = bpy.data.materials.get("odent_text_mat") or bpy.data.materials.new(
            "odent_text_mat"
        )
        mat.diffuse_color = self.text_color
        mat.roughness = 0.6
        self.text_ob.active_material = mat

        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE_NEAREST"}
        bpy.context.scene.tool_settings.use_snap_align_rotation = True
        bpy.context.scene.tool_settings.use_snap_rotate = True

        # run modal
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not event.type in {"ESC", "T"}:
            return {"PASS_THROUGH"}
        if event.type in {"ESC"}:

            try:
                bpy.data.objects.remove(self.text_ob)
            except:
                pass
            bpy.context.scene.tool_settings.use_snap = False
            area3D, space3D, region_3d = CtxOverride(context)
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
            ODENT_GpuDrawText(["Cancelled"])
            sleep(1)
            ODENT_GpuDrawText()
            return {"CANCELLED"}
        if event.type in {"T"}:
            if event.value == "PRESS":
                if self.text_ob.mode == "EDIT":
                    return {"PASS_THROUGH"}

                self.text_to_mesh(context)

                bpy.context.scene.tool_settings.use_snap = False
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                ODENT_GpuDrawText(["Finished"])
                sleep(1)
                ODENT_GpuDrawText()
                return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def text_to_mesh(self, context):
        ODENT_GpuDrawText(["Text Remesh..."])
        with context.temp_override(active_object=self.text_ob):

            remesh_modif = self.text_ob.modifiers.new("REMESH", "REMESH")
            remesh_modif.voxel_size = 0.05
            bpy.ops.object.convert(target="MESH")


class ODENT_OT_GuideAddComponent(bpy.types.Operator):
    """set Guide cutters"""

    bl_idname = "wm.odent_guide_add_component"
    bl_label = "Add Guide Component"
    bl_options = {"REGISTER", "UNDO"}

    guide_component: EnumProperty(
        name="Guide Component",
        description="Guide Component",
        items=set_enum_items(
            ["Cube", "Sphere", "Cylinder", "3D Text", "Guide Sleeve Cutter"]
        ),
        default="Cube",
    )  # type: ignore
    component_type: EnumProperty(
        name="Component Type",
        description="Component Type",
        items=set_enum_items(["ADD", "CUT"]),
        default="CUT",
    )  # type: ignore
    component_size: FloatProperty(
        description="Component Size", default=5, step=1, precision=2
    )  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "guide_component")
        if self.guide_component in ["3D Text"]:
            layout.prop(self, "component_type")
        elif self.guide_component in ["Cube", "Sphere", "Cylinder"]:
            layout.prop(self, "component_type")
            layout.prop(self, "component_size")

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)
        else:
            self.report({"WARNING"}, "Active space must be a View3d")
            return {"CANCELLED"}

    def execute(self, context):

        if self.guide_component == "3D Text" and (
            not context.object or not context.object.select_get()
        ):
            message = ["Please Select target object and retry"]
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        elif self.guide_component == "Guide Sleeve Cutter":
            sleeves = [
                obj
                for obj in context.selected_objects
                if obj.get(OdentConstants.ODENT_TYPE_TAG)
                == OdentConstants.IMPLANT_SLEEVE_TYPE
            ]
            if not sleeves or not context.object in sleeves:
                message = ["Please Select guide sleeve(s)."]
                ODENT_GpuDrawText(
                    message_list=message, rect_color=OdentColors.red, sleep_time=2
                )
                return {"CANCELLED"}
            else:
                bpy.ops.object.mode_set(mode="OBJECT")
                self.add_guide_sleeve_cutters(context, sleeves)
                return {"FINISHED"}

        self.mat_add = bpy.data.materials.get(
            "mat_component_add"
        ) or bpy.data.materials.new(name="mat_component_add")
        self.mat_add.diffuse_color = [0.0, 1.0, 0.0, 1.0]
        self.mat_cut = bpy.data.materials.get(
            "mat_component_cut"
        ) or bpy.data.materials.new(name="mat_component_cut")
        self.mat_cut.diffuse_color = [1.0, 0.0, 0.0, 1.0]

        self.preffix = ""
        if self.component_type == "ADD":
            self.preffix = "_ADD_"

        message = [
            "Please left click to set the component position",
            "<ENTER> to confirm  <ESC> to cancell",
        ]
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

        ODENT_GpuDrawText(message)

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not event.type in {"ESC", "RET"}:
            return {"PASS_THROUGH"}

        elif event.type in {"ESC"}:
            if event.value == "PRESS":
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                ODENT_GpuDrawText(["Cancelled"])
                sleep(1)
                ODENT_GpuDrawText()
                return {"CANCELLED"}
        elif event.type in {"RET"}:
            if event.value == "RELEASE":
                # bpy.ops.object.select_all(action="DESELECT")
                if self.guide_component == "Cube":
                    bpy.ops.mesh.primitive_cube_add(
                        size=self.component_size, align="CURSOR"
                    )
                    self.object = cube = context.object
                    n = len([o for o in bpy.data.objects if "Cube_Component" in o.name])
                    cube.name = f"{self.preffix}Cube_Component({n+1})"
                    bevel = context.object.modifiers.new(name="bevel", type="BEVEL")
                    bevel.width = 0.3
                    bevel.segments = 3
                    bpy.ops.object.convert(target="MESH", keep_original=False)

                    MoveToCollection(
                        cube, OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
                    )
                    cube.active_material = (
                        self.mat_add if self.component_type == "ADD" else self.mat_cut
                    )
                    area3D, space3D, region_3d = CtxOverride(bpy.context)
                    with bpy.context.temp_override(
                        area=area3D, space_data=space3D, region=region_3d
                    ):
                        bpy.ops.wm.tool_set_by_id(name="builtin.transform")
                    ODENT_GpuDrawText()

                elif self.guide_component == "Sphere":
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=self.component_size, align="CURSOR"
                    )
                    self.object = sphere = context.object
                    n = len(
                        [o for o in bpy.data.objects if "Sphere_Component" in o.name]
                    )
                    sphere.name = f"{self.preffix}Sphere_Component({n+1})"

                    subsurf = context.object.modifiers.new(
                        name="subsurf", type="SUBSURF"
                    )
                    subsurf.levels = 1
                    bpy.ops.object.convert(target="MESH", keep_original=False)

                    MoveToCollection(
                        sphere, OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
                    )
                    sphere.active_material = (
                        self.mat_add if self.component_type == "ADD" else self.mat_cut
                    )
                    area3D, space3D, region_3d = CtxOverride(bpy.context)
                    with bpy.context.temp_override(
                        area=area3D, space_data=space3D, region=region_3d
                    ):
                        bpy.ops.wm.tool_set_by_id(name="builtin.transform")
                    ODENT_GpuDrawText()

                elif self.guide_component == "Cylinder":
                    bpy.ops.mesh.primitive_cylinder_add(
                        radius=self.component_size / 2,
                        depth=self.component_size * 2,
                        align="CURSOR",
                    )
                    self.object = cylinder = context.object
                    n = len(
                        [o for o in bpy.data.objects if "Cylinder_Component" in o.name]
                    )
                    cylinder.name = f"{self.preffix}Cylinder_Component({n+1})"

                    bevel = context.object.modifiers.new(name="bevel", type="BEVEL")
                    bevel.width = 0.3
                    bevel.segments = 3
                    bpy.ops.object.convert(target="MESH", keep_original=False)

                    MoveToCollection(
                        cylinder, OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
                    )
                    cylinder.active_material = (
                        self.mat_add if self.component_type == "ADD" else self.mat_cut
                    )
                    area3D, space3D, region_3d = CtxOverride(bpy.context)
                    with bpy.context.temp_override(
                        area=area3D, space_data=space3D, region=region_3d
                    ):
                        bpy.ops.wm.tool_set_by_id(name="builtin.transform")
                    ODENT_GpuDrawText()

                elif self.guide_component == "3D Text":
                    self.object = None
                    if self.component_type == "CUT":
                        bpy.ops.wm.odent_guide_3d_text("EXEC_DEFAULT", add=False)
                    else:
                        bpy.ops.wm.odent_guide_3d_text("EXEC_DEFAULT", add=True)
                if self.object:
                    # we need to activate the added components that will be placed in the guide components collection,
                    # thus we need to ensure that the collection is visible and not hidden by the user.
                    hide_collection(
                        False, OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
                    )

                    bpy.ops.object.select_all(action="DESELECT")
                    self.object.select_set(True)
                    context.view_layer.objects.active = self.object

                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def add_guide_sleeve_cutters(self, context, sleeves):
        for sleeve in sleeves:
            context.scene.cursor.location = [0, 0, 0]
            bpy.ops.object.select_all(action="DESELECT")
            # context.view_layer.objects.active = sleeve
            # sleeve.select_set(True)
            _radius = sleeve.dimensions.x / 2 + 1
            _depth = sleeve.dimensions.z
            z_trans = _depth - 0.01
            mtx = sleeve.matrix_world.copy()

            bpy.ops.mesh.primitive_cylinder_add(
                vertices=64, radius=_radius, depth=_depth
            )
            sleeve_cutter = context.object
            sleeve_cutter.location.z = _depth / 2
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

            # sleeve_cutter.dimensions = sleeve.dimensions
            sleeve_cutter.matrix_world = mtx

            # bpy.ops.object.duplicate_move()

            # sleeve_cutter = context.object
            # bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

            # sleeve_cutter.scale.x *=2
            # sleeve_cutte_depthr.scale.y *=2

            bpy.ops.transform.translate(
                value=(0, 0, z_trans),
                orient_type="LOCAL",
                constraint_axis=(False, False, True),
            )
            # translate_local(sleeve_cutter, z_trans, "Z")
            sleeve_cutter.lock_location = (True, True, True)
            name = sleeve.name.split("_ADD_")[-1]
            sleeve_cutter.name = f"{name}_cutter"
            mat_cut = bpy.data.materials.get(
                "mat_component_cut"
            ) or bpy.data.materials.new(name="mat_component_cut")
            mat_cut.diffuse_color = [1.0, 0.0, 0.0, 1.0]
            mat_cut.use_nodes = True
            nodes = mat_cut.node_tree.nodes
            bsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
            bsdf_node.inputs[0].default_value = [1.0, 0.0, 0.0, 1.0]
            sleeve_cutter.active_material = mat_cut
            MoveToCollection(
                sleeve_cutter, OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
            )

        bpy.ops.object.select_all(action="DESELECT")

        return


class ODENT_OT_AddCustomSleeveCutter(bpy.types.Operator):
    """add a sleeve cutter from odent asset library"""

    bl_idname = "wm.add_custom_sleeve_cutter"
    bl_label = "Custom Sleeve Cutter"
    bl_options = {"REGISTER", "UNDO"}

    # @classmethod
    # def poll(cls, context):
    #     is_valid = context.object and context.object.select_get(
    #     ) and context.object.type == "MESH" and context.object.mode == 'OBJECT'

    #     if not is_valid :
    #         return False
    #     return True

    def modal(self, context, event):

        if self.can_update:
            message = [
                "Select target implant(s)/Fixing sleeves",
                "select sleeve cutter from Odent Library",
                "<ENTER> : confirm  <ESC> : cancel",
            ]
            ODENT_GpuDrawText(message_list=message)
            self.can_update = False

        if not event.type in {"ESC", "RET"}:
            return {"PASS_THROUGH"}

        elif event.type in {"ESC"}:
            if event.value == "PRESS":
                close_asset_browser(context, area=self.asset_browser_area)
                ODENT_GpuDrawText(["Cancelled"])
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}
        elif event.type in {"RET"}:
            if event.value == "RELEASE":
                result = get_selected_odent_assets(area=self.asset_browser_area)
                success, message, error, directory, filename = result.values()
                if not success:
                    if error == 1:
                        ODENT_GpuDrawText(message=message, rect_color=OdentColors.red)
                        sleep(2)
                        ODENT_GpuDrawText()
                        return {"CANCELLED"}
                    elif error == 2:
                        ODENT_GpuDrawText(message=message, rect_color=OdentColors.red)
                        return {"RUNNING_MODAL"}

                else:

                    objs = context.selected_objects.copy()
                    assets = []
                    bpy.ops.wm.append(
                        directory=directory,
                        filename=filename,
                        clear_asset_data=True,
                        autoselect=True,
                    )
                    asset = bpy.data.objects[filename]
                    assets.append(asset)

                    bpy.ops.object.select_all(action="DESELECT")
                    context.view_layer.objects.active = asset
                    asset.select_set(True)
                    bpy.ops.object.transform_apply(
                        location=False, rotation=False, scale=True
                    )
                    for i, obj in enumerate(objs):
                        flip = False
                        if obj in self.implts:
                            tooth_number = obj.get(
                                OdentConstants.ODENT_IMPLANT_REMOVE_CODE_TAG
                            )
                            if tooth_number < 31:
                                flip = True
                        bpy.ops.object.select_all(action="DESELECT")
                        context.view_layer.objects.active = obj
                        obj.select_set(True)
                        bpy.ops.object.transform_apply(
                            location=False, rotation=False, scale=True
                        )

                        bpy.ops.object.select_all(action="DESELECT")
                        context.view_layer.objects.active = asset
                        asset.select_set(True)

                        bpy.ops.object.duplicate_move()
                        asset_dup = context.object
                        asset_dup.name = f"{i+1}_{filename}"
                        asset_dup[OdentConstants.ODENT_TYPE_TAG] = (
                            "odent_custom_sleeve cutter"
                        )
                        MoveToCollection(
                            asset_dup, OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
                        )

                        # asset_dup.matrix_world[:3] = obj.matrix_world[:3]
                        bpy.ops.object.select_all(action="DESELECT")
                        obj.select_set(True)
                        asset_dup.select_set(True)
                        context.view_layer.objects.active = obj
                        bpy.ops.wm.odent_align_to_active(invert_z=flip)
                        bpy.ops.wm.odent_parent_object(display_info=False)

                    bpy.data.objects.remove(asset)
                    bpy.ops.object.select_all(action="DESELECT")
                    ODENT_GpuDrawText()
                    close_asset_browser(context, area=self.asset_browser_area)

                    return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def defer(self):
        params = self.asset_browser_space.params
        if not params:
            return 0

        try:
            params.asset_library_reference = OdentConstants.ODENT_LIB_NAME

        except TypeError:
            # If the reference doesn't exist.
            params.asset_library_reference = "LOCAL"

        params.import_type = "APPEND"
        self.can_update = True

    def invoke(self, context, event):

        if context.space_data.type == "VIEW_3D":
            self.implts = [
                o
                for o in context.scene.objects[:]
                if o.get(OdentConstants.ODENT_TYPE_TAG)
                == OdentConstants.ODENT_IMPLANT_TYPE
            ]
            self.fixing_sleeves = [
                o
                for o in context.scene.objects[:]
                if o.get(OdentConstants.ODENT_TYPE_TAG)
                == OdentConstants.FIXING_SLEEVE_TYPE
            ]
            objs = self.implts + self.fixing_sleeves
            if not objs:
                message = [
                    "Cancelled :",
                    "Please add implants/fixing pin first and retry",
                ]
                ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.red)
                sleep(3)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

            self.asset_browser_area, self.asset_browser_space = open_asset_browser()
            self.can_update = False

            bpy.app.timers.register(self.defer)

            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}

        else:
            self.report({"WARNING"}, "Active space must be a View3d")
            return {"CANCELLED"}


class ODENT_OT_AddFixingPin(bpy.types.Operator):
    """add a guide fixing pin"""

    bl_idname = "wm.add_fixing_pin"
    bl_label = "Add Fixing Pin"
    bl_options = {"REGISTER", "UNDO"}

    sleeve_diameter: FloatProperty(
        name="Sleeve diameter",
        default=4.0,
        min=0.0,
        max=20.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Sleeve Diameter",
    )  # type: ignore

    drill_diameter: FloatProperty(
        name="Drill diameter",
        default=2.0,
        min=0.0,
        max=100.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Drill Diameter",
    )  # type: ignore
    drill_lenght: FloatProperty(
        name="Drill lenght",
        default=20.0,
        min=0.0,
        max=100.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Drill lenght",
    )  # type: ignore
    bone_depth: FloatProperty(
        name="Bone depth",
        default=6,
        min=0.0,
        max=100.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Bone Drilling depth",
    )  # type: ignore
    offset: FloatProperty(
        name="Offset",
        default=0.1,
        min=0.0,
        max=1.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Offset",
    )  # type: ignore

    def draw(self, context):
        layout = self.layout
        g = layout.grid_flow(columns=2, align=True)
        g.prop(self, "drill_diameter")
        g.prop(self, "drill_lenght")
        g = layout.grid_flow(columns=1, align=True)
        g.prop(self, "bone_depth")
        g = layout.grid_flow(columns=2, align=True)
        g.prop(self, "sleeve_diameter")
        g.prop(self, "offset")

    def add_fpin(self, context):
        n = 0
        fpins = [
            o
            for o in context.scene.objects[:]
            if o.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.FIXING_PIN_TYPE
        ]
        if fpins:
            n = len(fpins)
        coll_name = OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
        hide_collection(False, coll_name)

        # Add pin
        bpy.ops.object.select_all(action="DESELECT")
        pin = AppendObject(
            "fixing_pin", coll_name=OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
        )
        pin.name = f"Fixing_Pin({n+1})"
        pin[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.FIXING_PIN_TYPE

        pin.dimensions[:2] = [self.drill_diameter, self.drill_diameter]
        pin.location = [0, 0, -self.bone_depth]
        pin.select_set(True)
        context.view_layer.objects.active = pin
        # with context.temp_override(active_object=pin):
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

        pin.matrix_world[:3] = context.scene.cursor.matrix[:3]
        bpy.ops.wm.odent_lock_objects()

        # Add sleeve
        bpy.ops.object.select_all(action="DESELECT")
        sleeve = AppendObject(
            "sleeve", coll_name=OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
        )
        sleeve.name = f"_ADD_Fixing_Sleeve({n+1})"
        sleeve[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.FIXING_SLEEVE_TYPE
        sleeve.dimensions = [
            self.sleeve_diameter,
            self.sleeve_diameter,
            self.drill_lenght - self.bone_depth,
        ]
        sleeve.select_set(True)
        context.view_layer.objects.active = sleeve
        # with context.temp_override(active_object=sleeve):
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
        sleeve.matrix_world[:3] = context.scene.cursor.matrix[:3]

        child_of = pin.constraints.new(type="CHILD_OF")
        child_of.target = sleeve
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        return sleeve, pin

    def modal(self, context, event):
        if not event.type in {"ESC", "RET"}:
            return {"PASS_THROUGH"}

        elif event.type in {"ESC"}:
            if event.value == "PRESS":
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                # message = ["Cancelled"]
                # ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.green)
                # sleep(1)
                ODENT_GpuDrawText()
                return {"CANCELLED"}
        elif event.type in {"RET"}:
            if event.value == "RELEASE":
                fsleeve, fpin = self.add_fpin(context)

                if self.slices_pointer:
                    remove_pointer_lock()
                    with context.temp_override(
                        active_object=fsleeve, 
                        selected_objects=[fsleeve], 
                        selected_editable_objects=[fsleeve]):
                        bpy.ops.wm.odent_pointer_to_active()
                        bpy.ops.wm.odent_lock_object_to_pointer()
                        

                    # bpy.ops.object.select_all(action="DESELECT")
                    # fsleeve.select_set(True)
                    # context.view_layer.objects.active = fsleeve

                    # bpy.ops.wm.odent_pointer_to_active()

                    # bpy.ops.object.select_all(action="DESELECT")
                    # fsleeve.select_set(True)
                    # context.view_layer.objects.active = fsleeve

                    # bpy.ops.wm.odent_lock_object_to_pointer()

                message = [
                    f"{fpin.name} added.",
                    "<Left click> : Set position",
                    "<ENTER> add pin  <ESC> to cancell",
                ]
                ODENT_GpuDrawText(message)

                return {"RUNNING_MODAL"}
        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.slices_pointer = None
            slices_pointer_check_list = [
                obj
                for obj in context.scene.objects
                if obj.get(OdentConstants.ODENT_TYPE_TAG)
                == OdentConstants.SLICES_POINTER_TYPE
            ]
            if slices_pointer_check_list:
                self.slices_pointer = slices_pointer_check_list[0]
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)
        else:
            self.report({"WARNING"}, "Active space must be a View3d")
            return {"CANCELLED"}

    def execute(self, context):
        message = [
            "<Left click> : Set the pin position",
            "<ENTER> to confirm  <ESC> to cancell",
        ]
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

        ODENT_GpuDrawText(message)

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


class ODENT_OT_AddSplint(bpy.types.Operator):
    """Add Splint"""

    bl_idname = "wm.odent_add_splint"
    bl_label = "Splint"
    bl_options = {"REGISTER", "UNDO"}

    thikness: FloatProperty(
        description="SPLINT thikness", default=2, step=1, precision=2
    )  # type: ignore

    def execute(self, context):

        Splint = Metaball_Splint(self.BaseMesh, self.thikness)

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.object:
            message = ["Please select a base mesh!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        if not context.object.select_get() or context.object.type != "MESH":
            message = ["Please select a base mesh!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            self.BaseMesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_Survey(bpy.types.Operator):
    "Survey the model from view top"

    bl_idname = "wm.odent_survey"
    bl_label = "Survey Model"
    bl_options = {"REGISTER", "UNDO"}

    SurveyColor: FloatVectorProperty(
        name="Survey Color",
        description="Survey Color",
        default=[0.2, 0.12, 0.17, 1.0],
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )  # type: ignore

    def execute(self, context):
        ODENT_Props = bpy.context.scene.ODENT_Props
        bpy.ops.object.mode_set(mode="OBJECT")
        Old_Survey_mat = bpy.data.materials.get("ODENT_survey_mat")
        if Old_Survey_mat:
            OldmatSlotsIds = [
                i
                for i in range(len(self.Model.material_slots))
                if self.Model.material_slots[i].material == Old_Survey_mat
            ]
            if OldmatSlotsIds:
                for idx in OldmatSlotsIds:
                    self.Model.active_material_index = idx
                    bpy.ops.object.material_slot_remove()

        _, space3D, _ = CtxOverride(context)
        view_mtx = space3D.region_3d.view_matrix.copy()
        if not self.Model.data.materials[:]:
            ModelMat = bpy.data.materials.get(
                "ODENT_Neutral_mat"
            ) or bpy.data.materials.new("ODENT_Neutral_mat")
            ModelMat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            self.Model.active_material = ModelMat

        Survey_mat = bpy.data.materials.get(
            "ODENT_survey_mat"
        ) or bpy.data.materials.new("ODENT_survey_mat")
        Survey_mat.diffuse_color = self.SurveyColor
        self.Model.data.materials.append(Survey_mat)
        self.Model.active_material_index = len(self.Model.material_slots) - 1

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        # #############################____Surveying____###############################
        survey_faces_index_list = []

        obj = self.Model
        View_Local_Z = obj.matrix_world.inverted().to_quaternion() @ (
            space3D.region_3d.view_rotation @ Vector((0, 0, 1))
        )

        survey_faces_Idx = [
            f.index for f in obj.data.polygons if f.normal.dot(View_Local_Z) < -0.000001
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")

        for i in survey_faces_Idx:
            f = obj.data.polygons[i]
            f.select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        Survey_Vg = obj.vertex_groups.get("ODENT_survey_vg") or obj.vertex_groups.new(
            name="ODENT_survey_vg"
        )
        # obj.vertex_groups.active_index = Survey_Vg.index
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        # Store Survey direction :
        SurveyInfo_Dict = eval(ODENT_Props.SurveyInfo)
        SurveyInfo_Dict[obj.as_pointer()] = (View_Local_Z, Survey_mat)
        ODENT_Props.SurveyInfo = str(SurveyInfo_Dict)

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.active_object:
            message = ["Please select Model to survey!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        if (
            not context.active_object.select_get()
            or context.active_object.type != "MESH"
        ):
            message = ["Please select Model to survey!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            self.Model = context.active_object
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_ModelBase(bpy.types.Operator):
    """Make a model base from top user view prspective"""

    bl_idname = "wm.odent_model_base"
    bl_label = "Model Base"
    bl_options = {"REGISTER", "UNDO"}

    ModelType: EnumProperty(
        items=set_enum_items(["Upper Model", "Lower Model"]),
        description="Model Type",
        default="Upper Model",
    )  # type: ignore
    BaseHeight: FloatProperty(
        description="Base Height ", default=10, step=1, precision=2
    )  # type: ignore
    HollowModel: BoolProperty(
        name="Make Hollow Model",
        description="Add Hollow Model",
        default=False,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )

    def execute(self, context):

        # Check base boarder :
        TargetMesh = context.object
        NonManifoldVerts = count_non_manifold_verts(TargetMesh)  # mode = OBJECT

        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        # bpy.ops.mesh.select_all(action="DESELECT")
        # bpy.ops.mesh.select_non_manifold()
        # bpy.ops.object.mode_set(mode="OBJECT")
        # NonManifoldVerts = [
        #     v for v in TargetMesh.data.vertices if v.select]

        if not NonManifoldVerts:
            txt = [
                "Operation cancelled !",
                "Can't make model base from Closed mesh.",
            ]
            ODENT_GpuDrawText(message_list=txt, rect_color=OdentColors.red)
            sleep(3)
            ODENT_GpuDrawText()

            return {"CANCELLED"}

        else:
            txt = ["Processing ..."]
            ODENT_GpuDrawText(message_list=txt)

            BaseHeight = self.BaseHeight

            ####### Duplicate Target Mesh #######
            bpy.ops.object.select_all(action="DESELECT")
            TargetMesh.select_set(True)
            bpy.context.view_layer.objects.active = TargetMesh
            bpy.ops.object.duplicate_move()

            ModelBase = context.object
            ModelBase.name = f"(BASE MODEL){TargetMesh.name}"
            ModelBase.data.name = ModelBase.name
            obj = ModelBase
            # Relax border loop :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.odent.looptools_relax(
                input="selected",
                interpolation="cubic",
                iterations="3",
                regular=True,
            )
            bpy.ops.mesh.remove_doubles(threshold=0.1)

            # Make some calcul of average z_cordinate of border vertices :

            bpy.ops.object.mode_set(mode="OBJECT")

            obj_mx = obj.matrix_world.copy()
            verts = obj.data.vertices
            global_z_cords = [(obj_mx @ v.co)[2] for v in verts]

            HollowOffset = 0
            if self.ModelType == "Upper Model":
                Extrem_z = max(global_z_cords)
                Delta = BaseHeight
                if self.HollowModel:
                    HollowOffset = 4
                    BisectPlaneLoc = Vector((0, 0, Extrem_z + Delta))
                    BisectPlaneNormal = Vector((0, 0, 1))

            if self.ModelType == "Lower Model":
                Extrem_z = min(global_z_cords)
                Delta = -BaseHeight
                if self.HollowModel:
                    HollowOffset = -4
                    BisectPlaneLoc = Vector((0, 0, Extrem_z + Delta))
                    BisectPlaneNormal = Vector((0, 0, -1))

            # Border_2 = Extrude 1st border loop no translation :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.extrude_region_move()

            # change Border2 vertices zco to min_z - base_height  :

            bpy.ops.object.mode_set(mode="OBJECT")
            selected_verts = [v for v in verts if v.select]

            for v in selected_verts:
                global_v_co = obj_mx @ v.co
                v.co = obj_mx.inverted() @ Vector(
                    (
                        global_v_co[0],
                        global_v_co[1],
                        Extrem_z + Delta + HollowOffset,
                    )
                )

            # fill base :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.edge_face_add()
            bpy.ops.mesh.dissolve_limited()

            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_holes(sides=100)

            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            bpy.ops.object.material_slot_remove_all()

            mat = bpy.data.materials.get("blue stone mat") or bpy.data.materials.new(
                "blue stone mat"
            )
            mat.diffuse_color = [0.411693, 0.600872, 0.8, 1.0]
            obj.active_material = mat

            NonManifoldVerts = count_non_manifold_verts(obj)
            if NonManifoldVerts:
                txt = ["Base model have bad geometry remeshing ..."]
                ODENT_GpuDrawText(message_list=txt)
                remesh = obj.modifiers.new("Remesh", "REMESH")
                remesh.mode = "SHARP"
                remesh.octree_depth = 8
                bpy.ops.object.convert(target="MESH")
                bpy.ops.wm.odent_voxelremesh(VoxelSize=0.1)

            if self.HollowModel:
                txt = ["Processing Hollowed Model ..."]
                ODENT_GpuDrawText(message_list=txt)
                bpy.ops.wm.odent_hollow_model(thikness=2)
                HollowModel = context.object
                bpy.ops.object.material_slot_remove_all()

                mat = bpy.data.materials.get(
                    "yellow stone mat"
                ) or bpy.data.materials.new("yellow stone mat")
                mat.diffuse_color = [0.8, 0.652387, 0.435523, 1.0]
                HollowModel.active_material = mat

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")

                bpy.ops.mesh.bisect(
                    plane_co=BisectPlaneLoc,
                    plane_no=BisectPlaneNormal,
                    use_fill=True,
                    clear_inner=False,
                    clear_outer=True,
                )
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bisect(
                    plane_co=BisectPlaneLoc,
                    plane_no=BisectPlaneNormal,
                    use_fill=True,
                    clear_inner=False,
                    clear_outer=True,
                )
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

            txt = ["Base model created successfully"]
            if self.HollowModel:
                txt = ["Base and hollowed models created successfully"]

            ODENT_GpuDrawText(message_list=txt, rect_color=OdentColors.green)
            sleep(3)
            ODENT_GpuDrawText()
            return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_hollow_model(bpy.types.Operator):
    """Create a hollow Dental Model from closed Model"""

    bl_idname = "wm.odent_hollow_model"
    bl_label = "Hollow Model"
    bl_options = {"REGISTER", "UNDO"}

    thikness: FloatProperty(
        description="OFFSET", default=2, step=1, precision=2
    )  # type: ignore
    display_info: BoolProperty(
        name="display info",
        description="display info footer",
        default=False,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )

    def execute(self, context):
        Model = context.object
        # NonManifoldVerts = count_non_manifold_verts(Model) #mode = OBJECT

        # if NonManifoldVerts:
        #     txt = [
        #         "Operation cancelled !",
        #         "the mesh is not tight.",
        #     ]
        #     print(txt)
        #     if self.display_info :
        #         ODENT_GpuDrawText(message_list=txt, rect_color=OdentColors.red)
        #         sleep(3)
        #         ODENT_GpuDrawText()

        #     return {"CANCELLED"}

        # Prepare scene settings :
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Duplicate Model to Model_hollow:

        bpy.ops.object.select_all(action="DESELECT")
        Model.select_set(True)
        bpy.context.view_layer.objects.active = Model
        bpy.ops.object.duplicate_move()

        # Rename Model_hollow....

        Model_hollow = context.object
        Model_hollow.name = f"(Model hollowed){Model.name}"

        # Duplicate Model_hollow and make a low resolution duplicate :

        bpy.ops.object.duplicate_move()
        Model_lowres = context.object

        bpy.ops.wm.odent_voxelremesh(VoxelSize=0.5)

        # Add Metaballs :

        obj = Model_lowres

        loc, rot, scale = obj.matrix_world.decompose()

        verts = obj.data.vertices
        vcords = [rot @ v.co + loc for v in verts]
        mball_elements_cords = [vco - vcords[0] for vco in vcords[1:]]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        thikness = self.thikness
        radius = thikness * 5 / 8

        bpy.ops.object.metaball_add(
            type="BALL", radius=radius, enter_editmode=False, location=vcords[0]
        )

        Mball_object = context.object
        mball = Mball_object.data
        mball.resolution = 0.6
        context.object.data.update_method = "FAST"

        for i in range(len(mball_elements_cords)):
            element = mball.elements.new()
            element.co = mball_elements_cords[i]
            element.radius = radius * 2

        bpy.ops.object.convert(target="MESH")
        Mball_object = context.object

        # Make boolean intersect operation :
        bpy.ops.object.select_all(action="DESELECT")
        Model_hollow.select_set(True)
        bpy.context.view_layer.objects.active = Model_hollow
        bool_modif = Model_hollow.modifiers.new("bool", "BOOLEAN")
        bool_modif.object = Mball_object
        bool_modif.operation = "INTERSECT"
        bpy.ops.object.convert(target="MESH")

        # Delet Model_lowres and Mball_object:
        bpy.data.objects.remove(Model_lowres)
        bpy.data.objects.remove(Mball_object)

        return {"FINISHED"}

    def invoke(self, context, event):

        Active_Obj = context.active_object

        if not Active_Obj:
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        if Active_Obj.select_get() == False or Active_Obj.type != "MESH":
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_BlockModel(bpy.types.Operator):
    "Blockout Model (Remove Undercuts)"

    bl_idname = "wm.odent_block_model"
    bl_label = "BLOCK Model"
    bl_options = {"REGISTER", "UNDO"}

    printing_offset: FloatProperty(
        name="Printing Offset",
        default=0.1,
        min=0.0,
        max=1.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Implant Diameter",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "MESH"

    def invoke(self, context, event):
        surveyed_model = context.object
        Pointer = surveyed_model.as_pointer()
        ODENT_Props = bpy.context.scene.ODENT_Props
        SurveyInfo_Dict = eval(ODENT_Props.SurveyInfo)
        if not Pointer in SurveyInfo_Dict.keys():
            message = ["Please Survey Model before Blockout !"]
            ODENT_GpuDrawText(message)
            sleep(2)
            ODENT_GpuDrawText()
            return {"CANCELLED"}
        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)

    def execute(self, context):
        surveyed_model = context.object
        surveyed_model_pointer = surveyed_model.as_pointer()
        ODENT_Props = bpy.context.scene.ODENT_Props
        SurveyInfo_Dict = eval(ODENT_Props.SurveyInfo)

        View_Local_Z, Survey_mat = SurveyInfo_Dict[surveyed_model_pointer]
        ExtrudeVector = -20 * (
            surveyed_model.matrix_world.to_quaternion() @ View_Local_Z
        )

        # print(ExtrudeVector)

        # duplicate Model :
        bpy.ops.object.select_all(action="DESELECT")
        surveyed_model.select_set(True)
        bpy.context.view_layer.objects.active = surveyed_model

        bpy.ops.object.duplicate_move()
        blocked_model = bpy.context.view_layer.objects.active
        blocked_model.name = f"(BLOCKED)_{blocked_model.name}"

        bpy.ops.object.material_slot_remove_all()

        blocked_model.active_material = Survey_mat
        bpy.ops.object.mode_set(mode="EDIT")

        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.extrude_region_move()
        bpy.ops.transform.translate(value=ExtrudeVector)

        bpy.ops.object.mode_set(mode="OBJECT")
        blocked_model.data.remesh_mode = "VOXEL"
        blocked_model.data.remesh_voxel_size = 0.2
        blocked_model.data.use_remesh_fix_poles = True
        blocked_model.data.use_remesh_preserve_volume = True

        bpy.ops.object.voxel_remesh()

        return {"FINISHED"}


class ODENT_OT_UndercutsPreview(bpy.types.Operator):
    "Survey the model from view"

    bl_idname = "wm.odent_undercuts_preview"
    bl_label = "Preview Undercuts"
    bl_options = {"REGISTER", "UNDO"}

    undercuts_color = [0.54, 0.13, 0.5, 1.0]

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return (
            context.object.type == "MESH"
            and context.object.mode == "OBJECT"
            and context.object.select_get()
        )

    def execute(self, context):
        obj = context.object
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if not obj.material_slots:
            undercuts_mat_white = bpy.data.materials.get(
                "undercuts_preview_mat_white"
            ) or bpy.data.materials.new("undercuts_preview_mat_white")
            undercuts_mat_white.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            obj.active_material = undercuts_mat_white

        for i, slot in enumerate(obj.material_slots):
            if slot.material.name == "undercuts_preview_mat_color":
                obj.active_material_index = i
                bpy.ops.object.material_slot_remove()

        undercuts_mat_color = bpy.data.materials.get(
            "undercuts_preview_mat_color"
        ) or bpy.data.materials.new("undercuts_preview_mat_color")
        undercuts_mat_color.diffuse_color = self.undercuts_color
        obj.data.materials.append(undercuts_mat_color)
        obj.active_material_index = len(obj.material_slots) - 1

        # #############################____Surveying____###############################
        _, space3D, _ = CtxOverride(context)
        view_rotation_mtx = space3D.region_3d.view_rotation.to_matrix().to_4x4()

        view_z_local = obj.matrix_world.inverted().to_quaternion() @ (
            space3D.region_3d.view_rotation @ Vector((0, 0, 1))
        )

        survey_faces_index_list = [
            f.index for f in obj.data.polygons if f.normal.dot(view_z_local) < -0.000001
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")

        for i in survey_faces_index_list:
            obj.data.polygons[i].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        undercuts_vg = obj.vertex_groups.get("undercuts_vg") or obj.vertex_groups.new(
            name="undercuts_vg"
        )
        obj.vertex_groups.active_index = undercuts_vg.index
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode="OBJECT")

        obj["undercuts_vector"] = view_z_local
        obj["undercuts_view_rotation_mtx"] = list(view_rotation_mtx)

        return {"FINISHED"}


class ODENT_OT_BlockoutNew(bpy.types.Operator):
    "Create a blockout model for undercuts"

    bl_idname = "wm.odent_blockout_new"
    bl_label = "Blockout Model"
    bl_options = {"REGISTER", "UNDO"}

    undercuts_color = [0.54, 0.13, 0.5, 1.0]
    printing_offset: FloatProperty(
        name="Offset",
        default=0.1,
        min=0.0,
        max=1.0,
        step=1,
        precision=3,
        unit="LENGTH",
        description="Implant Diameter",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        return context.object.get("undercuts_vector") and context.object.get(
            "undercuts_view_rotation_mtx"
        )

    def make_blocked(self, context, obj):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        # check non manifold
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.object.mode_set(mode="OBJECT")

        non_manifold = [v for v in obj.data.vertices if v.select]

        if non_manifold:
            message = ["Adding mesh base ..."]
            ODENT_GpuDrawText(message)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.extrude_region_move()
            bpy.ops.object.mode_set(mode="OBJECT")
            extruded_verts = [v for v in obj.data.vertices if v.select]

            min_z, min_z_id = sorted(
                [(self.undercuts_vector.dot(v.co), v.index) for v in obj.data.vertices]
            )[0]
            min_co_local = obj.data.vertices[min_z_id].co

            for v in extruded_verts:
                offset_z = self.undercuts_vector.normalized() @ (min_co_local - v.co)
                v.co = v.co + ((offset_z - 5) * self.undercuts_vector.normalized())

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.remove_doubles(threshold=0.1)
            bpy.ops.odent.looptools_relax(
                input="selected",
                interpolation="cubic",
                iterations="5",
                regular=True,
            )

            bpy.ops.mesh.edge_face_add()
            bpy.ops.mesh.dissolve_limited()
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_holes(sides=100)

            # check non manifold
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.object.mode_set(mode="OBJECT")

            non_manifold = [v for v in obj.data.vertices if v.select]
            if non_manifold:
                message = ["Remeshing mesh base ..."]
                ODENT_GpuDrawText(message)
                remesh = obj.modifiers.new("Remesh", "REMESH")
                remesh.octree_depth = 8
                remesh.mode = "SHARP"
                bpy.ops.object.convert(target="MESH")

        survey_faces_index_list = [
            f.index
            for f in obj.data.polygons
            if f.normal.dot(self.undercuts_vector) < -0.000001
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode="OBJECT")

        for i in survey_faces_index_list:
            obj.data.polygons[i].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.extrude_region_move()
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="OBJECT")

        selected_verts = [v for v in obj.data.vertices if v.select]
        min_z, min_z_id = sorted(
            [(self.undercuts_vector.dot(v.co), v.index) for v in obj.data.vertices]
        )[0]
        min_co_local = obj.data.vertices[min_z_id].co
        for v in selected_verts:
            offset_z = self.undercuts_vector.normalized() @ (min_co_local - v.co)
            v.co = v.co + (offset_z * self.undercuts_vector.normalized())

        message = ["Remeshing Blocked model ..."]
        ODENT_GpuDrawText(message)

        modif_remesh = self.blocked.modifiers.new("remesh", type="REMESH")
        modif_remesh.voxel_size = 0.2
        bpy.ops.object.convert(target="MESH")

    def invoke(self, context, event):
        self.target = context.object
        self.undercuts_vector = Vector(self.target["undercuts_vector"])
        self.view_rotation_mtx = Matrix(self.target["undercuts_view_rotation_mtx"])
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=500)

    def execute(self, context):

        message = ["Processing Blocked Model..."]
        ODENT_GpuDrawText(message)

        offset = round(self.printing_offset, 2)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        bpy.context.view_layer.objects.active = self.target
        bpy.ops.object.duplicate_move()

        self.blocked = bpy.context.object
        # context.view_layer.objects.active = self.blocked
        bpy.ops.object.material_slot_remove_all()

        mat = bpy.data.materials.get(
            "undercuts_preview_mat_color"
        ) or bpy.data.materials.new("undercuts_preview_mat_color")
        mat.diffuse_color = self.undercuts_color
        self.blocked.active_material = mat

        if offset:

            displace = self.blocked.modifiers.new("displace", type="DISPLACE")
            displace.mid_level = 0
            displace.strength = offset
            message = ["displacement applied = " + str(round(offset, 2))]
            ODENT_GpuDrawText(message)

        bpy.ops.object.convert(target="MESH")

        self.make_blocked(context, self.blocked)
        context.view_layer.objects.active = self.blocked

        self.blocked.name = f"Blocked_offset({offset})_{self.target.name}"
        guide_components_coll = add_collection(
            OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
        )
        # MoveToCollection(self.blocked,guide_components_coll.name)
        for col in self.blocked.users_collection:
            col.objects.unlink(self.blocked)
        guide_components_coll.objects.link(self.blocked)

        message = ["Finished."]
        ODENT_GpuDrawText(
            message_list=message, rect_color=OdentColors.green, sleep_time=2
        )
        return {"FINISHED"}


class ODENT_OT_add_offset(bpy.types.Operator):
    """Add offset to mesh"""

    bl_idname = "wm.odent_add_offset"
    bl_label = "Add Offset"
    bl_options = {"REGISTER", "UNDO"}

    Offset: FloatProperty(
        description="OFFSET", default=0.1, step=1, precision=2
    )  # type: ignore

    def execute(self, context):

        offset = round(self.Offset, 2)

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.modifier_add(type="DISPLACE")
        bpy.context.object.modifiers["Displace"].mid_level = 0
        bpy.context.object.modifiers["Displace"].strength = offset
        bpy.ops.object.modifier_apply(modifier="Displace")

        return {"FINISHED"}

    def invoke(self, context, event):

        Active_Obj = context.active_object

        if not Active_Obj:
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        if Active_Obj.select_get() == False or Active_Obj.type != "MESH":
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_AlignPoints(bpy.types.Operator):
    """Add Align Refference points"""

    bl_idname = "wm.odent_alignpoints"
    bl_label = "ALIGN POINTS"
    bl_options = {"REGISTER", "UNDO"}

    TargetColor = (0, 1, 0, 1)  # Green
    SourceColor = (1, 0, 0, 1)  # Red
    CollName = "ALIGN POINTS"
    TargetChar = "B"
    SourceChar = "A"

    def IcpPipline(
        self,
        SourceObj,
        TargetObj,
        SourceVidList,
        TargetVidList,
        VertsLimite,
        Iterations,
        Precision,
    ):

        MaxDist = 0.0
        for i in range(Iterations):

            SourceVcoList = [
                SourceObj.matrix_world @ SourceObj.data.vertices[idx].co
                for idx in SourceVidList
            ]
            TargetVcoList = [
                TargetObj.matrix_world @ TargetObj.data.vertices[idx].co
                for idx in TargetVidList
            ]

            (
                SourceKdList,
                TargetKdList,
                DistList,
                SourceIndexList,
                TargetIndexList,
            ) = KdIcpPairs(SourceVcoList, TargetVcoList, VertsLimite=VertsLimite)

            TransformMatrix = KdIcpPairsToTransformMatrix(
                TargetKdList=TargetKdList, SourceKdList=SourceKdList
            )
            SourceObj.matrix_world = TransformMatrix @ SourceObj.matrix_world
            for RefP in self.SourceRefPoints:
                RefP.matrix_world = TransformMatrix @ RefP.matrix_world
            # Update scene :
            SourceObj.update_tag()
            bpy.context.view_layer.update()

            SourceObj = self.SourceObject

            SourceVcoList = [
                SourceObj.matrix_world @ SourceObj.data.vertices[idx].co
                for idx in SourceVidList
            ]
            _, _, DistList, _, _ = KdIcpPairs(
                SourceVcoList, TargetVcoList, VertsLimite=VertsLimite
            )
            MaxDist = max(DistList)
            a3d, s3d, r3d = CtxOverride(bpy.context)
            with bpy.context.temp_override(area=a3d, space_data=s3d, region=r3d):
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            #######################################################
            if MaxDist <= Precision:
                self.ResultMessage = [
                    "Allignement Done !",
                    f"Max Distance < or = {Precision} mm",
                ]
                print(f"Number of iterations = {i}")
                print(f"Precision of {Precision} mm reached.")
                print(f"Max Distance = {round(MaxDist, 6)} mm")
                break

        if MaxDist > Precision:
            print(f"Number of iterations = {i}")
            print(f"Max Distance = {round(MaxDist, 6)} mm")
            self.ResultMessage = [
                "Allignement Done !",
                f"Max Distance = {round(MaxDist, 6)} mm",
            ]

    def modal(self, context, event):

        ############################################
        if not event.type in {
            self.TargetChar,
            self.SourceChar,
            "DEL",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}
        #########################################
        if event.type == self.TargetChar:
            # Add Target Refference point :
            if event.value == ("PRESS"):
                if self.TargetVoxelMode:

                    CursorToVoxelPoint(Volume=self.TargetObject, CursorMove=True)

                color = self.TargetColor
                CollName = self.CollName
                self.Targetpc += 1
                name = f"B{self.Targetpc}"
                RefP = AddRefPoint(name, color, CollName)
                self.TargetRefPoints.append(RefP)
                self.TotalRefPoints.append(RefP)
                bpy.ops.object.select_all(action="DESELECT")

        #########################################
        if event.type == self.SourceChar:
            # Add Source Refference point :
            if event.value == ("PRESS"):
                if self.SourceVoxelMode:

                    CursorToVoxelPoint(Volume=self.SourceObject, CursorMove=True)

                color = self.SourceColor
                CollName = self.CollName
                self.SourceCounter += 1
                name = f"M{self.SourceCounter}"
                RefP = AddRefPoint(name, color, CollName)
                self.SourceRefPoints.append(RefP)
                self.TotalRefPoints.append(RefP)
                bpy.ops.object.select_all(action="DESELECT")

        ###########################################
        elif event.type == ("DEL"):
            if event.value == ("PRESS"):
                if self.TotalRefPoints:
                    obj = self.TotalRefPoints.pop()
                    name = obj.name
                    if name.startswith("B"):
                        self.Targetpc -= 1
                        self.TargetRefPoints.pop()
                    if name.startswith("M"):
                        self.SourceCounter -= 1
                        self.SourceRefPoints.pop()
                    bpy.data.objects.remove(obj)
                    bpy.ops.object.select_all(action="DESELECT")

        ###########################################
        elif event.type == "RET":

            if event.value == ("PRESS"):

                start = tpc()

                TargetObj = self.TargetObject
                SourceObj = self.SourceObject

                #############################################
                condition = (
                    len(self.TargetRefPoints) == len(self.SourceRefPoints)
                    and len(self.TargetRefPoints) >= 3
                )
                if not condition:
                    message = [
                        "          Please check the following :",
                        "   - The number of Base Refference points and,",
                        "       Align Refference points should match!",
                        "   - The number of Base Refference points ,",
                        "         and Align Refference points,",
                        "       should be superior or equal to 3",
                        "        <<Please check and retry !>>",
                    ]
                    icon = "COLORSET_01_VEC"
                    bpy.ops.wm.odent_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )

                else:

                    TransformMatrix = RefPointsToTransformMatrix(
                        self.TargetRefPoints, self.SourceRefPoints
                    )

                    SourceObj.matrix_world = TransformMatrix @ SourceObj.matrix_world
                    for SourceRefP in self.SourceRefPoints:
                        SourceRefP.matrix_world = (
                            TransformMatrix @ SourceRefP.matrix_world
                        )

                    for i, SP in enumerate(self.SourceRefPoints):
                        TP = self.TargetRefPoints[i]
                        MidLoc = (SP.location + TP.location) / 2
                        SP.location = TP.location = MidLoc

                    # Update scene :
                    context.view_layer.update()
                    for obj in [TargetObj, SourceObj]:
                        obj.update_tag()
                    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

                    self.ResultMessage = []
                    if not self.TargetVoxelMode and not self.SourceVoxelMode:
                        #########################################################
                        # ICP alignement :
                        print("ICP Align processing...")
                        IcpVidDict = VidDictFromPoints(
                            TargetRefPoints=self.TargetRefPoints,
                            SourceRefPoints=self.SourceRefPoints,
                            TargetObj=TargetObj,
                            SourceObj=SourceObj,
                            radius=3,
                        )
                        ODENT_Props = bpy.context.scene.ODENT_Props
                        ODENT_Props.IcpVidDict = str(IcpVidDict)

                        SourceVidList, TargetVidList = (
                            IcpVidDict[SourceObj],
                            IcpVidDict[TargetObj],
                        )

                        self.IcpPipline(
                            SourceObj=SourceObj,
                            TargetObj=TargetObj,
                            SourceVidList=SourceVidList,
                            TargetVidList=TargetVidList,
                            VertsLimite=10000,
                            Iterations=30,
                            Precision=0.0001,
                        )

                    ##########################################################
                    self.FullSpace3D.overlay.show_outline_selected = True
                    self.FullSpace3D.overlay.show_object_origins = True
                    self.FullSpace3D.overlay.show_annotation = True
                    self.FullSpace3D.overlay.show_text = True
                    self.FullSpace3D.overlay.show_extras = True
                    self.FullSpace3D.overlay.show_floor = True
                    self.FullSpace3D.overlay.show_axis_x = True
                    self.FullSpace3D.overlay.show_axis_y = True
                    ###########################################################
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)
                    with bpy.context.temp_override(
                        area=self.FullArea3D,
                        space_data=self.FullSpace3D,
                        region=self.FullRegion3D,
                    ):

                        bpy.ops.object.select_all(action="DESELECT")
                        bpy.ops.wm.tool_set_by_id(name="builtin.select")
                        bpy.ops.screen.region_toggle(region_type="UI")
                        bpy.ops.screen.screen_full_area()

                    bpy.context.scene.tool_settings.use_snap = False
                    bpy.context.scene.cursor.location = (0, 0, 0)

                    if self.Solid:
                        self.FullSpace3D.shading.background_color = (
                            self.background_color
                        )
                        self.FullSpace3D.shading.background_type = self.background_type

                    TargetObj = self.TargetObject
                    SourceObj = self.SourceObject

                    if self.TotalRefPoints:
                        for RefP in self.TotalRefPoints:
                            bpy.data.objects.remove(RefP)

                    AlignColl = bpy.data.collections.get("ALIGN POINTS")
                    if AlignColl:
                        bpy.data.collections.remove(AlignColl)

                    ODENT_Props = context.scene.ODENT_Props
                    ODENT_Props.AlignModalState = False

                    if self.ResultMessage:
                        print(self.ResultMessage)

                    ##########################################################

                    finish = tpc()
                    print(f"Alignement finshed in {finish-start} secondes")

                    return {"FINISHED"}

        ###########################################
        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                ##########################################################
                self.FullSpace3D.overlay.show_outline_selected = True
                self.FullSpace3D.overlay.show_object_origins = True
                self.FullSpace3D.overlay.show_annotation = True
                self.FullSpace3D.overlay.show_text = True
                self.FullSpace3D.overlay.show_extras = True
                self.FullSpace3D.overlay.show_floor = True
                self.FullSpace3D.overlay.show_axis_x = True
                self.FullSpace3D.overlay.show_axis_y = True
                ###########################################################
                for Name in self.visibleObjects:
                    obj = bpy.data.objects.get(Name)
                    if obj:
                        obj.hide_set(False)
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.scene.cursor.location = (0, 0, 0)
                with bpy.context.temp_override(
                    area=self.FullArea3D,
                    space_data=self.FullSpace3D,
                    region=self.FullRegion3D,
                ):

                    bpy.ops.object.select_all(action="DESELECT")
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")

                    bpy.ops.screen.region_toggle(region_type="UI")
                    bpy.ops.screen.screen_full_area()

                if self.Solid:
                    self.FullSpace3D.shading.background_color = self.background_color
                    self.FullSpace3D.shading.background_type = self.background_type

                TargetObj = self.TargetObject
                SourceObj = self.SourceObject

                if self.TotalRefPoints:
                    for RefP in self.TotalRefPoints:
                        bpy.data.objects.remove(RefP)

                AlignColl = bpy.data.collections.get("ALIGN POINTS")
                if AlignColl:
                    bpy.data.collections.remove(AlignColl)

                ODENT_Props = context.scene.ODENT_Props
                ODENT_Props.AlignModalState = False

                # message = [
                #     " The Align Operation was Cancelled!",
                # ]

                # icon = "COLORSET_02_VEC"
                # bpy.ops.wm.odent_message_box(
                #     "INVOKE_DEFAULT", message=str(message), icon=icon
                # )
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        Condition_1 = len(bpy.context.selected_objects) != 2
        Condition_2 = bpy.context.selected_objects and not bpy.context.active_object
        Condition_3 = bpy.context.selected_objects and not (
            bpy.context.active_object in bpy.context.selected_objects
        )

        if Condition_1 or Condition_2 or Condition_3:

            message = [
                "Selection is invalid !",
                "Please Deselect all objects,",
                "Select the Object to Align and ,",
                "<SHIFT + Select> the Base Object.",
                "Click info button for more info.",
            ]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":
                ODENT_Props = context.scene.ODENT_Props
                ODENT_Props.AlignModalState = True
                # Prepare scene  :
                ##########################################################

                bpy.context.space_data.overlay.show_outline_selected = False
                bpy.context.space_data.overlay.show_object_origins = False
                bpy.context.space_data.overlay.show_annotation = False
                bpy.context.space_data.overlay.show_text = False
                bpy.context.space_data.overlay.show_extras = False
                bpy.context.space_data.overlay.show_floor = False
                bpy.context.space_data.overlay.show_axis_x = False
                bpy.context.space_data.overlay.show_axis_y = False
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.scene.tool_settings.transform_pivot_point = (
                    "INDIVIDUAL_ORIGINS"
                )
                bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

                ###########################################################
                self.TargetObject = bpy.context.active_object
                self.SourceObject = [
                    obj
                    for obj in bpy.context.selected_objects
                    if not obj is self.TargetObject
                ][0]

                VisObj = bpy.context.visible_objects
                self.visibleObjects = [obj.name for obj in VisObj]
                for obj in VisObj:
                    if not obj in [self.TargetObject, self.SourceObject]:
                        obj.hide_set(True)

                self.Solid = False
                if bpy.context.space_data.shading.type == "SOLID":
                    self.Solid = True
                    self.background_type = (
                        bpy.context.space_data.shading.background_type
                    )
                    bpy.context.space_data.shading.background_type = "VIEWPORT"
                    self.background_color = tuple(
                        bpy.context.space_data.shading.background_color
                    )
                    bpy.context.space_data.shading.background_color = (0.0, 0.0, 0.0)

                self.TargetVoxelMode = self.TargetObject.name.startswith(
                    "BD"
                ) and self.TargetObject.name.endswith("_CTVolume")
                self.SourceVoxelMode = self.SourceObject.name.startswith(
                    "BD"
                ) and self.SourceObject.name.endswith("_CTVolume")
                self.TargetRefPoints = []
                self.SourceRefPoints = []
                self.TotalRefPoints = []

                self.Targetpc = 0
                self.SourceCounter = 0

                bpy.ops.screen.screen_full_area()
                self.FullArea3D, self.FullSpace3D, self.FullRegion3D = CtxOverride(
                    context
                )

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class ODENT_OT_AutoAlign(bpy.types.Operator):
    """auto mesh registration"""

    bl_idname = "wm.odent_auto_align"
    bl_label = "ALIGN AUTO"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        items=set_enum_items(["global", "icp", "global+icp"]),
        description="auto align mode",
        default="global+icp",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if (
            context.object
            and context.object.select_get()
            and len(context.selected_objects) == 2
        ):
            if all([obj.type == "MESH" for obj in context.selected_objects]):
                return True
        return False

    def execute(self, context):
        # binary = shlex.quote(OdentConstants.MESH_REG_AUTO_PATH)
        # print("binary : ", binary)
        # print("source : ", self.src_path)
        # print("target : ", self.tgt_path)
        # print("transform path : ", self.transform_path)
        bpy.ops.object.select_all(action="DESELECT")
        context.view_layer.objects.active = self.source
        self.source.select_set(True)
        bpy.ops.wm.stl_export(
            "EXEC_DEFAULT", filepath=self.src_path, export_selected_objects=True
        )

        bpy.ops.object.select_all(action="DESELECT")
        context.view_layer.objects.active = self.target
        self.target.select_set(True)
        bpy.ops.wm.stl_export(filepath=self.tgt_path, export_selected_objects=True)

        # while True :
        #     if exists(self.src_path) and exists(self.tgt_path) :
        #         break

        # cmd = f'"{OdentConstants.MESH_REG_AUTO_PATH}" "{self.src_path}" "{self.tgt_path}" --transform_output "{self.transform_path}"'

        # if self.mode == "global" :
        #     cmd +=  " --registration_mode global"
        # elif self.mode == "icp" :
        #     cmd +=  " --registration_mode icp"

        cmd = [
            OdentConstants.MESH_REG_AUTO_PATH,
            self.src_path,
            self.tgt_path,
            "--transform_output",
            self.transform_path,
        ]
        if self.mode == "global":
            cmd.extend(["--registration_mode", "global"])
        elif self.mode == "icp":
            cmd.extend(["--registration_mode", "icp"])

        if is_linux():
            if not is_wine_installed():
                message = install_wine()
                if message:
                    odent_log(message)
                    ODENT_GpuDrawText(
                        message_list=message, rect_color=OdentColors.red, sleep_time=2
                    )
                    return {"CANCELLED"}
            cmd.insert(0, "wine")
        # Start the process
        odent_log(cmd)
        process = subprocess.Popen(cmd)

        # Wait for the process to finish
        process.wait()
        if process.returncode != 0:
            message = ["cancelled registration problem occured!"]
            odent_log(message)
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        # subprocess.call(cmd, shell=True)
        counter = 1
        while counter < 20:
            if exists(self.transform_path):
                break
            counter += 1
            sleep(1)

        transform = load_matrix_from_file(self.transform_path)
        if not transform:
            message = ["cancelled registration problem occured!"]
            odent_log(message)
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.red, sleep_time=2
            )
            return {"CANCELLED"}
        self.source.matrix_world = transform @ self.source.matrix_world
        bpy.ops.object.select_all(action="DESELECT")
        os.remove(self.src_path)
        os.remove(self.tgt_path)
        os.remove(self.transform_path)
        message = ["Finished!"]
        ODENT_GpuDrawText(
            message_list=message, rect_color=OdentColors.green, sleep_time=2
        )

        return {"FINISHED"}

    def invoke(self, context, event):
        self.target = context.object
        self.source = [obj for obj in context.selected_objects if obj != self.target][0]
        temp_path = tempfile.gettempdir()
        self.src_path = join(temp_path, "source.stl")
        self.src_path_safe = shlex.quote(self.src_path)
        self.tgt_path = join(temp_path, "target.stl")
        self.tgt_path_safe = shlex.quote(self.tgt_path)
        self.transform_path = join(temp_path, "transform.txt")
        self.transform_path_safe = shlex.quote(self.transform_path)
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_AlignPointsInfo(bpy.types.Operator):
    """Add Align Refference points"""

    bl_idname = "wm.odent_alignpointsinfo"
    bl_label = "INFO"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        message = [
            "\u2588 Deselect all objects,",
            "\u2588 Select the Object to Align,",
            "\u2588 Press <SHIFT + Click> to select the Base Object,",
            "\u2588 Click <ALIGN> button,",
            f"      Press <Left Click> to Place Cursor,",
            f"      Press <'B'> to Add Green Point (Base),",
            f"      Press <'A'> to Add Red Point (Align),",
            f"      Press <'DEL'> to delete Point,",
            f"      Press <'ESC'> to Cancel Operation,",
            f"      Press <'ENTER'> to execute Alignement.",
            "\u2588 NOTE :",
            "3 Green Points and 3 Red Points,",
            "are the minimum required for Alignement!",
        ]

        icon = "COLORSET_02_VEC"
        bpy.ops.wm.odent_message_box("INVOKE_DEFAULT", message=str(message), icon=icon)

        return {"FINISHED"}


########################################################################
# Mesh Tools Operators
########################################################################
class ODENT_OT_AddColor(bpy.types.Operator):
    """Add color material"""

    bl_idname = "wm.odent_add_color"
    bl_label = "Add Color"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.select_get()
            and context.object.type in ["MESH", "CURVE"]
        )

    def execute(self, context):
        obj = context.object
        matName = f"{obj.name}_Mat"
        mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
        mat.use_nodes = False
        mat.diffuse_color = [0.8, 0.8, 0.8, 1.0]

        obj.active_material = mat

        return {"FINISHED"}


class ODENT_OT_RemoveColor(bpy.types.Operator):
    """Remove color material"""

    bl_idname = "wm.odent_remove_color"
    bl_label = "Remove Color"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.select_get()
            and context.object.type in ["MESH", "CURVE"]
            and context.object.material_slots
        )

    def execute(self, context):
        bpy.ops.object.material_slot_remove_all()

        return {"FINISHED"}


class ODENT_OT_JoinObjects(bpy.types.Operator):
    "Join Objects"

    bl_idname = "wm.odent_join_objects"
    bl_label = "JOIN :"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object and len(context.selected_objects) >= 2

    def execute(self, context):

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj in context.selected_objects
            and len(context.selected_objects) >= 2
        )

        if not condition:

            message = [" Please select objects to join !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            bpy.ops.object.join()

            return {"FINISHED"}


class ODENT_OT_SeparateObjects(bpy.types.Operator):
    "Separate Objects"

    bl_idname = "wm.odent_separate_objects"
    bl_label = "SEPARATE :"
    bl_options = {"REGISTER", "UNDO"}

    items = ["Selection", "Loose Parts", ""]
    SeparateMode: EnumProperty(
        items=set_enum_items(items), description="SeparateMode", default="Loose Parts"
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.type == "MESH"
            and len(context.selected_objects) == 1
        )

    def execute(self, context):

        if self.SeparateMode == "Loose Parts":
            bpy.ops.mesh.separate(type="LOOSE")

        if self.SeparateMode == "Selection":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.separate(type="SELECTED")

        bpy.ops.object.mode_set(mode="OBJECT")

        Parts = list(context.selected_objects)

        if Parts and len(Parts) > 1:
            for obj in Parts:
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

            bpy.ops.object.select_all(action="DESELECT")
            Parts[-1].select_set(True)
            bpy.context.view_layer.objects.active = Parts[-1]

        return {"FINISHED"}

    def invoke(self, context, event):

        self.ActiveObj = context.active_object
        condition = (
            self.ActiveObj
            and self.ActiveObj.type == "MESH"
            and self.ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = ["Please select the target object !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_Parent(bpy.types.Operator):
    "Parent Object"

    bl_idname = "wm.odent_parent_object"
    bl_label = "PARENT"
    bl_options = {"REGISTER", "UNDO"}

    display_info: BoolProperty(default=True)  # type: ignore

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.select_get()
            and len(context.selected_objects) >= 2
        )

    def execute(self, context):
        active_cobject = context.object
        other_selected_objects = [
            obj for obj in context.selected_objects if obj is not context.object
        ]
        bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

        # for obj in other_selected_objects:
        #     if obj.constraints:
        #         for c in obj.constraints:
        #             if c.type == "CHILD_OF":
        #                 context.view_layer.objects.active = obj
        #                 bpy.ops.constraint.apply(constraint=c.name)

        #     child_constraint = obj.constraints.new("CHILD_OF")
        #     child_constraint.target = active_cobject

        message = [f"selected object(s) parented to {active_cobject.name}"]
        if self.display_info:
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.green, sleep_time=2
            )
        return {"FINISHED"}


class ODENT_OT_Unparent(bpy.types.Operator):
    "Un-Parent objects"

    bl_idname = "wm.odent_unparent_objects"
    bl_label = "UnParent"
    bl_options = {"REGISTER", "UNDO"}
    
    
    display_info: BoolProperty(default=True)  # type: ignore

    @classmethod
    def poll(cls, context):
        return context.object and context.object.select_get()

    def execute(self, context):
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

        # for obj in context.selected_objects:
        #     if obj.constraints:
        #         for c in obj.constraints:
        #             if c.type == "CHILD_OF":
        #                 context.view_layer.objects.active = obj
        #                 bpy.ops.constraint.apply(constraint=c.name)

        message = ["Selected object(s) unparented "]
        if self.display_info:
            ODENT_GpuDrawText(
                message_list=message, rect_color=OdentColors.green, sleep_time=2
            )
        return {"FINISHED"}


class ODENT_OT_align_to_cursor(bpy.types.Operator):
    """Align Model To Front view"""

    bl_idname = "wm.odent_align_to_cursor"
    bl_label = "Align to Cursor"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object

    def execute(self, context):
        context.object.matrix_world[:3] = context.scene.cursor.matrix[:3]

        return {"FINISHED"}


class ODENT_OT_align_to_front(bpy.types.Operator):
    """Align Model To Front view"""

    bl_idname = "wm.odent_align_to_front"
    bl_label = "Align to Front"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        if not context.object.type in ["MESH", "CURVE"]:
            return False
        return True

    def execute(self, context):
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        _, space_data, _ = CtxOverride(context)
        obj = context.object
        view3d_rot_matrix = space_data.region_3d.view_rotation.to_matrix().to_4x4()
        Eul_90x_matrix = Euler((radians(90), 0, 0), "XYZ").to_matrix().to_4x4()

        # Rotate Model :
        obj.matrix_world = (
            Eul_90x_matrix @ view3d_rot_matrix.inverted() @ obj.matrix_world
        )

        # view3d_rot_matrix = context.space_data.region_3d.view_rotation.to_matrix().to_4x4()
        # obj.matrix_world = view3d_rot_matrix.inverted() @ obj.matrix_world
        # obj.rotation_euler.rotate_axis("X", math.pi)
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):

            bpy.ops.view3d.view_all(center=True)
            bpy.ops.view3d.view_axis(type="FRONT")
            bpy.ops.wm.tool_set_by_id(name="builtin.select")

        return {"FINISHED"}


class ODENT_OT_to_center(bpy.types.Operator):
    "Center Model to world origin"

    bl_idname = "wm.odent_to_center"
    bl_label = "TO CENTER"
    bl_options = {"REGISTER", "UNDO"}

    yellow_stone = [1.0, 0.36, 0.06, 1.0]

    @classmethod
    def poll(cls, context):
        return context.object and context.object.select_get()

    def modal(self, context, event):

        if not event.type in {"RET", "ESC"}:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == "RET":

            if event.value == ("PRESS"):
                self.target.location -= context.scene.cursor.location
                context.scene.cursor.location = (0, 0, 0)
                context.view_layer.objects.active = self.target
                a3d, s3d, r3d = CtxOverride(bpy.context)
                with bpy.context.temp_override(area=a3d, space_data=s3d, region=r3d):
                    # bpy.ops.view3d.snap_cursor_to_center()
                    bpy.ops.view3d.view_all(center=True)
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                    # bpy.ops.object.transform_apply()
                ODENT_GpuDrawText()

            return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                ODENT_GpuDrawText()

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if context.space_data.type == "VIEW_3D":

            bpy.ops.object.mode_set(mode="OBJECT")
            self.target = context.view_layer.objects.active
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

            message = [
                "Left-Click : to place cursor",
                "Enter : to center model",
            ]

            ODENT_GpuDrawText(message)
            wm = context.window_manager
            wm.modal_handler_add(self)
            return {"RUNNING_MODAL"}

        else:

            self.report({"WARNING"}, "Active space must be a View3d")

            return {"CANCELLED"}


class ODENT_OT_center_cursor(bpy.types.Operator):
    """Cursor to World Origin"""

    bl_idname = "wm.odent_center_cursor"
    bl_label = "Center Cursor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        bpy.ops.view3d.snap_cursor_to_center()

        return {"FINISHED"}


class ODENT_OT_decimate(bpy.types.Operator):
    """Decimate to ratio"""

    bl_idname = "wm.odent_decimate"
    bl_label = "Decimate Model"
    bl_options = {"REGISTER", "UNDO"}
    
    decimate_ratio: FloatProperty(default=0)  # type: ignore

    @classmethod
    def poll(cls, context):

        return (
            context.object
            and context.object.select_get()
            and context.object.type in ["MESH", "CURVE"]
        )

    def execute(self, context):
        ODENT_Props = context.scene.ODENT_Props
        if not self.decimate_ratio:
            self.decimate_ratio = round(ODENT_Props.decimate_ratio, 2)

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.modifier_add(type="DECIMATE")
        bpy.context.object.modifiers["Decimate"].ratio = self.decimate_ratio
        bpy.ops.object.modifier_apply(modifier="Decimate")

        return {"FINISHED"}


class ODENT_OT_fill(bpy.types.Operator):
    """fill edge or face"""

    bl_idname = "wm.odent_fill"
    bl_label = "FILL"
    bl_options = {"REGISTER", "UNDO"}

    Fill_treshold: IntProperty(
        name="Hole Fill Treshold",
        description="Hole Fill Treshold",
        default=400,
    )  # type: ignore

    @classmethod
    def poll(cls, context):

        return (
            context.object
            and context.object.select_get()
            and context.object.type in ["MESH"]
        )

    def execute(self, context):

        Mode = context.object.mode
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)

        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()

        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode=Mode)

        return {"FINISHED"}

    def invoke(self, context, event):

        if context.object.mode == "EDIT":
            bpy.ops.mesh.edge_face_add()
            return {"FINISHED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_retopo_smooth(bpy.types.Operator):
    """Retopo sculpt for filled holes"""

    bl_idname = "wm.odent_retopo_smooth"
    bl_label = "Retopo Smooth"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):

        return (
            context.object
            and context.object.select_get()
            and context.object.type in ["MESH"]
        )

    def execute(self, context):
        obj = context.object
        if obj.mode == "SCULPT":
            bpy.ops.object.mode_set(mode="OBJECT")
            return {"FINISHED"}

        # Prepare scene settings :
        # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="SCULPT")
        sculpt = bpy.context.scene.tool_settings.sculpt
        bpy.ops.brush.asset_activate(
            asset_library_type="ESSENTIALS",
            asset_library_identifier="",
            relative_asset_identifier="brushes/essentials_brushes-mesh_sculpt.blend/Brush/Density",
        )
        if not obj.use_dynamic_topology_sculpting:
            bpy.ops.sculpt.dynamic_topology_toggle()

        brush = bpy.data.brushes["Density"]
        # set brush radius
        # context.scene.tool_settings.unified_paint_settings.size = 50

        brush.strength = 0.5
        brush.auto_smooth_factor = 0.5
        brush.use_automasking_topology = True
        brush.use_frontface = True

        sculpt.detail_type_method = "CONSTANT"
        sculpt.constant_detail_resolution = 1.7
        # bpy.ops.sculpt.sample_detail_size(mode="DYNTOPO")

        return {"FINISHED"}


class ODENT_OT_clean_mesh2(bpy.types.Operator):
    """Fill small and medium holes and remove small parts"""

    bl_idname = "wm.odent_clean_mesh2"
    bl_label = "CLEAN MESH"
    bl_options = {"REGISTER", "UNDO"}

    Fill_treshold: IntProperty(
        name="Holes Fill Treshold",
        description="Hole Fill Treshold",
        default=100,
    )  # type: ignore

    @classmethod
    def poll(cls, context):

        return (
            context.object
            and context.object.select_get()
            and context.object.type in ["MESH"]
        )

    def execute(self, context):

        obj = context.object

        ####### Get model to clean #######
        bpy.ops.object.mode_set(mode="OBJECT")
        # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete_loose(use_faces=True)

        ############ clean non_manifold borders ##############
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete_loose(use_faces=True)

        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.select_less()
        bpy.ops.mesh.delete(type="VERT")

        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()

        bpy.ops.odent.looptools_relax(
            input="selected",
            interpolation="cubic",
            iterations="3",
            regular=True,
        )

        bpy.ops.object.mode_set(mode="OBJECT")

        print("Clean Mesh finished.")

        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_clean_mesh(bpy.types.Operator):
    """Fill small and medium holes and remove small parts"""

    bl_idname = "wm.odent_clean_mesh"
    bl_label = "CLEAN MESH"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ActiveObj = context.active_object

        if not ActiveObj:
            message = [" Invalid Selection ", "Please select Target mesh ! "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}
        else:
            Conditions = [
                not ActiveObj.select_set,
                not ActiveObj.type == "MESH",
            ]

            if Conditions[0] or Conditions[1]:
                message = [" Invalid Selection ", "Please select Target mesh ! "]
                icon = "COLORSET_01_VEC"
                bpy.ops.wm.odent_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:

                ####### Get model to clean #######
                bpy.ops.object.mode_set(mode="OBJECT")
                # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
                Obj = ActiveObj
                bpy.ops.object.select_all(action="DESELECT")
                Obj.select_set(True)
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.context.tool_settings.mesh_select_mode = (True, False, False)

                ####### Remove doubles, Make mesh consistent (face normals) #######
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles(threshold=0.1)
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)

                ############ clean non_manifold borders ##############
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.delete(type="VERT")

                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.select_more()
                bpy.ops.mesh.select_less()
                bpy.ops.mesh.delete(type="VERT")

                ####### Fill Holes #######

                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.fill_holes(sides=100)
                bpy.ops.mesh.quads_convert_to_tris(
                    quad_method="BEAUTY", ngon_method="BEAUTY"
                )

                ####### Relax borders #######
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.remove_doubles(threshold=0.1)

                bpy.ops.odent.looptools_relax(
                    input="selected",
                    interpolation="cubic",
                    iterations="1",
                    regular=True,
                )

                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")
                Obj.select_set(True)
                bpy.context.view_layer.objects.active = Obj

                print("Clean Mesh finished.")

                return {"FINISHED"}


class ODENT_OT_VoxelRemesh(bpy.types.Operator):
    """Voxel Remesh Operator"""

    bl_idname = "wm.odent_voxelremesh"
    bl_label = "REMESH"
    bl_options = {"REGISTER", "UNDO"}

    VoxelSize: FloatProperty(
        name="Voxel Size",
        description="Remesh Voxel Size",
        default=0.1,  # changed  0.1 to 0.5
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=10,
        precision=1,
    )  # type: ignore

    @classmethod
    def poll(cls, context):

        return (
            context.object
            and context.object.select_get()
            and context.object.type in ["MESH"]
        )

    def execute(self, context):
        obj = context.object
        remesh = obj.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = self.VoxelSize
        bpy.ops.object.convert(target="MESH", keep_original=False)

        return {"FINISHED"}

    def invoke(self, context, event):

        self.VoxelSize = 0.5  # changed  0.1 to 0.5
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_RibbonCutterAdd(bpy.types.Operator):
    """mesh curve cutter tool"""

    bl_idname = "wm.odent_ribboncutteradd"
    bl_label = "DRAW CURVE"
    bl_options = {"REGISTER", "UNDO"}

    closeCurve: BoolProperty(name="close cutting curve", default=True)  # type: ignore

    @classmethod
    def poll(cls, context):

        base_mesh = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        return base_mesh

    def add_curve_cutter(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(
            radius=1, enter_editmode=False, align="CURSOR"
        )
        # Set cutting_tool name :
        self.cutter = context.object
        self.cutter.name = "ODENT_Ribbon_Cutter"
        self.cutter[OdentConstants.ODENT_TYPE_TAG] = "curvecutter3"
        self.cutter["odent_target"] = self.base_mesh.name

        # CurveCutter settings :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.curve.dissolve_verts()
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        self.cutter.data.dimensions = "3D"
        self.cutter.data.twist_smooth = 3
        self.cutter.data.bevel_resolution = 6
        self.cutter.data.use_fill_caps = True
        self.cutter.data.bevel_depth = 0.3
        self.cutter.data.extrude = 4
        self.cutter.data.offset = -0.3
        context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat = bpy.data.materials.get(
            "Odent_curve_cutter_mat"
        ) or bpy.data.materials.new("Odent_curve_cutter_mat")
        mat.diffuse_color = [1, 0, 0, 1]
        mat.roughness = 0.3
        bpy.ops.object.mode_set(mode="OBJECT")
        self.cutter.active_material = mat

        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Odent Cutters")

    def add_cutter_point(self):
        bpy.context.view_layer.objects.active = self.cutter
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.extrude(mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.curve.select_all(action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(mode="OBJECT")

    def del_cutter_point(self):
        try:
            bpy.context.view_layer.objects.active = self.cutter
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(action="SELECT")
                bpy.ops.curve.handle_type_set(type="AUTOMATIC")
                bpy.ops.curve.select_all(action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass

    def modal(self, context, event):

        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}
        elif event.type == "RET" and self.counter == 0:
            return {"PASS_THROUGH"}
        elif event.type == "DEL" and self.counter == 0:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid:
                    self.add_cutter_point()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:

            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid:
                    self.add_curve_cutter(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:

            if event.value == ("PRESS"):
                area, space_data, region_3d = CtxOverride(context)
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.context.view_layer.objects.active = self.cutter
                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    self.cutter.select_set(True)

                    if self.closeCurve:
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.curve.cyclic_toggle()
                        bpy.ops.object.mode_set(mode="OBJECT")

                    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    space_data.overlay.show_outline_selected = True
                    ODENT_GpuDrawText()
                    return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.odent.message_box("INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            self.scn = context.scene
            self.counter = 0

            self.start_objects = bpy.data.objects[:]
            self.start_collections = bpy.data.collections[:]
            self.start_visible_objects = bpy.context.visible_objects[:]

            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            self.base_mesh.select_set(True)
            context.view_layer.objects.active = self.base_mesh

            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
            context.window_manager.modal_handler_add(self)
            txt = [
                "Left click : draw curve | DEL : roll back | ESC : to cancell operation",
                "ENTER : to finalise",
            ]
            ODENT_GpuDrawText(txt)
            return {"RUNNING_MODAL"}


class ODENT_OT_RibbonCutter_Perform_Cut(bpy.types.Operator):
    "Performe CurveCutter1 Operation"

    bl_idname = "wm.odent_ribboncutter_perform_cut"
    bl_label = "CUT"
    bl_options = {"REGISTER", "UNDO"}
    
    items = ["Remove Small Part", "Remove Big Part", "Keep All"]

    CutMode: EnumProperty(
        name="Splint Cut Mode",
        items=set_enum_items(items),
        description="Splint Cut Mode",
        default="Keep All",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        cutters = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == "curvecutter3"
        ]
        return cutters

    def execute(self, context):

        start_unvis_objects = [
            obj for obj in context.scene.objects if not obj in context.visible_objects
        ]

        txt = ["Processing ..."]
        ODENT_GpuDrawText(message_list=txt)
        area3D, space3D, region_3d = CtxOverride(context)
        bpy.context.scene.tool_settings.use_snap = False
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.snap_cursor_to_center()

        datadict = {}

        allcutters = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == "curvecutter3"
        ]
        for c in allcutters:
            target_name = c["odent_target"]
            if not datadict.get(target_name):
                datadict.update({target_name: [c.name]})
            else:
                datadict[target_name].append(c.name)

        for target_name, cutter_names in datadict.items():
            if not bpy.data.objects.get(target_name):
                continue
            target = bpy.data.objects.get(target_name)
            target.hide_set(False)
            target.hide_viewport = False
            target.hide_select = False
            cutters = [
                bpy.data.objects.get(n) for n in cutter_names if bpy.data.objects.get(n)
            ]
            cutters_list = []
            for cc in cutters:
                bpy.context.view_layer.objects.active = cc
                cc.hide_set(False)
                cc.hide_viewport = False
                cc.hide_select = False
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                cc.select_set(True)
                bpy.ops.object.duplicate_move()
                cc = context.object

                # remove material :
                bpy.ops.object.material_slot_remove_all()

                # convert CurveCutter to mesh :
                bpy.ops.object.convert(target="MESH")
                cutter = context.object
                cutters_list.append(cutter)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in cutters_list:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            if len(cutters_list) > 1:
                bpy.ops.object.join()

            cutter = context.object
            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            # boolean modifier difference :
            bool = target.modifiers.new(name="Bool", type="BOOLEAN")
            bool.operation = "DIFFERENCE"
            bool.object = cutter
            bpy.ops.object.convert(target="MESH", keep_original=False)

            bpy.data.objects.remove(cutter)

            VisObj = [obj.name for obj in context.visible_objects]
            for obj in context.visible_objects:
                if not obj is target:
                    obj.hide_set(True)

            bpy.ops.wm.odent_separate_objects(SeparateMode="Loose Parts")

            if not self.CutMode == "Keep All":

                target_Max = (
                    max(
                        [
                            [len(obj.data.polygons), obj.name]
                            for obj in context.visible_objects
                        ]
                    )
                )[1]
                target_min = (
                    min(
                        [
                            [len(obj.data.polygons), obj.name]
                            for obj in context.visible_objects
                        ]
                    )
                )[1]

                if self.CutMode == "Remove Small Part":
                    result = bpy.data.objects.get(target_Max)
                    for obj in context.visible_objects:
                        if not obj is result:
                            bpy.data.objects.remove(obj)

                if self.CutMode == "Remove Big Part":
                    result = bpy.data.objects.get(target_min)
                    for obj in context.visible_objects:
                        if not obj is result:
                            bpy.data.objects.remove(obj)

                result.select_set(True)
                bpy.context.view_layer.objects.active = result
                # bpy.ops.object.shade_flat()

            if self.CutMode == "Keep All":
                bpy.ops.object.select_all(action="DESELECT")

            for objname in VisObj:
                obj = bpy.data.objects.get(objname)
                if obj:
                    obj.hide_set(False)
        area3D, space3D, region_3d = CtxOverride(context)
        bpy.context.scene.tool_settings.use_snap = False
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.snap_cursor_to_center()

        col = bpy.data.collections.get("Odent Cutters")
        if col:
            for obj in col.objects:
                bpy.data.objects.remove(obj)

            bpy.data.collections.remove(col)

        for obj in context.view_layer.objects:
            if obj in start_unvis_objects:
                try:
                    obj.hide_set(True)
                except:
                    pass
            else:
                try:
                    obj.hide_set(False)
                    obj.hide_viewport = False
                    obj.hide_select = False
                except:
                    pass

        txt = ["Done."]
        ODENT_GpuDrawText(message_list=txt, rect_color=OdentColors.green)
        sleep(1)
        ODENT_GpuDrawText()

        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_CurveCutterAdd(bpy.types.Operator):
    """description of this Operator"""

    bl_idname = "wm.odent_curvecutteradd"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        ODENT_Props = context.scene.ODENT_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    CurveCutterName = ODENT_Props.CurveCutterNameProp
                    CurveCutter = bpy.data.objects[CurveCutterName]
                    CurveCutter.select_set(True)
                    bpy.context.view_layer.objects.active = CurveCutter
                    bpy.ops.object.mode_set(mode="OBJECT")

                    if ODENT_Props.CurveCutCloseMode == "Close Curve":
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.curve.cyclic_toggle()
                        bpy.ops.object.mode_set(mode="OBJECT")

                    # bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Shrinkwrap")

                    bpy.context.object.data.bevel_depth = 0
                    bpy.context.object.data.extrude = 2
                    bpy.context.object.data.offset = 0

                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    space3D.overlay.show_outline_selected = True

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.data.objects.remove(self.Cutter)
                Coll = bpy.data.collections.get("ODENT-4D Cutters")
                if Coll:
                    Hooks = [obj for obj in Coll.objects if "Hook" in obj.name]
                    if Hooks:
                        for obj in Hooks:
                            bpy.data.objects.remove(obj)
                CuttingTargetName = context.scene.ODENT_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if not context.object or not context.object.type == "MESH":

            message = ["Please select the target mesh !"]
            ODENT_GpuDrawText(message)
            sleep(2)
            ODENT_GpuDrawText()
            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Assign Model name to CuttingTarget property :
                CuttingTarget = bpy.context.object
                bpy.context.scene.ODENT_Props.CuttingTargetNameProp = CuttingTarget.name

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget
                # Hide everything but model :
                # bpy.ops.object.hide_view_set(unselected=True)

                CuttingCurveAdd()
                self.Cutter = context.active_object

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class ODENT_OT_CurveCutterCut(bpy.types.Operator):
    "Performe Curve Cutting Operation"

    bl_idname = "wm.odent_curvecuttercut"
    bl_label = "CURVE CUTTER CUT"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ODENT_Props = context.scene.ODENT_Props

        # Get CuttingTarget :
        CuttingTargetName = ODENT_Props.CuttingTargetNameProp
        CuttingTarget = bpy.data.objects.get(CuttingTargetName)
        CurveCuttersList = [
            obj
            for obj in context.visible_objects
            if obj.type == "CURVE" and obj.name.startswith("ODENT_Curve_Cut")
        ]

        if not CurveCuttersList or not CuttingTarget:

            message = [" Please Add Curve Cutters first !"]
            ODENT_GpuDrawText(message)
            sleep(2)
            ODENT_GpuDrawText()
            return {"CANCELLED"}

        else:

            # Get CurveCutter :
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")

            CurveMeshesList = []
            for CurveCutter in CurveCuttersList:
                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                # remove material :
                bpy.ops.object.material_slot_remove_all()

                # convert CurveCutter to mesh :
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.convert(target="MESH")
                CurveMesh = context.object
                CurveMeshesList.append(CurveMesh)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in CurveMeshesList:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            if len(CurveMeshesList) > 1:
                bpy.ops.object.join()

            CurveCutter = context.object

            # CurveCutter.select_set(True)
            # bpy.context.view_layer.objects.active = CurveCutter

            bpy.context.scene.tool_settings.use_snap = False
            bpy.ops.view3d.snap_cursor_to_center()

            # # Make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            curve_vgroup = CurveCutter.vertex_groups.new(name="curve_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            CuttingTarget.select_set(True)
            bpy.context.view_layer.objects.active = CuttingTarget

            # delete old vertex groups :
            CuttingTarget.vertex_groups.clear()

            # deselect all vertices :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            # Join CurveCutter to CuttingTarget :
            CurveCutter.select_set(True)
            bpy.ops.object.join()
            CuttingTarget = context.object
            area3D, space3D, region_3d = CtxOverride(bpy.context)
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):
                bpy.ops.object.hide_view_set(unselected=True)

            # intersect make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.intersect()

            intersect_vgroup = CuttingTarget.vertex_groups.new(name="intersect_vgroup")
            CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_assign()

            # OtherObjList = [obj for obj in bpy.data.objects if obj!= CuttingTarget]
            # hide all but object
            # bpy.ops.object.mode_set(mode="OBJECT")
            # bpy.ops.object.hide_view_set(unselected=True)

            # delete curve_vgroup :
            # bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            curve_vgroup = CuttingTarget.vertex_groups["curve_vgroup"]

            CuttingTarget.vertex_groups.active_index = curve_vgroup.index
            bpy.ops.object.vertex_group_select()
            bpy.ops.mesh.select_more()

            bpy.ops.mesh.delete(type="VERT")

            # # 1st methode :
            SplitSeparator(CuttingTarget=CuttingTarget)
            bpy.data.collections.remove(bpy.data.collections["ODENT-4D Cutters"])

            return {"FINISHED"}


class ODENT_OT_CurveCutter1_New(bpy.types.Operator):
    """mesh curve cutter tool"""

    bl_idname = "wm.odent_curvecutter1_new"
    bl_label = "DRAW CURVE"
    bl_options = {"REGISTER", "UNDO"}

    closeCurve: BoolProperty(name="close cutting curve", default=True)  # type: ignore

    @classmethod
    def poll(cls, context):

        base_mesh = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        if not base_mesh:
            return False
        return (
            context.object.get(OdentConstants.ODENT_TYPE_TAG)
            != OdentConstants.CURVE_CUTTER1_TAG
        )

    def add_curve_cutter(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            # Prepare scene settings :
            bpy.context.scene.tool_settings.use_snap = True
            bpy.context.scene.tool_settings.snap_elements = {"FACE"}

            # ....Add Curve ....... :
            bpy.ops.curve.primitive_bezier_curve_add(
                radius=1, enter_editmode=False, align="CURSOR"
            )
            # Set cutting_tool name :
            self.cutter = bpy.context.view_layer.objects.active
            self.cutter.name = "ODENT_Cutter"
            self.cutter[OdentConstants.ODENT_TYPE_TAG] = (
                OdentConstants.CURVE_CUTTER1_TAG
            )
            self.cutter["odent_target"] = self.base_mesh.name

            # CurveCutter settings :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            self.cutter.data.splines[0].bezier_points[-1].select_control_point = True

            bpy.ops.curve.dissolve_verts()
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

            bpy.context.object.data.dimensions = "3D"
            bpy.context.object.data.twist_smooth = 3
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.context.object.data.bevel_depth = 0.1
            bpy.context.object.data.bevel_resolution = 6
            bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
            bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
            bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
            bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
            bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = (
                True
            )

            # Add color material :
            mat = bpy.data.materials.get(
                "Odent_curve_cutter_mat"
            ) or bpy.data.materials.new("Odent_curve_cutter_mat")
            mat.diffuse_color = [0.1, 1, 0.4, 1.0]
            mat.roughness = 0.3
            bpy.ops.object.mode_set(mode="OBJECT")
            self.cutter.active_material = mat

            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
            space3D.overlay.show_outline_selected = False

            shrinkwrap = self.cutter.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
            shrinkwrap.target = self.base_mesh
            shrinkwrap.wrap_mode = "ABOVE_SURFACE"
            shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Odent Cutters")

    def add_cutter_point(self):
        bpy.context.view_layer.objects.active = self.cutter
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.extrude(mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.curve.select_all(action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(mode="OBJECT")

    def del_cutter_point(self):
        try:
            bpy.context.view_layer.objects.active = self.cutter
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(action="SELECT")
                bpy.ops.curve.handle_type_set(type="AUTOMATIC")
                bpy.ops.curve.select_all(action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass

    def modal(self, context, event):

        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}
        elif event.type == "RET" and self.counter == 0:
            return {"PASS_THROUGH"}
        elif event.type == "DEL" and self.counter == 0:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid:
                    self.add_cutter_point()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:

            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid:
                    self.add_curve_cutter(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:

            if event.value == ("PRESS"):
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):

                    bpy.context.view_layer.objects.active = self.cutter
                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    self.cutter.select_set(True)

                    if self.closeCurve:
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.curve.cyclic_toggle()
                        bpy.ops.object.mode_set(mode="OBJECT")

                    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

                    bpy.context.object.data.bevel_depth = 0
                    bpy.context.object.data.extrude = 2
                    bpy.context.object.data.offset = 0

                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    space3D.overlay.show_outline_selected = True
                ODENT_GpuDrawText()
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.odent.message_box("INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            self.scn = context.scene
            self.counter = 0

            self.start_objects = bpy.data.objects[:]
            self.start_collections = bpy.data.collections[:]
            self.start_visible_objects = bpy.context.visible_objects[:]

            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            self.base_mesh.select_set(True)
            context.view_layer.objects.active = self.base_mesh

            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
            context.window_manager.modal_handler_add(self)
        txt = [
            "Left click : draw curve | DEL : roll back | ESC : to cancell operation",
            "ENTER : to finalise",
        ]
        ODENT_GpuDrawText(txt)
        return {"RUNNING_MODAL"}


class ODENT_OT_CurveCutter1_New_Perform_Cut(bpy.types.Operator):
    "Performe CurveCutter1 Operation"

    bl_idname = "wm.odent_curvecutter1_new_perform_cut"
    bl_label = "CUT"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        cutters = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == "curvecutter1"
        ]
        return cutters

    def execute(self, context):

        start_unvis_objects = [
            obj for obj in context.scene.objects if not obj in context.visible_objects
        ]

        txt = ["Processing ..."]
        ODENT_GpuDrawText(message_list=txt)
        area3D, space3D, region_3d = CtxOverride(context)
        bpy.context.scene.tool_settings.use_snap = False
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.snap_cursor_to_center()

        datadict = {}

        allcurvecutters1 = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == "curvecutter1"
        ]
        for c in allcurvecutters1:
            target_name = c["odent_target"]
            if not datadict.get(target_name):
                datadict.update({target_name: [c.name]})
            else:
                datadict[target_name].append(c.name)

        for target_name, curvecutters1_names in datadict.items():
            if not bpy.data.objects.get(target_name):
                continue
            target = bpy.data.objects.get(target_name)
            target.hide_set(False)
            target.hide_viewport = False
            target.hide_select = False
            curvecutters1 = [
                bpy.data.objects.get(n)
                for n in curvecutters1_names
                if bpy.data.objects.get(n)
            ]
            cutters1_list = []
            for cc in curvecutters1:
                bpy.context.view_layer.objects.active = cc
                cc.hide_set(False)
                cc.hide_viewport = False
                cc.hide_select = False
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                cc.select_set(True)

                # remove material :
                bpy.ops.object.material_slot_remove_all()

                # convert CurveCutter to mesh :
                bpy.ops.object.convert(target="MESH")
                cutter = context.object
                cutters1_list.append(cutter)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in cutters1_list:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            if len(cutters1_list) > 1:
                bpy.ops.object.join()

            cutter = context.object

            # # Make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            curve_vgroup = cutter.vertex_groups.new(name="curve_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            # delete old vertex groups :
            target.vertex_groups.clear()

            # deselect all vertices :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            # Join CurveCutter to CuttingTarget :
            cutter.select_set(True)
            bpy.ops.object.join()
            target = context.object
            # bpy.ops.object.hide_view_set( unselected=True)

            # intersect make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.intersect()

            intersect_vgroup = target.vertex_groups.new(name="intersect_vgroup")
            target.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_assign()

            # OtherObjList = [obj for obj in bpy.data.objects if obj!= CuttingTarget]
            # hide all but object
            # bpy.ops.object.mode_set(mode="OBJECT")
            # bpy.ops.object.hide_view_set(unselected=True)

            # delete curve_vgroup :
            # bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            curve_vgroup = target.vertex_groups["curve_vgroup"]

            target.vertex_groups.active_index = curve_vgroup.index
            bpy.ops.object.vertex_group_select()
            bpy.ops.mesh.select_more()

            bpy.ops.mesh.delete(type="VERT")
            area3D, space3D, region_3d = CtxOverride(bpy.context)
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):
                bpy.ops.object.hide_view_set(unselected=True)
            # # 1st methode :
            SplitSeparator(CuttingTarget=target)
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.data.collections.remove(bpy.data.collections["Odent Cutters"])

        for obj in context.view_layer.objects:
            if obj in start_unvis_objects:
                try:
                    obj.hide_set(True)
                except:
                    pass
            else:
                try:
                    obj.hide_set(False)
                    obj.hide_viewport = False
                    obj.hide_select = False
                except:
                    pass

        txt = ["Done."]
        ODENT_GpuDrawText(message_list=txt, rect_color=OdentColors.green)
        sleep(1)
        ODENT_GpuDrawText()

        return {"FINISHED"}


class ODENT_OT_AddTube(bpy.types.Operator):
    """Add Curve Tube"""

    bl_idname = "wm.odent_add_tube"
    bl_label = "ADD TUBE"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        target = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        return target

    def execute(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            self.odent_props = context.scene.ODENT_Props
            self.target = context.object
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            self.target.select_set(True)

            self.scn = context.scene
            self.counter = 0
            self.target = context.object
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
            context.window_manager.modal_handler_add(self)
        message = ["Please draw Tube, and Press <ENTER>"]
        ODENT_GpuDrawText(message)
        return {"RUNNING_MODAL"}

    def add_tube(self, context):

        # Prepare scene settings :
        bpy.ops.transform.select_orientation(orientation="GLOBAL")
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}
        bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(
            radius=1, enter_editmode=False, align="CURSOR"
        )

        self.tube = context.view_layer.objects.active
        self.tube.name = "_ADD_ODENT_GuideTube"
        curve = self.tube.data
        curve.name = "ODENT_GuideTube"

        guide_components_coll = add_collection(
            OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
        )
        MoveToCollection(self.tube, guide_components_coll.name)

        # Tube settings :
        # dg = context.evaluated_depsgraph_get()
        # temp_obj = self.tube.evaluated_get(dg).copy()
        # temp_obj.data.

        # obj.data = temp_obj.data
        context.view_layer.objects.active = self.tube
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        curve.splines[0].bezier_points[-1].select_control_point = True

        bpy.ops.curve.delete(type="VERT")

        print("tube debug")
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor()
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.object.mode_set(mode="OBJECT")

        curve.dimensions = "3D"
        curve.twist_smooth = 3
        curve.use_fill_caps = True

        curve.bevel_depth = self.odent_props.TubeWidth
        curve.bevel_resolution = 10
        bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat_tube = bpy.data.materials.get(
            "mat_ODENT_guide_tube"
        ) or bpy.data.materials.new("mat_ODENT_guide_tube")
        mat_tube.diffuse_color = [0.03, 0.20, 0.14, 1.0]  # [0.1, 0.4, 1.0, 1.0]
        mat_tube.roughness = 0.3
        mat_tube.use_nodes = True
        nodes = mat_tube.node_tree.nodes
        pbsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
        pbsdf_node.inputs[0].default_value = [0.03, 0.20, 0.14, 1.0]
        # mat_tube.node_tree.nodes["Principled BSDF"].inputs[0].default_value = [
        # 0.03, 0.20, 0.14, 1.0]

        self.tube.active_material = mat_tube

        bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
        bpy.context.space_data.overlay.show_outline_selected = False

        bpy.ops.object.modifier_add(type="SHRINKWRAP")
        bpy.context.object.modifiers["Shrinkwrap"].target = self.target
        bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
        bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True

        os.system("cls") if os.name == "nt" else os.system("clear")

    def add_tube_point(self, context, obj):
        context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.extrude(mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.curve.select_all(action="DESELECT")
        points = obj.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(mode="OBJECT")

    def del_tube_point(self, context, obj):
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            points = obj.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = obj.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(type="VERT")
                points = obj.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(action="SELECT")
                bpy.ops.curve.handle_type_set(type="AUTOMATIC")
                bpy.ops.curve.select_all(action="DESELECT")
                points = obj.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass

    def cancell(self, context):
        try:
            bpy.data.objects.remove(self.tube, do_unlink=True)
        except Exception:
            pass
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
            self.scn.tool_settings.use_snap = False
            space3D.overlay.show_outline_selected = True
            bpy.ops.wm.tool_set_by_id(name="builtin.select")

    def modal(self, context, event):

        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                self.cancell(context)

                message = ["CANCELLED"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

        elif event.type == "RET" and self.counter == 0:
            if event.value == ("PRESS"):

                message = ["Warning : Use left click to draw Tube, and Press <ENTER>"]
                ODENT_GpuDrawText(message)

                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                n = len(self.tube.data.splines[0].bezier_points[:])
                if n <= 1:
                    message = [
                        "Warning : Please draw at least 2 Tube points, and Press <ENTER>"
                    ]
                    ODENT_GpuDrawText(message)

                    return {"RUNNING_MODAL"}

                else:
                    area3D, space3D, region_3d = CtxOverride(context)
                    with bpy.context.temp_override(
                        area=area3D, space_data=space3D, region=region_3d
                    ):
                        bpy.context.view_layer.objects.active = self.tube
                        bpy.ops.object.mode_set(mode="OBJECT")
                        self.tube.select_set(True)
                        if self.odent_props.TubeCloseMode == "Close Tube":
                            bpy.ops.object.mode_set(mode="EDIT")
                            bpy.ops.curve.cyclic_toggle()
                            bpy.ops.object.mode_set(mode="OBJECT")

                        bpy.ops.wm.tool_set_by_id(name="builtin.select")
                        self.scn.tool_settings.use_snap = False
                        space3D.overlay.show_outline_selected = True
                        bpy.ops.wm.tool_set_by_id(name="builtin.select")
                    os.system("cls") if os.name == "nt" else os.system("clear")

                    message = ["Tube created"]
                    ODENT_GpuDrawText(message)
                    sleep(2)
                    ODENT_GpuDrawText()

                    return {"FINISHED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                _, space_data, _ = CtxOverride(context)
                if space_data.type == "VIEW_3D":
                    self.add_tube_point(context, self.tube)
                    os.system("cls") if os.name == "nt" else os.system("clear")
                    return {"RUNNING_MODAL"}
                else:
                    return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                _, space_data, _ = CtxOverride(context)
                if space_data.type == "VIEW_3D":
                    self.add_tube(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                else:
                    return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_tube_point(context, self.tube)
                os.system("cls") if os.name == "nt" else os.system("clear")

            return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}


class ODENT_OT_CurveCutterAdd2(bpy.types.Operator):
    """add curve cutter v2"""

    bl_idname = "wm.odent_curvecutteradd2"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    closeCurve: BoolProperty(name="close cutting curve", default=True)  # type: ignore

    @classmethod
    def poll(cls, context):

        base_mesh = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        if not base_mesh:
            return False
        return context.object.get(OdentConstants.ODENT_TYPE_TAG) not in [
            OdentConstants.CURVE_CUTTER2_TAG,
            OdentConstants.SPLIT_CUTTER_HOOK_POINT,
        ]

    def add_curve_cutter(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(
            radius=1, enter_editmode=False, align="CURSOR"
        )
        # Set cutting_tool name :
        self.cutter = context.object
        self.cutter.name = "ODENT_Cutter"
        self.cutter[OdentConstants.ODENT_TYPE_TAG] = "curvecutter2"
        self.cutter["odent_target"] = self.base_mesh.name
        self.cutter["odent_close_curve"] = self.closeCurve

        # CurveCutter settings :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.curve.dissolve_verts()
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

        self.cutter.data.dimensions = "3D"
        self.cutter.data.twist_smooth = 3

        self.cutter.data.bevel_depth = 0.3
        self.cutter.data.bevel_resolution = 2
        context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat = bpy.data.materials.get(
            "Odent_curve_cutter_mat"
        ) or bpy.data.materials.new("Odent_curve_cutter_mat")
        mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
        mat.roughness = 0.3
        self.cutter.active_material = mat

        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Odent Cutters")
        self.cutters_collection = bpy.data.collections.get("Odent Cutters")

    def add_cutter_point(self):
        bpy.context.view_layer.objects.active = self.cutter
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.extrude(mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.curve.select_all(action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(mode="OBJECT")

    def add_cutter_point_interactive(self):
        self.is_extruding = True
        bpy.context.view_layer.objects.active = self.cutter
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.extrude_move("INVOKE_DEFAULT")
        # bpy.ops.curve.extrude( "INVOKE_DEFAULT")

        # bpy.ops.view3d.snap_selected_to_cursor( use_offset=False)
        # bpy.ops.curve.select_all( action="SELECT")
        # bpy.ops.curve.handle_type_set( type="AUTOMATIC")
        # bpy.ops.curve.select_all( action="DESELECT")
        # points = self.cutter.data.splines[0].bezier_points[:]
        # points[-1].select_control_point = True
        # bpy.ops.object.mode_set( mode="OBJECT")

    def del_cutter_point(self):
        try:
            bpy.context.view_layer.objects.active = self.cutter
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.select_all(action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(action="SELECT")
                bpy.ops.curve.handle_type_set(type="AUTOMATIC")
                bpy.ops.curve.select_all(action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass

    def AddCurveSphere(self, context, Name, i, CollName):
        bpy.ops.object.select_all(action="DESELECT")
        bezier_points = self.cutter.data.splines[0].bezier_points[:]
        Bpt = bezier_points[i]
        loc = self.cutter.matrix_world @ Bpt.co
        Hook = AddMarkupPoint(
            name=Name, color=(0, 1, 0, 1), loc=loc, Diameter=0.5, CollName=CollName
        )
        Hook[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.SPLIT_CUTTER_HOOK_POINT

        bpy.context.view_layer.objects.active = Hook
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        Hook.select_set(True)
        self.cutter.select_set(True)
        bpy.context.view_layer.objects.active = self.cutter
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        bezier_points = self.cutter.data.splines[0].bezier_points[:]
        Bpt = bezier_points[i]
        Bpt.select_control_point = True
        bpy.ops.object.hook_add_selob(use_bone=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.cutter.select_set(True)
        bpy.context.view_layer.objects.active = self.cutter

        return Hook

    def modal(self, context, event):

        if not event.type in ["RET", "ESC", "LEFTMOUSE", "RIGHTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}
        elif event.type == "RET" and self.counter == 0:
            return {"PASS_THROUGH"}
        elif event.type == "DEL" and self.counter == 0:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}
        elif event.type == ("RIGHTMOUSE") and self.counter == 1:
            if not self.is_extruding:
                if event.value == ("PRESS"):
                    return {"RUNNING_MODAL"}
                elif event.value == ("RELEASE"):
                    self.add_cutter_point_interactive()
        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                if self.is_extruding:
                    self.is_extruding = False
                    return {"PASS_THROUGH"}

                _is_valid = click_is_in_view3d(context, event)
                # print(_is_valid)
                if _is_valid:
                    self.add_cutter_point()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:

            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                # print(_is_valid)
                if _is_valid:
                    self.add_curve_cutter(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:

            if event.value == ("PRESS"):
                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):

                    bpy.context.view_layer.objects.active = self.cutter
                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    self.cutter.select_set(True)

                    if self.closeCurve:
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.curve.cyclic_toggle()
                        bpy.ops.object.mode_set(mode="OBJECT")

                    # bpy.ops.object.modifier_apply( modifier="Shrinkwrap")

                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    space3D.overlay.show_outline_selected = True

                    bezier_points = self.cutter.data.splines[0].bezier_points[:]
                    Hooks = [
                        obj
                        for obj in self.cutters_collection.objects
                        if "Hook" in obj.name
                    ]
                    for i in range(len(bezier_points)):
                        Hook = self.AddCurveSphere(
                            context,
                            Name=f"Hook_{i}",
                            i=i,
                            CollName="Odent Cutters",
                        )
                        Hooks.append(Hook)
                        hook_modifier = self.cutter.modifiers[-1]
                        bpy.ops.object.modifier_move_up(modifier=hook_modifier.name)
                    # print(Hooks)
                    # for h in Hooks:
                    #     for o in Hooks:
                    #         if not o is h:
                    #             delta = o.location - h.location

                    #             distance = sqrt(
                    #                 delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2
                    #             )
                    #             if distance <= 0.5:
                    #                 # center = h.location + (delta/2)
                    #                 o.location = h.location #= center
                    bpy.context.space_data.overlay.show_relationship_lines = False
                    bpy.context.scene.tool_settings.use_snap = True
                    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                    bpy.context.scene.tool_settings.snap_target = "CENTER"
                    bpy.ops.object.select_all(action="DESELECT")

                    self.cutter.hide_select = True
                ODENT_GpuDrawText()
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            self.is_extruding = False
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.odent.message_box("INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            self.scn = context.scene
            self.counter = 0

            self.start_objects = bpy.data.objects[:]
            self.start_collections = bpy.data.collections[:]
            self.start_visible_objects = bpy.context.visible_objects[:]

            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            self.base_mesh.select_set(True)
            context.view_layer.objects.active = self.base_mesh

            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
            context.window_manager.modal_handler_add(self)
        txt = [
            "Left click : draw curve | DEL : roll back | ESC : to cancell operation",
            "ENTER : to finalise",
        ]
        ODENT_GpuDrawText(txt)
        return {"RUNNING_MODAL"}


class ODENT_OT_CurveCutter2_Cut_New(bpy.types.Operator):
    "Performe CurveCutter2 Operation"

    bl_idname = "wm.odent_curvecutter2_cut_new"
    bl_label = "Cut"
    bl_options = {"REGISTER", "UNDO"}

    Resolution: IntProperty(
        name="Cut Resolution",
        description="Cutting curve Resolution",
        default=2,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        cutters = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == "curvecutter2"
        ]
        return cutters

    def split_cut(self, context):
        bpy.context.view_layer.objects.active = self.target
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        intersect_vgroup = self.target.vertex_groups["intersect_vgroup"]
        self.target.vertex_groups.active_index = intersect_vgroup.index
        bpy.ops.object.vertex_group_select()

        bpy.ops.mesh.edge_split(type="VERT")

        # Separate by loose parts :
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.mesh.separate(type="LOOSE")

        for obj in bpy.context.selected_objects:
            if not obj.data or not obj.data.polygons or len(obj.data.polygons) < 10:
                bpy.data.objects.remove(obj)
            else:
                context.view_layer.objects.active = obj

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles()
                bpy.ops.mesh.select_all(action="DESELECT")

    def execute(self, context):

        start = tpc()
        start_unvis_objects = [
            obj for obj in context.scene.objects if not obj in context.visible_objects
        ]

        txt = ["Processing ..."]
        ODENT_GpuDrawText(message_list=txt)
        area3D, space3D, region_3d = CtxOverride(context)

        datadict = {}

        allcurvecutters2 = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == "curvecutter2"
        ]
        for c in allcurvecutters2:
            target_name = c["odent_target"]
            if not datadict.get(target_name):
                datadict.update({target_name: [c.name]})
            else:
                datadict[target_name].append(c.name)

        for target_name, curvecutters2_names in datadict.items():
            if not bpy.data.objects.get(target_name):
                continue
            self.target = target = bpy.data.objects.get(target_name)
            hide_object(False, target)
            curvecutters2 = [
                bpy.data.objects.get(n)
                for n in curvecutters2_names
                if bpy.data.objects.get(n)
            ]
            cutters2_list = []
            for cc in curvecutters2:
                hide_object(False, cc)
                context.view_layer.objects.active = cc

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                cc.select_set(True)

                cc.data.bevel_depth = 0
                cc.data.resolution_u = self.Resolution

                # hook_modifiers = [
                # mod.name for mod in cc.modifiers if "Hook" in mod.name
                # ]
                # for mod in hook_modifiers:
                #     bpy.ops.object.modifier_apply(modifier=mod)

                # cc.data.bevel_depth = 0
                # cc.data.resolution_u = self.Resolution
                # bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

                # convert CurveCutter to mesh :
                odent_close_curve = cc["odent_close_curve"]
                bpy.ops.object.convert(target="MESH")
                cutter = context.object
                cutter["odent_close_curve"] = odent_close_curve
                cutters2_list.append(cutter)

            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            context.view_layer.objects.active = target
            me = target.data
            # initiate a KDTree :
            size = len(me.vertices)
            kd = kdtree.KDTree(size)

            for v_id, v in enumerate(me.vertices):
                kd.insert(v.co, v_id)

            kd.balance()

            Loop = []
            for cc_mesh in cutters2_list:

                CutterCoList = [
                    target.matrix_world.inverted() @ cc_mesh.matrix_world @ v.co
                    for v in cc_mesh.data.vertices
                ]
                Closest_VIDs = [
                    kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))
                ]
                if cc_mesh["odent_close_curve"]:
                    CloseState = True
                else:
                    CloseState = False

                # CutLine = ShortestPath(
                #     target, Closest_VIDs, close=CloseState)
                CutLine = ConnectPath(target, Closest_VIDs, close=CloseState)

                Loop.extend(CutLine)

            # print(Loop)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")

            bpy.ops.object.mode_set(mode="OBJECT")
            for Id in Loop:
                me.vertices[Id].select = True

            for vg in target.vertex_groups:
                target.vertex_groups.remove(vg)

            bpy.ops.object.mode_set(mode="EDIT")
            vg = target.vertex_groups.new(name="intersect_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # print("Split Cut Line...")

            # Split :
            self.split_cut(context)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")

        print("Remove Cutter tool...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        for obj in context.visible_objects:
            if obj.type == "MESH" and len(obj.data.polygons) <= 10:
                bpy.data.objects.remove(obj)
        col = bpy.data.collections["Odent Cutters"]
        for obj in col.objects:
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(col)

        finish = tpc()

        context.scene.tool_settings.use_snap = False
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.snap_cursor_to_center()

        print("finished in : ", finish - start, "secondes")
        ODENT_GpuDrawText()
        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_CurveCutter2_ShortPath(bpy.types.Operator):
    "Shortpath Curve Cutting tool"

    bl_idname = "wm.odent_curvecutter2_shortpath"
    bl_label = "ShortPath"
    bl_options = {"REGISTER", "UNDO"}

    Resolution: IntProperty(
        name="Cut Resolution",
        description="Cutting curve Resolution",
        default=3,
    )  # type: ignore

    def execute(self, context):

        start = tpc()
        ODENT_Props = bpy.context.scene.ODENT_Props
        ###########################################################################

        CuttingTarget = self.CuttingTarget
        CurveCuttersList = self.CurveCuttersList

        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        CuttingTarget.hide_select = False
        # delete old vertex groups :
        CuttingTarget.vertex_groups.clear()

        CurveMeshesList = []
        for CurveCutter in CurveCuttersList:
            CurveCutter.hide_select = False
            bpy.ops.object.select_all(action="DESELECT")
            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

            HookModifiers = [
                mod.name for mod in CurveCutter.modifiers if "Hook" in mod.name
            ]
            for mod in HookModifiers:
                bpy.ops.object.modifier_apply(modifier=mod)

            CurveCutter.data.bevel_depth = 0
            CurveCutter.data.resolution_u = self.Resolution
            bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

            bpy.ops.object.convert(target="MESH")
            CurveCutter = context.object
            bpy.ops.object.modifier_add(type="SHRINKWRAP")
            CurveCutter.modifiers["Shrinkwrap"].target = CuttingTarget
            bpy.ops.object.convert(target="MESH")

            CurveMesh = context.object
            CurveMeshesList.append(CurveMesh)

        bpy.ops.object.select_all(action="DESELECT")
        CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = CuttingTarget
        me = CuttingTarget.data
        # initiate a KDTree :
        size = len(me.vertices)
        kd = kdtree.KDTree(size)

        for v_id, v in enumerate(me.vertices):
            kd.insert(v.co, v_id)

        kd.balance()
        Loop = []
        for CurveCutter in CurveMeshesList:

            CutterCoList = [
                CuttingTarget.matrix_world.inverted() @ CurveCutter.matrix_world @ v.co
                for v in CurveCutter.data.vertices
            ]
            Closest_VIDs = [
                kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))
            ]
            print("Get closest verts list done")
            if ODENT_Props.CurveCutCloseMode == "Close Curve":
                CloseState = True
            else:
                CloseState = False
            CutLine = ShortestPath(CuttingTarget, Closest_VIDs, close=CloseState)
            Loop.extend(CutLine)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")
        for Id in Loop:
            me.vertices[Id].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        vg = CuttingTarget.vertex_groups.new(name="intersect_vgroup")
        bpy.ops.object.vertex_group_assign()

        print("Shrinkwrap Modifier...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for CurveCutter in CurveMeshesList:
            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

        if len(CurveMeshesList) > 1:
            bpy.ops.object.join()

        CurveCutter = context.object
        print("CurveCutter", CurveCutter)
        print("CuttingTarget", CuttingTarget)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = CuttingTarget

        bpy.ops.object.modifier_add(type="SHRINKWRAP")
        CuttingTarget.modifiers["Shrinkwrap"].wrap_method = "NEAREST_VERTEX"
        CuttingTarget.modifiers["Shrinkwrap"].vertex_group = vg.name
        CuttingTarget.modifiers["Shrinkwrap"].target = CurveCutter
        bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

        print("Relax Cut Line...")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.odent.looptools_relax(
            input="selected", interpolation="cubic", iterations="3", regular=True
        )

        print("Split Cut Line...")

        # Split :
        SplitSeparator(CuttingTarget=CuttingTarget)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")

        print("Remove Cutter tool...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        for obj in context.visible_objects:
            if obj.type == "MESH" and len(obj.data.polygons) <= 10:
                bpy.data.objects.remove(obj)
        col = bpy.data.collections["ODENT-4D Cutters"]
        for obj in col.objects:
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(col)

        finish = tpc()
        print("finished in : ", finish - start, "secondes")

        return {"FINISHED"}

    def invoke(self, context, event):

        ODENT_Props = bpy.context.scene.ODENT_Props

        # Get CuttingTarget :
        CuttingTargetName = ODENT_Props.CuttingTargetNameProp
        self.CuttingTarget = bpy.data.objects.get(CuttingTargetName)

        # Get CurveCutter :
        self.CurveCuttersList = [
            obj for obj in bpy.data.objects if "ODENT_Curve_Cut2" in obj.name
        ]

        if not self.CurveCuttersList or not self.CuttingTarget:

            message = [" Please Add Curve Cutters first !"]
            ODENT_GpuDrawText(message)
            sleep(2)
            ODENT_GpuDrawText()
            return {"CANCELLED"}
        else:

            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_CurveCutterAdd3(bpy.types.Operator):
    """description of this Operator"""

    bl_idname = "wm.odent_curvecutteradd3"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        ODENT_Props = context.scene.ODENT_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                CurveCutterName = ODENT_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                bpy.ops.object.mode_set(mode="OBJECT")

                if ODENT_Props.CurveCutCloseMode == "Close Curve":
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                # bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Shrinkwrap")

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                # bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                bezier_points = CurveCutter.data.splines[0].bezier_points[:]
                for i in range(len(bezier_points)):
                    AddCurveSphere(
                        Name=f"Hook_{i}",
                        Curve=CurveCutter,
                        i=i,
                        CollName="ODENT-4D Cutters",
                    )
                bpy.context.space_data.overlay.show_relationship_lines = False

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                CurveCutterName = bpy.context.scene.ODENT_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter
                bpy.ops.object.delete(use_global=False, confirm=False)

                CuttingTargetName = context.scene.ODENT_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.space_data.overlay.show_outline_selected = True
                bpy.context.scene.tool_settings.snap_target = "CENTER"
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.space_data.overlay.show_relationship_lines = False

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if bpy.context.selected_objects == []:

            message = ["Please select the target object !"]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Assign Model name to CuttingTarget property :
                CuttingTarget = bpy.context.view_layer.objects.active
                bpy.context.scene.ODENT_Props.CuttingTargetNameProp = CuttingTarget.name

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.hide_view_clear()
                bpy.ops.object.select_all(action="DESELECT")

                # for obj in bpy.data.objects:
                #     if "CuttingCurve" in obj.name:
                #         obj.select_set(True)
                #         bpy.ops.object.delete(use_global=False, confirm=False)

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget
                # Hide everything but model :
                area3D, space3D, region_3d = CtxOverride(bpy.context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.object.hide_view_set(unselected=True)

                CuttingCurveAdd2()

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class ODENT_OT_CurveCutterCut3(bpy.types.Operator):
    "Performe Curve Cutting Operation"

    bl_idname = "wm.odent_curvecuttercut3"
    bl_label = "CURVE CUTTER CUT"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        # Get CuttingTarget :
        CuttingTargetName = bpy.context.scene.ODENT_Props.CuttingTargetNameProp
        CuttingTarget = bpy.data.objects.get(CuttingTargetName)

        # Get CurveCutter :
        bpy.ops.object.select_all(action="DESELECT")

        CurveCuttersList = [
            obj
            for obj in context.visible_objects
            if obj.type == "CURVE" and obj.name.startswith("ODENT_Curve_Cut")
        ]

        if not CurveCuttersList:

            message = [
                " Can't find curve Cutters ",
                "Please ensure curve Cutters are not hiden !",
            ]

            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        if CurveCuttersList:
            CurveMeshesList = []
            for CurveCutter in CurveCuttersList:
                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                # remove material :
                bpy.ops.object.material_slot_remove_all()

                # Change CurveCutter setting   :
                CurveCutter.data.bevel_depth = 0
                CurveCutter.data.resolution_u = 6

                # Add shrinkwrap modif outside :
                bpy.ops.object.modifier_add(type="SHRINKWRAP")
                CurveCutter.modifiers["Shrinkwrap"].use_apply_on_spline = True
                CurveCutter.modifiers["Shrinkwrap"].target = CuttingTarget
                CurveCutter.modifiers["Shrinkwrap"].offset = 0.5
                CurveCutter.modifiers["Shrinkwrap"].wrap_mode = "OUTSIDE"

                # duplicate curve :
                bpy.ops.object.duplicate_move()
                CurveCutterDupli = context.object
                CurveCutterDupli.modifiers["Shrinkwrap"].wrap_mode = "INSIDE"
                CurveCutterDupli.modifiers["Shrinkwrap"].offset = 0.8

                IntOut = []
                for obj in [CurveCutter, CurveCutterDupli]:
                    # convert CurveCutter to mesh :
                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.convert(target="MESH")
                    CurveMesh = context.object
                    IntOut.append(CurveMesh)

                bpy.ops.object.select_all(action="DESELECT")
                for obj in IntOut:
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                bpy.ops.object.join()
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bridge_edge_loops()
                bpy.ops.object.mode_set(mode="OBJECT")
                CurveMeshesList.append(context.object)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in CurveMeshesList:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj

            if len(CurveMeshesList) > 1:
                bpy.ops.object.join()

            CurveCutter = context.object

            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

            bpy.context.scene.tool_settings.use_snap = False
            bpy.ops.view3d.snap_cursor_to_center()

            # # Make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.mesh.select_all(action="SELECT")
            curve_vgroup = CurveCutter.vertex_groups.new(name="curve_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            CuttingTarget.select_set(True)
            bpy.context.view_layer.objects.active = CuttingTarget

            # delete old vertex groups :
            CuttingTarget.vertex_groups.clear()

            # deselect all vertices :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            ###############################################################

            # Join CurveCutter to CuttingTarget :
            CurveCutter.select_set(True)
            bpy.ops.object.join()
            area3D, space3D, region_3d = CtxOverride(bpy.context)
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):
                bpy.ops.object.hide_view_set(unselected=True)

            # intersect make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.intersect()

            intersect_vgroup = CuttingTarget.vertex_groups.new(name="intersect_vgroup")
            CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_assign()

            # # delete curve_vgroup :
            # bpy.ops.object.mode_set(mode="EDIT")
            # bpy.ops.mesh.select_all(action="DESELECT")
            # curve_vgroup = CuttingTarget.vertex_groups["curve_vgroup"]

            # CuttingTarget.vertex_groups.active_index = curve_vgroup.index
            # bpy.ops.object.vertex_group_select()
            # bpy.ops.mesh.delete(type="FACE")

            # bpy.ops.ed.undo_push()
            # # 1st methode :
            # SplitSeparator(CuttingTarget=CuttingTarget)

            # for obj in context.visible_objects:
            #     if len(obj.data.polygons) <= 10:
            #         bpy.data.objects.remove(obj)
            # for obj in context.visible_objects:
            #     if obj.name.startswith("Hook"):
            #         bpy.data.objects.remove(obj)

            # print("Cutting done with first method")

            return {"FINISHED"}


class ODENT_OT_AddSquareCutter(bpy.types.Operator):
    """Square Cutting Tool add"""

    bl_idname = "wm.odent_add_square_cutter"
    bl_label = "Square Cut"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and context.object.type == "MESH"
            and context.object.select_get()
        )

    def add_square_cutter(self, context):

        _, space_data, _ = CtxOverride(context)
        context.view_layer.objects.active = self.target
        bpy.ops.object.mode_set(mode="OBJECT")

        mesh_center = np.mean(
            [self.target.matrix_world @ v.co for v in self.target.data.vertices], axis=0
        )
        view_rotation_4x4 = space_data.region_3d.view_rotation.to_matrix().to_4x4()
        dim = max(self.target.dimensions) * 1.5
        # Add cube :
        bpy.ops.mesh.primitive_cube_add(size=dim)

        square_cutter = context.object
        for obj in bpy.data.objects:
            if "SquareCutter" in obj.name:
                bpy.data.objects.remove(obj)
        square_cutter.name = "SquareCutter"

        # Reshape and align cube :

        square_cutter.matrix_world = view_rotation_4x4

        square_cutter.location = mesh_center

        square_cutter.display_type = "WIRE"
        square_cutter.scale[1] = 0.5
        square_cutter.scale[2] = 2

        # Subdivide cube 10 iterations 3 times :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=100)

        # Make cube normals consistent :
        bpy.ops.object.mode_set(mode="OBJECT")

        return square_cutter.name

    def cut(self, context):
        cutting_mode = context.scene.ODENT_Props.cutting_mode
        bpy.context.view_layer.objects.active = self.target
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)

        # Add Boolean Modifier :
        bool_modif = self.target.modifiers.new(name="Boolean", type="BOOLEAN")
        bool_modif.object = self.square_cutter

        # Apply boolean modifier :
        if cutting_mode == "Cut inner":
            bool_modif.operation = "DIFFERENCE"

        if cutting_mode == "Keep inner":
            bool_modif.operation = "INTERSECT"

        bpy.ops.object.convert(target="MESH")

        bpy.ops.object.select_all(action="DESELECT")
        self.square_cutter.select_set(True)
        bpy.context.view_layer.objects.active = self.square_cutter

    def modal(self, context, event):
        if not event.type in {"RET", "ESC"}:
            return {"PASS_THROUGH"}
        if event.type == "RET":
            if event.value == ("PRESS"):
                self.square_cutter = bpy.data.objects.get(self.square_cutter_name)
                if not self.square_cutter:
                    message = ["Cancelled, No Square Cutter Found ..."]
                    ODENT_GpuDrawText(message)
                    sleep(2)
                    ODENT_GpuDrawText()
                    return {"CANCELLED"}
                else:
                    txt = ["Processing ..."]
                    ODENT_GpuDrawText(message_list=txt)
                    self.cut(context)
                    txt = ["Square Cut Done ..."]
                    ODENT_GpuDrawText(message_list=txt, rect_color=OdentColors.green)
                    sleep(2)
                    txt = ["ENTER : to make another cut / ESC : to resume."]
                    ODENT_GpuDrawText(message_list=txt)
                    return {"RUNNING_MODAL"}

        elif event.type == ("ESC"):
            self.square_cutter = bpy.data.objects.get(self.square_cutter_name)
            message = ["Finished"]
            try:
                bpy.data.objects.remove(self.square_cutter)
            except:
                pass

            for obj in self.start_visible_objects:
                obj.hide_set(False)
            ODENT_GpuDrawText(message)
            sleep(2)
            ODENT_GpuDrawText()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def execute(self, context):

        # if context.space_data.type == "VIEW_3D":
        self.target = context.object
        bpy.ops.object.mode_set(mode="OBJECT")
        self.start_visible_objects = context.visible_objects[:]

        for obj in self.start_visible_objects:
            if not obj is self.target:
                obj.hide_set(True)

        self.square_cutter_name = self.add_square_cutter(context)

        message = [
            " Press <ENTER> to Cut, <ESC> to exit",
        ]
        ODENT_GpuDrawText(message)

        context.window_manager.modal_handler_add(self)

        return {"RUNNING_MODAL"}


class ODENT_OT_square_cut_confirm(bpy.types.Operator):
    """confirm Square Cut operation"""

    bl_idname = "wm.odent_square_cut_confirm"
    bl_label = "Tirm"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.scene["square_cutter_target"]:

            message = ["Please add square cutter first !"]
            ODENT_GpuDrawText(message)
            sleep(2)
            ODENT_GpuDrawText()
            return {"CANCELLED"}

        elif not context.scene["square_cutter"]:
            message = ["Cancelled, can't find the square cutter !"]
            ODENT_GpuDrawText(message)
            sleep(2)
            ODENT_GpuDrawText()

        else:

            cutting_mode = context.scene.ODENT_Props.cutting_mode
            target = bpy.data.objects.get(context.scene["square_cutter_target"])
            if not target:
                message = ["Cancelled, can't find the target !"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}
            cutter = bpy.data.objects.get(context.scene["square_cutter"])
            if not cutter:
                message = ["Cancelled, can't find the cutter !"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
            bpy.ops.object.mode_set(mode="OBJECT")

            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            # Make Model normals consitent :

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            # Add Boolean Modifier :
            bool_modif = target.modifiers.new(name="Boolean", type="BOOLEAN")
            bool_modif.object = cutter

            # Apply boolean modifier :
            if cutting_mode == "Cut inner":
                bool_modif.operation = "DIFFERENCE"
                bpy.ops.object.modifier_apply(modifier="Boolean")

            if cutting_mode == "Keep inner":
                bool_modif.operation = "INTERSECT"
            bpy.ops.object.convert(target="MESH")

            # Delete resulting loose geometry :
            bpy.data.objects.remove(cutter, do_unlink=True)

            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            return {"FINISHED"}


class ODENT_OT_square_cut_exit(bpy.types.Operator):
    """Square Cutting Tool Exit"""

    bl_idname = "wm.odent_square_cut_exit"
    bl_label = "Exit"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        # Delete frame :
        try:

            frame = bpy.data.objects["my_frame_cutter"]
            bpy.ops.object.select_all(action="DESELECT")
            frame.select_set(True)

            bpy.ops.object.select_all(action="INVERT")
            Model = bpy.context.selected_objects[0]

            bpy.ops.object.select_all(action="DESELECT")
            frame.select_set(True)
            bpy.context.view_layer.objects.active = frame

            bpy.ops.object.delete(use_global=False, confirm=False)

            bpy.ops.object.select_all(action="DESELECT")
            Model.select_set(True)
            bpy.context.view_layer.objects.active = Model

        except Exception:
            pass

        return {"FINISHED"}


class ODENT_OT_PaintArea(bpy.types.Operator):
    """Vertex paint area context toggle"""

    bl_idname = "wm.odent_paintarea_toggle"
    bl_label = "PAINT AREA"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ActiveObj = context.active_object
        if not ActiveObj:

            message = ["Please select the target object !"]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            condition = ActiveObj.type == "MESH" and ActiveObj.select_get() == True

            if not condition:

                message = ["Please select the target object !"]
                icon = "COLORSET_02_VEC"
                bpy.ops.wm.odent_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )

                return {"CANCELLED"}

            else:

                bpy.ops.object.mode_set(mode="VERTEX_PAINT")
                bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")

                DrawBrush = bpy.data.brushes.get("Draw")
                DrawBrush.blend = "MIX"
                DrawBrush.color = (0.0, 1.0, 0.0)
                DrawBrush.strength = 1.0
                DrawBrush.use_frontface = True
                DrawBrush.use_alpha = True
                DrawBrush.stroke_method = "SPACE"
                DrawBrush.curve_preset = "CUSTOM"
                DrawBrush.cursor_color_add = (0.0, 0.0, 1.0, 0.9)
                DrawBrush.use_cursor_overlay = True

                bpy.context.tool_settings.vertex_paint.tool_slots[0].brush = DrawBrush

                for vg in ActiveObj.vertex_groups:
                    ActiveObj.vertex_groups.remove(vg)

                for VC in ActiveObj.data.vertex_colors:
                    ActiveObj.data.vertex_colors.remove(VC)

                ActiveObj.data.vertex_colors.new(name="ODENT_PaintCutter_VC")

                return {"FINISHED"}


class ODENT_OT_PaintAreaPlus(bpy.types.Operator):
    """Vertex paint area Paint Plus toggle"""

    bl_idname = "wm.odent_paintarea_plus"
    bl_label = "PLUS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            _, space3D, _ = CtxOverride(context)
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")
            DrawBrush = bpy.data.brushes.get("Draw")
            context.tool_settings.vertex_paint.tool_slots[0].brush = DrawBrush
            DrawBrush.blend = "MIX"
            DrawBrush.color = (0.0, 1.0, 0.0)
            DrawBrush.strength = 1.0
            DrawBrush.use_frontface = True
            DrawBrush.use_alpha = True
            DrawBrush.stroke_method = "SPACE"
            DrawBrush.curve_preset = "CUSTOM"
            DrawBrush.cursor_color_add = (0.0, 0.0, 1.0, 0.9)
            DrawBrush.use_cursor_overlay = True
            space3D.show_region_header = False
            space3D.show_region_header = True

            return {"FINISHED"}


class ODENT_OT_PaintAreaMinus(bpy.types.Operator):
    """Vertex paint area Paint Minus toggle"""

    bl_idname = "wm.odent_paintarea_minus"
    bl_label = "MINUS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            _, space3D, _ = CtxOverride(context)
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")
            LightenBrush = bpy.data.brushes.get("Lighten")
            context.tool_settings.vertex_paint.tool_slots[0].brush = LightenBrush
            LightenBrush.blend = "MIX"
            LightenBrush.color = (1.0, 1.0, 1.0)
            LightenBrush.strength = 1.0
            LightenBrush.use_frontface = True
            LightenBrush.use_alpha = True
            LightenBrush.stroke_method = "SPACE"
            LightenBrush.curve_preset = "CUSTOM"
            LightenBrush.cursor_color_add = (1, 0.0, 0.0, 0.9)
            LightenBrush.use_cursor_overlay = True
            space3D.show_region_header = False
            space3D.show_region_header = True

            return {"FINISHED"}


class ODENT_OT_PaintCut(bpy.types.Operator):
    """Vertex paint Cut"""

    bl_idname = "wm.odent_paint_cut"
    bl_label = "CUT"
    bl_options = {"REGISTER", "UNDO"}

    Cut_Modes_List = ["Cut", "Make Copy (Shell)", "Remove Painted", "Keep Painted"]
    items = []
    for i in range(len(Cut_Modes_List)):
        item = (str(Cut_Modes_List[i]), str(Cut_Modes_List[i]), str(""), int(i))
        items.append(item)

    Cut_Mode_Prop: EnumProperty(
        name="Cut Mode", items=items, description="Cut Mode", default="Cut"
    )  # type: ignore

    def execute(self, context):

        VertexPaintCut(mode=self.Cut_Mode_Prop)
        bpy.ops.ed.undo_push(message="ODENT Paint Cutter")

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class ODENT_OT_XrayToggle(bpy.types.Operator):
    """ """

    bl_idname = "wm.odent_xray_toggle"
    bl_label = "2D Image to 3D Matching"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.view3d.toggle_xray()

        return {"FINISHED"}


class ODENT_OT_FilpCameraAxial90Plus(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_axial_90_plus"
    bl_label = "Axial 90"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        camera_axial = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_axial, axis="Z", angle=90)
        child_of = camera_axial.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraAxial90Minus(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_axial_90_minus"
    bl_label = "Axial 90"
    bl_options = {"REGISTER", "UNDO"}
    
    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        camera_axial = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_axial, axis="Z", angle=-90)
        child_of = camera_axial.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraAxialUpDown(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_axial_up_down"
    bl_label = "Axial Up/Down"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        camera_axial = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_axial, axis="X", angle=180)
        child_of = camera_axial.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraAxialLeftRight(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_axial_left_right"
    bl_label = "Axial L/R"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "axial" in obj.name.lower()
        ]
        camera_axial = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_axial, axis="Y", angle=180)
        child_of = camera_axial.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraCoronal90Plus(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_coronal_90_plus"
    bl_label = "Coronal 90"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        camera_coronal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_coronal, axis="Y", angle=-90)
        child_of = camera_coronal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraCoronal90Minus(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_coronal_90_minus"
    bl_label = "Coronal 90"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        camera_coronal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_coronal, axis="Y", angle=90)
        child_of = camera_coronal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraCoronalUpDown(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_coronal_up_down"
    bl_label = "Coronal Up/Down"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        camera_coronal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_coronal, axis="X", angle=180)
        child_of = camera_coronal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraCoronalLeftRight(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_coronal_left_right"
    bl_label = "Coronal L/R"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "coronal" in obj.name.lower()
        ]
        camera_coronal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_coronal, axis="Z", angle=180)
        child_of = camera_coronal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraSagittal90Plus(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_sagittal_90_plus"
    bl_label = "Sagittal 90"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        camera_sagittal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_sagittal, axis="X", angle=-90)
        child_of = camera_sagittal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraSagittal90Minus(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_sagittal_90_minus"
    bl_label = "Sagittal 90"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        camera_sagittal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_sagittal, axis="X", angle=90)
        child_of = camera_sagittal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraSagittalUpDown(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_sagittal_up_down"
    bl_label = "Sagittal Up/Down"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        camera_sagittal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(target=slices_pointer, obj=camera_sagittal, axis="Y", angle=180)
        child_of = camera_sagittal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_FilpCameraSagittalLeftRight(bpy.types.Operator):
    bl_idname = "wm.odent_flip_camera_sagittal_left_right"
    bl_label = "Sagittal L/R"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]
        return camera_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICE_CAM_TYPE
            and "sagittal" in obj.name.lower()
        ]
        camera_sagittal = camera_checklist[0]

        slices_pointer_checklist = [
            obj
            for obj in context.scene.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

        slices_pointer = slices_pointer_checklist[0]

        bpy.ops.object.select_all(action="DESELECT")
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)
        rotate_local(target=slices_pointer, obj=camera_sagittal, axis="Z", angle=180)
        child_of = camera_sagittal.constraints.new(type="CHILD_OF")
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ODENT_OT_NormalsToggle(bpy.types.Operator):
    """Mesh check normals"""

    bl_idname = "wm.odent_normals_toggle"
    bl_label = "NORMALS TOGGLE"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and len(context.selected_objects) == 1
            and context.object.type == "MESH"
        )

    def execute(self, context):
        bpy.context.space_data.overlay.show_face_orientation = bool(
            int(bpy.context.space_data.overlay.show_face_orientation) - 1
        )

        return {"FINISHED"}


class ODENT_OT_FlipNormals(bpy.types.Operator):
    """Mesh filp normals"""

    bl_idname = "wm.odent_flip_normals"
    bl_label = "FLIP NORMALS"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object
            and len(context.selected_objects) == 1
            and context.object.type == "MESH"
        )

    def execute(self, context):
        obj = context.object
        mode = obj.mode
        if obj == "EDIT":
            bpy.ops.mesh.flip_normals()
        else:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.flip_normals()
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode=mode)
        return {"FINISHED"}


class ODENT_OT_SlicesPointerSelect(bpy.types.Operator):
    """select slices pointer"""

    bl_idname = "wm.odent_slices_pointer_select"
    bl_label = "SELECT SLICES POINTER"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return [
            obj
            for obj in bpy.data.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ]

    def execute(self, context):

        slices_pointer = [
            obj
            for obj in bpy.data.objects
            if obj.get(OdentConstants.ODENT_TYPE_TAG)
            == OdentConstants.SLICES_POINTER_TYPE
        ][0]

        hide_collection(
            _hide=False, colname=OdentConstants.SLICES_POINTER_COLLECTION_NAME
        )
        hide_object(_hide=False, obj=slices_pointer)
        for obj in context.scene.objects:
            try:
                obj.select_set(False)
            except:
                pass
        # bpy.ops.object.select_all(action='DESELECT')
        slices_pointer.select_set(True)
        context.view_layer.objects.active = slices_pointer

        return {"FINISHED"}


class ODENT_OT_OverhangsPreview(bpy.types.Operator):
    "Survey the model from view"

    bl_idname = "wm.odent_overhangs_preview"
    bl_label = "Preview Overhangs"
    bl_options = {"REGISTER", "UNDO"}

    overhangs_color = [1, 0.2, 0.2, 1.0]
    angle: bpy.props.FloatProperty(
        name="Angle", description="Overhangs Angle", default=45, min=0, max=90
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return (
            context.object.type == "MESH"
            and context.object.mode == "OBJECT"
            and context.object.select_get()
        )

    def compute_overhangs(self, context):
        bpy.ops.object.select_all(action="DESELECT")
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj

        view_z_local = self.obj.matrix_world.inverted().to_quaternion() @ Vector(
            (0, 0, 1)
        )

        overhangs_faces_index_list = [
            f.index
            for f in self.obj.data.polygons
            if abs(f.normal.dot(view_z_local)) > radians(self.angle)
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")

        for i in overhangs_faces_index_list:
            self.obj.data.polygons[i].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        self.obj.vertex_groups.active_index = self.overhangs_vg.index
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode="OBJECT")

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

    def modal(self, context, event):
        if event.type in {"ESC"}:
            self.cancel(context)
            return {"CANCELLED"}

        if event.type == "TIMER":
            # change theme color, silly!
            mtx = self.obj.matrix_world.copy()
            if not mtx == self.matrix_world:
                self.matrix_world = mtx
                self.compute_overhangs(context)

        return {"PASS_THROUGH"}

    def execute(self, context):
        bpy.ops.object.select_all(action="DESELECT")
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj

        self.overhangs_vg = self.obj.vertex_groups.get(
            "overhangs_vg"
        ) or self.obj.vertex_groups.new(name="overhangs_vg")

        if not self.obj.material_slots:
            mat_white = bpy.data.materials.get(
                "undercuts_preview_mat_white"
            ) or bpy.data.materials.new("undercuts_preview_mat_white")
            mat_white.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            self.obj.active_material = mat_white

        for i, slot in enumerate(self.obj.material_slots):
            if slot.material.name == "overhangs_preview_mat_color":
                self.obj.active_material_index = i
                bpy.ops.object.material_slot_remove()

        self.overhangs_mat_color = bpy.data.materials.get(
            "overhangs_preview_mat_color"
        ) or bpy.data.materials.new("overhangs_preview_mat_color")
        self.overhangs_mat_color.diffuse_color = self.overhangs_color
        self.obj.data.materials.append(self.overhangs_mat_color)
        self.obj.active_material_index = len(self.obj.material_slots) - 1

        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.ops.mesh.select_all(action="SELECT")
        # bpy.ops.mesh.normals_make_consistent(inside=False)
        # bpy.ops.mesh.select_all(action="DESELECT")
        # bpy.ops.object.mode_set(mode="OBJECT")

        # #############################____Surveying____###############################
        self.compute_overhangs(context)

        # add a modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        self.obj = context.object
        self.matrix_world = self.obj.matrix_world.copy()

        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=300)


class ODENT_OT_PathCutter(bpy.types.Operator):
    """New path cutter"""

    bl_idname = "wm.odent_add_path_cutter"
    bl_label = "Path Cutter"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):

        base_mesh = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        return base_mesh

    def get_select_last(self, context):
        context.view_layer.objects.active = self.base_mesh
        bpy.ops.object.mode_set(mode="EDIT")
        me = self.base_mesh.data
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()
        if bm.select_history:
            return [
                elem.index
                for elem in bm.select_history
                if isinstance(elem, bmesh.types.BMVert)
            ][-1]
        return None

    def modal(self, context, event):
        if not event.type in ["RET", "ESC", "LEFTMOUSE"]:
            return {"PASS_THROUGH"}

        elif event.type in ["LEFTMOUSE"]:

            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                if self.counter == 0:
                    self.start_vert = self.previous_vert = self.get_select_last(context)
                    if self.previous_vert:
                        self.counter += 1
                elif self.counter != 0:
                    if event.shift:
                        self.previous_vert = self.get_select_last(context)
                        self.counter += 1

                        return {"PASS_THROUGH"}
                    else:
                        self.last_vert = self.get_select_last(context)
                        # print("previous_vert", self.previous_vert)
                        # print("last_vert", self.last_vert)
                        if self.last_vert:

                            path = 0

                            bpy.ops.mesh.select_all(action="DESELECT")
                            me = self.base_mesh.data
                            self.bm = bmesh.from_edit_mesh(me)
                            self.bm.verts.ensure_lookup_table()
                            for id in [self.previous_vert, self.last_vert]:
                                self.bm.verts[id].select = True

                            try:
                                bpy.ops.mesh.vert_connect_path()
                                message = [
                                    f"Geodesic path selected {self.counter}",
                                    "Cut : Press ENTER | Cancell : Press ESC",
                                ]
                                print(message)
                                ODENT_GpuDrawText(message)
                                path = 1
                            except Exception as e:
                                print(e)
                                pass
                            if path == 0:
                                try:
                                    bpy.ops.mesh.select_all(action="DESELECT")
                                    for id in [self.previous_vert, self.last_vert]:
                                        self.bm.verts[id].select = True
                                    bpy.ops.mesh.shortest_path_select()
                                    message = [
                                        "Shortest path selected",
                                        "Cut : Press ENTER | Cancell : Press ESC",
                                    ]
                                    print(message)
                                    ODENT_GpuDrawText(message)
                                    path = 1
                                except Exception as e:
                                    print(e)
                                    pass

                            if path == 0:

                                bpy.ops.mesh.select_all(action="DESELECT")
                                bpy.ops.object.vertex_group_select()
                                bpy.ops.object.mode_set(mode="OBJECT")
                                self.base_mesh.data.vertices[
                                    self.previous_vert
                                ].select = True
                                bpy.ops.object.mode_set(mode="EDIT")
                                message = [
                                    "Invalid selection ! Continue...",
                                    "Cut : Press ENTER | Cancell : Press ESC",
                                ]
                                ODENT_GpuDrawText(message)

                                return {"RUNNING_MODAL"}

                            if path == 1:
                                bpy.ops.object.vertex_group_select()
                                bpy.ops.object.vertex_group_assign()
                                self.previous_vert = self.last_vert
                                self.counter += 1
                                return {"RUNNING_MODAL"}

                return {"RUNNING_MODAL"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                context.view_layer.objects.active = self.base_mesh
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

                message = ["CANCELLED"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

        elif event.type == "RET":
            if self.counter >= 2:
                if event.value == ("PRESS"):
                    path = 0
                    context.view_layer.objects.active = self.base_mesh
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="DESELECT")
                    me = self.base_mesh.data
                    self.bm = bmesh.from_edit_mesh(me)
                    self.bm.verts.ensure_lookup_table()
                    for id in [self.previous_vert, self.last_vert]:
                        self.bm.verts[id].select = True
                    try:
                        bpy.ops.mesh.vert_connect_path()
                        message = [f"Connected path selected {self.counter}"]
                        print(message)
                        ODENT_GpuDrawText(message)
                        path = 1
                    except Exception as e:
                        message = ["Error from Connect Path"]
                        print(message)
                        print(e)
                        ODENT_GpuDrawText(message)
                        pass
                    if path == 0:
                        try:
                            bpy.ops.mesh.select_all(action="DESELECT")
                            for id in [self.previous_vert, self.last_vert]:
                                self.bm.verts[id].select = True
                            bpy.ops.mesh.shortest_path_select()
                            message = ["Shortest path selected"]
                            print(message)
                            ODENT_GpuDrawText(message)
                            path = 1
                        except Exception as e:
                            message = ["Error from Shortest Path"]
                            print(message)
                            print(e)
                            pass
                    if path == 0:

                        bpy.ops.mesh.select_all(action="DESELECT")
                        bpy.ops.object.vertex_group_select()
                        self.bm.verts[self.previous_vert].select = True
                        message = ["Can't connect the path"]
                        ODENT_GpuDrawText(message)

                        return {"RUNNING_MODAL"}

                    if path == 1:
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.object.vertex_group_select()
                        bpy.ops.object.vertex_group_assign()

                        bpy.ops.mesh.select_mode(type="EDGE")
                        bpy.ops.mesh.edge_split(type="EDGE")
                        # bpy.ops.mesh.loop_to_region()
                        bpy.ops.wm.odent_separate_objects(
                            "EXEC_DEFAULT", SeparateMode="Loose Parts"
                        )

                        bpy.ops.object.mode_set(mode="OBJECT")

                        message = ["FINISHED ./"]
                        ODENT_GpuDrawText(message)
                        sleep(1)
                        ODENT_GpuDrawText()
                        return {"FINISHED"}
            else:
                message = ["Please select at least 2 vertices"]
                ODENT_GpuDrawText(message)
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def execute(self, context):
        self.base_mesh = context.object
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]
        area3D, space3D, region_3d = CtxOverride(context)
        with bpy.context.temp_override(
            area=area3D, space_data=space3D, region=region_3d
        ):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="DESELECT")
            for vg in self.base_mesh.vertex_groups:
                self.base_mesh.vertex_groups.remove(vg)

            self.vg = self.base_mesh.vertex_groups.new(name="cut_loop")

            self.start_vert = None
            self.previous_vert = None
            self.last_vert = None
            self.cut_loop = []

            bpy.ops.wm.tool_set_by_id(name="builtin.select")
            context.window_manager.modal_handler_add(self)
        message = [
            "################## Draw path, #################",
            "################## When done press ENTER. #################",
        ]
        ODENT_GpuDrawText(message)
        return {"RUNNING_MODAL"}


#######################################################################################
########################### Measurements : Operators ##############################
#######################################################################################
class ODENT_OT_OcclusalPlane(bpy.types.Operator):
    """Add Occlusal Plane"""

    bl_idname = "wm.odent_occlusalplane"
    bl_label = "OCCLUSAL PLANE"
    bl_options = {"REGISTER", "UNDO"}

    CollName = "Occlusal Points"
    OcclusalPoints = []

    def modal(self, context, event):

        if not event.type in ("R", "A", "L", "DEL", "RET", "ESC"):

            return {"PASS_THROUGH"}
        #########################################
        if event.type == "R":
            # Add Right point :
            if event.value == ("PRESS"):
                color = (1, 0, 0, 1)  # red
                CollName = self.CollName
                name = "Right_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.RightPoint = NewPoint
                bpy.ops.object.select_all(action="DESELECT")
                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Right_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.RightPoint.name)

        #########################################
        if event.type == "A":
            # Add Right point :
            if event.value == ("PRESS"):
                color = (0, 1, 0, 1)  # green
                CollName = self.CollName
                name = "Anterior_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.AnteriorPoint = NewPoint
                bpy.ops.object.select_all(action="DESELECT")

                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Anterior_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.AnteriorPoint.name)
        #########################################
        if event.type == "L":
            # Add Right point :
            if event.value == ("PRESS"):
                color = (0, 0, 1, 1)  # blue
                CollName = self.CollName
                name = "Left_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.LeftPoint = NewPoint
                bpy.ops.object.select_all(action="DESELECT")
                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Left_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.LeftPoint.name)
        #########################################

        elif event.type == ("DEL") and event.value == ("PRESS"):

            if self.OcclusalPoints:
                name = self.OcclusalPoints.pop()
                bpy.data.objects.remove(bpy.data.objects.get(name))

        elif event.type == "RET":
            if event.value == ("PRESS"):

                if not len(self.OcclusalPoints) == 3:
                    message = ["3 points needed", "Please check Info and retry"]
                    icon = "COLORSET_01_VEC"
                    bpy.ops.wm.odent_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )

                else:
                    OcclusalPlane = PointsToOcclusalPlane(
                        self.TargetObject,
                        self.RightPoint,
                        self.AnteriorPoint,
                        self.LeftPoint,
                        color=(0.0, 0.0, 0.2, 0.7),
                        subdiv=50,
                    )

                    #########################################################
                    self.FullSpace3D.overlay.show_outline_selected = True
                    self.FullSpace3D.overlay.show_object_origins = True
                    self.FullSpace3D.overlay.show_annotation = True
                    self.FullSpace3D.overlay.show_text = True
                    self.FullSpace3D.overlay.show_extras = True
                    self.FullSpace3D.overlay.show_floor = True
                    self.FullSpace3D.overlay.show_axis_x = True
                    self.FullSpace3D.overlay.show_axis_y = True
                    ##########################################################
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)
                    with context.temp_override(**self.FullOverride):
                        bpy.ops.object.select_all(action="DESELECT")
                        bpy.ops.wm.tool_set_by_id(name="builtin.select")
                        bpy.context.scene.tool_settings.use_snap = False
                        bpy.context.scene.cursor.location = (0, 0, 0)
                        bpy.ops.screen.region_toggle(region_type="UI")

                        self.FullSpace3D.shading.background_color = (
                            self.background_color
                        )
                        self.FullSpace3D.shading.background_type = self.background_type

                        bpy.ops.screen.screen_full_area()

                    if self.OcclusalPoints:
                        for name in self.OcclusalPoints:
                            P = bpy.data.objects.get(name)
                            if P:
                                bpy.data.objects.remove(P)
                    Coll = bpy.data.collections.get(self.CollName)
                    if Coll:
                        bpy.data.collections.remove(Coll)
                    ##########################################################
                    return {"FINISHED"}

        elif event.type == ("ESC"):

            ##########################################################
            self.FullSpace3D.overlay.show_outline_selected = True
            self.FullSpace3D.overlay.show_object_origins = True
            self.FullSpace3D.overlay.show_annotation = True
            self.FullSpace3D.overlay.show_text = True
            self.FullSpace3D.overlay.show_extras = True
            self.FullSpace3D.overlay.show_floor = True
            self.FullSpace3D.overlay.show_axis_x = True
            self.FullSpace3D.overlay.show_axis_y = True
            ###########################################################
            for Name in self.visibleObjects:
                obj = bpy.data.objects.get(Name)
                if obj:
                    obj.hide_set(False)

            with bpy.context.temp_override(
                area=self.FullArea3D,
                space_data=self.FullSpace3D,
                region=self.FullRegion3D,
            ):
                bpy.ops.object.select_all(action="DESELECT")
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.ops.screen.region_toggle(region_type="UI")
                bpy.ops.screen.screen_full_area()
            bpy.context.scene.tool_settings.use_snap = False
            bpy.context.scene.cursor.location = (0, 0, 0)

            self.FullSpace3D.shading.background_color = self.background_color
            self.FullSpace3D.shading.background_type = self.background_type

            if self.OcclusalPoints:
                for name in self.OcclusalPoints:
                    P = bpy.data.objects.get(name)
                    if P:
                        bpy.data.objects.remove(P)
            Coll = bpy.data.collections.get(self.CollName)
            if Coll:
                bpy.data.collections.remove(Coll)

            # message = [
            #     " The Occlusal Plane Operation was Cancelled!",
            # ]

            # icon = "COLORSET_02_VEC"
            # bpy.ops.wm.odent_message_box(
            #     "INVOKE_DEFAULT", message=str(message), icon=icon
            # )

            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        Condition_1 = bpy.context.selected_objects and bpy.context.active_object

        if not Condition_1:

            message = [
                "Please select Target object",
            ]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Prepare scene  :
                ##########################################################

                bpy.context.space_data.overlay.show_outline_selected = False
                bpy.context.space_data.overlay.show_object_origins = False
                bpy.context.space_data.overlay.show_annotation = False
                bpy.context.space_data.overlay.show_text = False
                bpy.context.space_data.overlay.show_extras = False
                bpy.context.space_data.overlay.show_floor = False
                bpy.context.space_data.overlay.show_axis_x = False
                bpy.context.space_data.overlay.show_axis_y = False
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.scene.tool_settings.transform_pivot_point = (
                    "INDIVIDUAL_ORIGINS"
                )
                bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

                ###########################################################
                self.TargetObject = bpy.context.active_object
                VisObj = bpy.context.visible_objects
                self.visibleObjects = [obj.name for obj in VisObj]

                for obj in VisObj:
                    if obj is not self.TargetObject:
                        obj.hide_set(True)
                self.Background = bpy.context.space_data.shading.type
                bpy.context.space_data.shading.type = "SOLID"
                self.background_type = bpy.context.space_data.shading.background_type
                bpy.context.space_data.shading.background_type = "VIEWPORT"
                self.background_color = tuple(
                    bpy.context.space_data.shading.background_color
                )
                bpy.context.space_data.shading.background_color = (0.0, 0.0, 0.0)
                bpy.ops.screen.region_toggle(region_type="UI")
                bpy.ops.object.select_all(action="DESELECT")
                bpy.ops.screen.screen_full_area()
                (
                    self.FullOverride,
                    self.FullArea3D,
                    self.FullSpace3D,
                    self.FullRegion3D,
                ) = context_override(context)

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class ODENT_OT_OcclusalPlaneInfo(bpy.types.Operator):
    """Add Align Refference points"""

    bl_idname = "wm.odent_occlusalplaneinfo"
    bl_label = "INFO"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        message = [
            "\u2588 Deselect all objects,",
            "\u2588 Select the target Object,",
            "\u2588 Click < OCCLUSAL PLANE > button,",
            f"      Press <Left Click> to Place Cursor,",
            f"      <'R'> to Add Right Point,",
            f"      <'A'> to Add Anterior median Point,",
            f"      <'L'> to Add Left Point,",
            f"      Press <'DEL'> to delete Point,",
            f"      Press <'ESC'> to Cancel Operation,",
            f"      Press <'ENTER'> to Add Occlusal Plane.",
        ]

        icon = "COLORSET_02_VEC"
        bpy.ops.wm.odent_message_box("INVOKE_DEFAULT", message=str(message), icon=icon)

        return {"FINISHED"}


class ODENT_OT_AddReferencePlanes(bpy.types.Operator):
    """Add Reference Planes"""

    bl_idname = "wm.odent_add_reference_planes"
    bl_label = "Add REFERENCE PLANES"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.object
        isvalid = ob and ob.select_get() and ob.type == "MESH"
        return isvalid

    def modal(self, context, event):

        if not (
            event.type
            in [
                "RET",
                "ESC",
            ]
            and event.value == "PRESS"
        ):

            return {"PASS_THROUGH"}
        #########################################
        elif event.type == "RET":
            if event.value == ("PRESS"):

                CurrentPointsNames = [P.name for P in self.CurrentPointsList]
                P_Names = [P for P in self.PointsNames if not P in CurrentPointsNames]
                if P_Names:
                    # if self.MarkupVoxelMode:
                    #     CursorToVoxelPoint(
                    #         Preffix=self.Preffix, CursorMove=True)

                    loc = context.scene.cursor.location
                    P = AddMarkupPoint(P_Names[0], self.Color, loc, 1, self.CollName)
                    self.CurrentPointsList.append(P)

                if not P_Names:

                    area3D, space3D, region_3d = CtxOverride(context)
                    RefPlanes = PointsToRefPlanes(
                        self.TargetObject,
                        self.CurrentPointsList,
                        color=(0.0, 0.0, 0.2, 0.7),
                        CollName=self.CollName,
                    )
                    bpy.ops.object.select_all(action="DESELECT")
                    for Plane in RefPlanes:
                        Plane.select_set(True)
                    CurrentPoints = [
                        bpy.data.objects.get(PName) for PName in CurrentPointsNames
                    ]
                    for P in CurrentPoints:
                        P.select_set(True)
                    self.TargetObject.select_set(True)
                    bpy.context.view_layer.objects.active = self.TargetObject
                    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
                    bpy.ops.object.select_all(action="DESELECT")
                    # self.DcmInfo[self.Preffix]["Frankfort"] = RefPlanes[0].name
                    # self.ODENT_Props.DcmInfo = str(self.DcmInfo)
                    ##########################################################
                    space3D.overlay.show_outline_selected = True
                    space3D.overlay.show_object_origins = True
                    space3D.overlay.show_annotation = True
                    space3D.overlay.show_text = True
                    space3D.overlay.show_extras = True
                    space3D.overlay.show_floor = True
                    space3D.overlay.show_axis_x = True
                    space3D.overlay.show_axis_y = True
                    # ###########################################################
                    area3D, space3D, region_3d = CtxOverride(context)
                    with bpy.context.temp_override(
                        area=area3D, space_data=space3D, region=region_3d
                    ):
                        bpy.ops.wm.tool_set_by_id(name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False

                    bpy.context.scene.cursor.location = (0, 0, 0)
                    # bpy.ops.screen.region_toggle( region_type="UI")
                    # self.ODENT_Props.ActiveOperator = "None"

                    return {"FINISHED"}

        #########################################

        elif event.type == "DEL" and event.value == "PRESS":
            if self.CurrentPointsList:
                P = self.CurrentPointsList.pop()
                bpy.data.objects.remove(P)

        elif event.type == ("ESC"):
            if self.CurrentPointsList:
                for P in self.CurrentPointsList:
                    bpy.data.objects.remove(P)

            _, space3D, _ = CtxOverride(context)
            ##########################################################
            space3D.overlay.show_outline_selected = True
            space3D.overlay.show_object_origins = True
            space3D.overlay.show_annotation = True
            space3D.overlay.show_text = True
            space3D.overlay.show_extras = True
            space3D.overlay.show_floor = True
            space3D.overlay.show_axis_x = True
            space3D.overlay.show_axis_y = True
            ###########################################################
            area3D, space3D, region_3d = CtxOverride(context)
            with bpy.context.temp_override(
                area=area3D, space_data=space3D, region=region_3d
            ):
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
            bpy.context.scene.tool_settings.use_snap = False

            bpy.context.scene.cursor.location = (0, 0, 0)
            # bpy.ops.screen.region_toggle( region_type="UI")
            # self.ODENT_Props.ActiveOperator = "None"
            # message = [
            #     " The Frankfort Plane Operation was Cancelled!",
            # ]

            # icon = "COLORSET_02_VEC"
            # bpy.ops.wm.odent_message_box(
            #     "INVOKE_DEFAULT", message=str(message), icon=icon
            # )

            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if context.space_data.type == "VIEW_3D":

            self.ODENT_Props = context.scene.ODENT_Props
            sd = context.space_data
            scn = context.scene
            # Prepare scene  :
            ##########################################################
            sd.overlay.show_outline_selected = False
            sd.overlay.show_object_origins = False
            sd.overlay.show_annotation = False
            sd.overlay.show_text = True
            sd.overlay.show_extras = False
            sd.overlay.show_floor = False
            sd.overlay.show_axis_x = False
            sd.overlay.show_axis_y = False
            scn.tool_settings.use_snap = True
            scn.tool_settings.snap_elements = {"FACE"}
            scn.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

            ###########################################################
            self.CollName = "REFERENCE PLANES"
            self.CurrentPointsList = []
            self.PointsNames = ["Na", "R_Or", "R_Po", "L_Or", "L_Po"]
            self.Color = [1, 0, 0, 1]  # Red color
            self.TargetObject = context.object
            self.visibleObjects = context.visible_objects.copy()
            # self.MarkupVoxelMode = (self.TargetObject.get(OdentConstants.ODENT_TYPE_TAG)=="CT_Voxel")
            # self.Preffix = self.TargetObject.name.split("_")[0]
            # DcmInfo = self.ODENT_Props.DcmInfo
            # self.DcmInfo = eval(DcmInfo)
            # area3D, space3D , region_3d = CtxOverride(context)
            # bpy.ops.screen.region_toggle( region_type="UI")
            # bpy.ops.object.select_all(action="DESELECT")
            # bpy.ops.object.select_all( action="DESELECT")

            context.window_manager.modal_handler_add(self)
            self.ODENT_Props.ActiveOperator = "odent.add_reference_planes"
            return {"RUNNING_MODAL"}

        else:
            message = [
                "Active space must be a View3d",
            ]

            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}


class ODENT_OT_AddMarkupPoint(bpy.types.Operator):
    """Add Markup point"""

    bl_idname = "wm.odent_add_markup_point"
    bl_label = "ADD MARKUP POINT"

    MarkupName: StringProperty(
        name="Markup Name",
        default="Markup 01",
        description="Markup Name",
    )  # type: ignore
    MarkupColor: FloatVectorProperty(
        name="Markup Color",
        description="Markup Color",
        default=[1.0, 0.0, 0.0, 1.0],
        size=4,
        subtype="COLOR",
    )  # type: ignore
    Markup_Diameter: FloatProperty(
        description="Diameter", default=1, step=1, precision=2
    )  # type: ignore

    CollName = "Markup Points"

    def execute(self, context):

        bpy.ops.object.mode_set(mode="OBJECT")

        if self.MarkupVoxelMode:
            Preffix = self.TargetObject.name.split("_")[0]
            CursorToVoxelPoint(Preffix=Preffix, CursorMove=True)

        Co = context.scene.cursor.location
        P = AddMarkupPoint(
            name=self.MarkupName,
            color=self.MarkupColor,
            loc=Co,
            Diameter=self.Markup_Diameter,
            CollName=self.CollName,
        )
        bpy.ops.object.select_all(action="DESELECT")
        self.TargetObject.select_set(True)
        bpy.context.view_layer.objects.active = self.TargetObject
        bpy.ops.object.mode_set(mode=self.mode)

        return {"FINISHED"}

    def invoke(self, context, event):

        self.ODENT_Props = bpy.context.scene.ODENT_Props

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select Target Object ! "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            if Active_Obj.select_get() == False:
                message = [" Please select Target Object ! "]
                icon = "COLORSET_02_VEC"
                bpy.ops.wm.odent_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:
                self.mode = Active_Obj.mode
                self.TargetObject = Active_Obj
                self.MarkupVoxelMode = CheckString(
                    self.TargetObject.name, ["BD", "_CTVolume"]
                )
                wm = context.window_manager
                return wm.invoke_props_dialog(self)


class ODENT_OT_CtVolumeOrientation(bpy.types.Operator):
    """CtVolume Orientation according to Frankfort Plane"""

    bl_idname = "wm.odent_ctvolume_orientation"
    bl_label = "CTVolume Orientation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ODENT_Props = bpy.context.scene.ODENT_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:

            Condition = CheckString(Active_Obj.name, ["BD"]) and CheckString(
                Active_Obj.name, ["_CTVolume", "Segmentation"], any
            )

            if not Condition:
                message = [
                    "CTVOLUME Orientation : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]

                icon = "COLORSET_02_VEC"
                bpy.ops.wm.odent_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name.split("_")[0]
                DcmInfo = eval(ODENT_Props.DcmInfo)
                if not "Frankfort" in DcmInfo[Preffix].keys():
                    message = [
                        "CTVOLUME Orientation : ",
                        "Please Add Reference Planes before CTVOLUME Orientation ! ",
                    ]
                    icon = "COLORSET_02_VEC"
                    bpy.ops.wm.odent_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )
                    return {"CANCELLED"}
                else:
                    Frankfort_Plane = bpy.data.objects.get(
                        DcmInfo[Preffix]["Frankfort"]
                    )
                    if not Frankfort_Plane:
                        message = [
                            "CTVOLUME Orientation : ",
                            "Frankfort Reference Plane has been removed",
                            "Please Add Reference Planes before CTVOLUME Orientation ! ",
                        ]
                        icon = "COLORSET_01_VEC"
                        bpy.ops.wm.odent_message_box(
                            "INVOKE_DEFAULT", message=str(message), icon=icon
                        )
                        return {"CANCELLED"}
                    else:
                        Vol = [
                            obj
                            for obj in bpy.data.objects
                            if Preffix in obj.name and "_CTVolume" in obj.name
                        ][0]
                        Vol.matrix_world = (
                            Frankfort_Plane.matrix_world.inverted() @ Vol.matrix_world
                        )
                        area3D, space3D, region_3d = CtxOverride(context)
                        with bpy.context.temp_override(
                            area=area3D, space_data=space3D, region=region_3d
                        ):
                            bpy.ops.view3d.view_center_cursor()
                            bpy.ops.view3d.view_all(center=True)
                        return {"FINISHED"}


class ODENT_OT_ResetCtVolumePosition(bpy.types.Operator):
    """Reset the CtVolume to its original Patient Position"""

    bl_idname = "wm.odent_reset_ctvolume_position"
    bl_label = "RESET CTVolume POSITION"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ODENT_Props = bpy.context.scene.ODENT_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.odent_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}
        else:
            Condition = CheckString(Active_Obj.name, ["BD"]) and CheckString(
                Active_Obj.name, ["_CTVolume", "Segmentation"], any
            )

            if not Condition:
                message = [
                    "Reset Position : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]
                icon = "COLORSET_01_VEC"
                bpy.ops.wm.odent_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name.split("_")[0]
                Vol = [
                    obj
                    for obj in bpy.data.objects
                    if CheckString(obj.name, [Preffix, "_CTVolume"])
                ][0]
                DcmInfoDict = eval(ODENT_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]
                TransformMatrix = DcmInfo["TransformMatrix"]
                Vol.matrix_world = TransformMatrix

                return {"FINISHED"}


class ODENT_OT_CleanMeshIterative(bpy.types.Operator):
    """clean mesh iterative"""

    bl_idname = "wm.odent_clean_mesh_iterative"
    bl_label = "Clean Mesh"
    bl_options = {"REGISTER", "UNDO"}
    
    keep_parts: EnumProperty(
        name="Keep Parts",
        description="Keep Parts",
        items=set_enum_items(["All", "Only Big"]),
        default="All",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return context.object

    def execute(self, context):
        info_dict = {}
        obj = context.object
        verts_count_start, edges_count_start, polygons_count_start = mesh_count(obj)
        info_dict["verts_count_start"] = verts_count_start
        info_dict["edges_count_start"] = edges_count_start
        info_dict["polygons_count_start"] = polygons_count_start

        non_manifold_count = count_non_manifold_verts(obj)
        if not non_manifold_count:
            message = ["Mesh is manifold"]
            ODENT_GpuDrawText(message_list=message, sleep_time=2)
            return {"CANCELLED"}

        merge_verts(obj, threshold=0.001, all=True)
        delete_loose(obj)
        delete_interior_faces(obj)
        fill_holes(obj, _all=True, hole_size=400)

        return {"FINISHED"}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class ODENT_OT_ConnectPathCutter(bpy.types.Operator):
    """mesh curve cutter tool"""

    bl_idname = "wm.odent_connect_path_cutter"
    bl_label = "Path cutter"
    bl_options = {"REGISTER", "UNDO"}

    Resolution: IntProperty(default=2)  # type: ignore
    closeCurve = True
    radius = 3
    curve_res = 2

    @classmethod
    def poll(cls, context):

        base_mesh = (
            context.object
            and context.object.select_get()
            and context.object.type == "MESH"
        )
        return base_mesh

    def resample_loop_evenly_np(self, loop_obj, spacing_mm=1):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = loop_obj.evaluated_get(depsgraph)

        # Convert evaluated curve to mesh
        mesh = eval_obj.to_mesh()

        # Get vertices as Nx3 NumPy array in world space
        verts_world = np.array([eval_obj.matrix_world @ v.co for v in mesh.vertices])
        edges = [(e.vertices[0], e.vertices[1]) for e in mesh.edges]

        # Reorder points to a continuous polyline (greedy walk)
        edge_map = {}
        for a, b in edges:
            edge_map.setdefault(a, []).append(b)
            edge_map.setdefault(b, []).append(a)

        ordered = [0]
        visited = {0}
        while True:
            last = ordered[-1]
            neighbors = edge_map.get(last, [])
            nexts = [n for n in neighbors if n not in visited]
            if not nexts:
                break
            nxt = nexts[0]
            ordered.append(nxt)
            visited.add(nxt)

        points = verts_world[ordered]  # (N, 3)

        # Compute segment lengths and total distance
        deltas = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        distances = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = distances[-1]

        # Determine resampling points
        num_samples = int(np.ceil(total_length / spacing_mm)) + 1
        target_distances = np.linspace(0, total_length, num_samples)

        # Interpolate resampled points
        resampled = []
        i = 0
        for td in target_distances:
            while i < len(distances) - 2 and distances[i + 1] < td:
                i += 1
            d1, d2 = distances[i], distances[i + 1]
            p1, p2 = points[i], points[i + 1]
            t = (td - d1) / (d2 - d1) if d2 > d1 else 0
            interp_point = p1 + t * (p2 - p1)
            resampled.append(interp_point)

        eval_obj.to_mesh_clear()
        return np.array(resampled)  # shape (M, 3)

    def perform_cut(self, context):
        timer = TimerLogger("Path cutter perform cut")
        override, area3D, space3D, region3D = context_override(context)
        with bpy.context.temp_override(**override):
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
            bpy.ops.view3d.snap_cursor_to_center()
            space3D.overlay.show_outline_selected = True
            context.scene.tool_settings.use_snap = False

        if not context.object:
            context.view_layer.objects.active = self.cutter

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.cutter.select_set(True)
        context.view_layer.objects.active = self.cutter

        hide_object(False, self.cutter)

        if self.closeCurve:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.cyclic_toggle()
            bpy.ops.object.mode_set(mode="OBJECT")

        message = ["processing..."]
        ODENT_GpuDrawText(message)

        self.cutter.data.bevel_depth = 0
        self.cutter.data.resolution_u = self.Resolution
        self.shrinkwrap.offset = 0.2

        bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
        bpy.ops.object.material_slot_remove_all()
        bpy.ops.object.convert(target="MESH")

        cutter = context.object
        target = self.base_mesh

        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target

        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="OBJECT")

        depsgraph = context.evaluated_depsgraph_get()
        eval_target = target.evaluated_get(depsgraph)
        eval_mesh = eval_target.to_mesh()

        # Build KDTree from face centers
        # bm_eval = bmesh.new()
        # bm_eval.from_mesh(eval_mesh)
        # bm_eval.faces.ensure_lookup_table()
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        timer.log("Preparation step complete")

        face_tree = kdtree.KDTree(len(bm.faces))
        for i, face in enumerate(bm.faces):
            center = face.calc_center_median()
            face_tree.insert(center, i)
        face_tree.balance()

        timer.log("KDTree step complete")
        # cutter_world_verts = self.resample_loop_evenly_np(cutter)

        cutter_world_verts = [cutter.matrix_world @ v.co for v in cutter.data.vertices]
        # bm = bmesh.new()
        # bm.from_mesh(eval_mesh)

        projected_verts = []
        projected_world_coords = []
        failed_find_range = 0
        failed_projection = 0

        # bm.verts.ensure_lookup_table()
        # bm.edges.ensure_lookup_table()
        # bm.faces.ensure_lookup_table()
        n = len(cutter_world_verts)
        timer.log("Finding hits start")
        for i in range(n):
            pt = Vector(cutter_world_verts[i])
            # bm.verts.ensure_lookup_table()
            # bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            found = face_tree.find_range(pt, self.radius)
            if not found:
                failed_find_range += 1
                continue

            # Compute average normal
            normals = [bm.faces[i].normal for (_, i, _) in found]
            avg_normal = sum(normals, Vector()) / len(normals)
            avg_normal.normalize()

            ray_origin = pt + avg_normal
            ray_origin_local = eval_target.matrix_world.inverted() @ ray_origin

            ray_dir = avg_normal
            ray_dir_inverse = -avg_normal

            result, hit_loc, hit_normal, face_idx = eval_target.ray_cast(
                ray_origin_local, ray_dir
            )
            result_inverse, hit_loc_inverse, hit_normal_inverse, face_idx_inverse = (
                eval_target.ray_cast(ray_origin_local, ray_dir_inverse)
            )
            if not result and not result_inverse:
                failed_projection += 1
                continue
            elif result and not result_inverse:
                _loc = hit_loc
                fid = face_idx
            elif not result and result_inverse:
                _loc = hit_loc_inverse
                fid = face_idx_inverse
            else:
                dist1 = (
                    (hit_loc - ray_origin_local).length
                    if face_idx != -1
                    else float("inf")
                )
                dist2 = (
                    (hit_loc_inverse - ray_origin_local).length
                    if face_idx_inverse != -1
                    else float("inf")
                )
                if dist1 < dist2:
                    _loc = hit_loc
                    fid = face_idx
                elif dist2 < float("inf"):
                    _loc = hit_loc_inverse
                    fid = face_idx_inverse
                else:
                    failed_projection += 1
                    continue

            face = bm.faces[fid]
            verts = list(face.verts)

            # Delete original face
            bmesh.ops.delete(bm, geom=[face], context="FACES")
            # Insert projected vertex
            new_vert = bm.verts.new(_loc)
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                try:
                    bm.faces.new([v1, v2, new_vert])
                except ValueError:
                    pass  # skip if face already exists
            bm.verts.index_update()
            projected_verts.append(new_vert)
            projected_world_coords.append(eval_target.matrix_world @ new_vert.co.copy())
            txt = f"Find hit {i} done"
            timer.log(txt)
            # ODENT_GpuDrawText(message)

        # timer.log("projection of vertices step complete")
        odent_log(
            [
                f"num points = {len(cutter_world_verts)}",
                f"failed find range = {failed_find_range}",
                f"failed projections = {failed_projection}",
            ]
        )

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        pairs = list(zip(projected_verts, projected_verts[1:] + [projected_verts[0]]))
        if not self.closeCurve:
            pairs = pairs[:-1]
        good_pairs = []
        for p in pairs:
            if p[0] != p[1]:
                good_pairs.append(p)

        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False

        n = len(pairs)
        for i, [v1, v2] in enumerate(pairs):
            bm.verts.ensure_lookup_table()
            percentage = i * 100 / n
            if percentage == 100:
                percentage == 97

            result = bmesh.ops.connect_vert_pair(bm, verts=[v1, v2])
            new_edges = result.get("edges", [])
            if not new_edges:
                odent_log([f"failed new edges = {i}"])
                continue
            for e in result.get("edges"):
                e.select = True

            if percentage:
                message = [f"processing...{int(percentage)}%"]
                ODENT_GpuDrawText(message, percentage=percentage)

        # cut_verts_ids = [v.index for v in bm.verts if v.select]
        timer.log("connect paths step complete")

        bm.to_mesh(target.data)
        bm.free()
        # bm_eval.free()

        eval_target.to_mesh_clear()
        bpy.context.view_layer.update()
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.edge_split(type="VERT")
        # bpy.ops.mesh.select_mode(type="EDGE")
        # bpy.ops.mesh.edge_split(type='EDGE')
        # bpy.ops.object.mode_set( mode="OBJECT")
        bpy.ops.wm.odent_separate_objects("EXEC_DEFAULT", SeparateMode="Loose Parts")

        bpy.data.objects.remove(cutter)
        coll = bpy.data.collections.get(OdentConstants.CUTTERS_COLL_NAME)
        if coll:
            bpy.data.collections.remove(coll)

        ODENT_GpuDrawText()

        timer.log("Separate step complete")

        return

    def perform_cut_bvhtree(self, context):
        timer = TimerLogger("Path cutter perform cut")
        override, area3D, space3D, region3D = context_override(context)
        with bpy.context.temp_override(**override):
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
            bpy.ops.view3d.snap_cursor_to_center()
            space3D.overlay.show_outline_selected = True
            context.scene.tool_settings.use_snap = False

        if not context.object:
            context.view_layer.objects.active = self.cutter

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.cutter.select_set(True)
        context.view_layer.objects.active = self.cutter

        hide_object(False, self.cutter)

        if self.closeCurve:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.curve.cyclic_toggle()
            bpy.ops.object.mode_set(mode="OBJECT")

        message = ["processing..."]
        ODENT_GpuDrawText(message)

        self.cutter.data.bevel_depth = 0
        self.cutter.data.resolution_u = self.Resolution
        self.shrinkwrap.offset = 0.2

        bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
        bpy.ops.object.material_slot_remove_all()
        bpy.ops.object.convert(target="MESH")

        cutter = context.object
        target = self.base_mesh

        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target

        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="OBJECT")

        depsgraph = context.evaluated_depsgraph_get()
        eval_target = target.evaluated_get(depsgraph)
        eval_mesh = eval_target.to_mesh()

        # Build KDTree from face centers
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        timer.log("Preparation step complete")
        _tree = bvhtree.BVHTree.FromBMesh(bm)

        # face_tree = kdtree.KDTree(len(bm_eval.faces))
        # for i, face in enumerate(bm_eval.faces):
        #     center = face.calc_center_median()
        #     face_tree.insert(center, i)
        # face_tree.balance()

        timer.log("KDTree step complete")
        # cutter_world_verts = self.resample_loop_evenly_np(cutter)

        cutter_world_verts = [cutter.matrix_world @ v.co for v in cutter.data.vertices]

        projected_verts = []
        projected_world_coords = []
        failed_find_range = 0
        failed_projection = 0

        n = len(cutter_world_verts)
        timer.log("Finding hits start")
        for i in range(n):
            bm.faces.ensure_lookup_table()
            pt_local = target.matrix_world.inverted() @ Vector(cutter_world_verts[i])
            # bm.verts.ensure_lookup_table()
            # bm.edges.ensure_lookup_table()

            # _tree = bvhtree.BVHTree.FromBMesh(bm)
            hits = _tree.find_nearest_range(pt_local, self.radius)
            if not hits:
                failed_find_range += 1
                continue

            # Compute average normal
            normals = [normal for (position, normal, index, distance) in hits]
            avg_normal = sum(normals, Vector()) / len(normals)
            avg_normal.normalize()

            ray_origin_local = pt_local + avg_normal

            ray_dir = avg_normal
            ray_dir_inverse = -avg_normal

            result, hit_loc, hit_normal, face_idx = eval_target.ray_cast(
                ray_origin_local, ray_dir
            )
            result_inverse, hit_loc_inverse, hit_normal_inverse, face_idx_inverse = (
                eval_target.ray_cast(ray_origin_local, ray_dir_inverse)
            )
            if not result and not result_inverse:
                failed_projection += 1
                continue
            elif result and not result_inverse:
                _loc = hit_loc
                fid = face_idx
            elif not result and result_inverse:
                _loc = hit_loc_inverse
                fid = face_idx_inverse
            else:
                dist1 = (
                    (hit_loc - ray_origin_local).length
                    if face_idx != -1
                    else float("inf")
                )
                dist2 = (
                    (hit_loc_inverse - ray_origin_local).length
                    if face_idx_inverse != -1
                    else float("inf")
                )
                if dist1 < dist2:
                    _loc = hit_loc
                    fid = face_idx
                elif dist2 < float("inf"):
                    _loc = hit_loc_inverse
                    fid = face_idx_inverse
                else:
                    failed_projection += 1
                    continue

            face = bm.faces[fid]
            verts = list(face.verts)

            # Delete original face
            bmesh.ops.delete(bm, geom=[face], context="FACES")
            # Insert projected vertex
            new_vert = bm.verts.new(_loc)
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                try:
                    bm.faces.new([v1, v2, new_vert])
                except ValueError:
                    pass  # skip if face already exists
            bm.verts.index_update()
            projected_verts.append(new_vert)
            projected_world_coords.append(eval_target.matrix_world @ new_vert.co.copy())
            txt = f"Find hit {i} done"
            timer.log(txt)
            # ODENT_GpuDrawText(message)

        # timer.log("projection of vertices step complete")
        odent_log(
            [
                f"num points = {len(cutter_world_verts)}",
                f"failed find range = {failed_find_range}",
                f"failed projections = {failed_projection}",
            ]
        )

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        pairs = list(zip(projected_verts, projected_verts[1:] + [projected_verts[0]]))
        if not self.closeCurve:
            pairs = pairs[:-1]
        good_pairs = []
        for p in pairs:
            if p[0] != p[1]:
                good_pairs.append(p)

        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False

        n = len(pairs)
        for i, [v1, v2] in enumerate(pairs):
            bm.verts.ensure_lookup_table()
            percentage = i * 100 / n
            if percentage == 100:
                percentage == 97

            result = bmesh.ops.connect_vert_pair(bm, verts=[v1, v2])
            new_edges = result.get("edges", [])
            if not new_edges:
                odent_log([f"failed new edges = {i}"])
                continue
            for e in result.get("edges"):
                e.select = True

            if percentage:
                message = [f"processing...{int(percentage)}%"]
                ODENT_GpuDrawText(message, percentage=percentage)

        # cut_verts_ids = [v.index for v in bm.verts if v.select]
        timer.log("connect paths step complete")

        bm.to_mesh(target.data)
        bm.free()

        eval_target.to_mesh_clear()
        bpy.context.view_layer.update()
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.edge_split(type="VERT")
        # bpy.ops.mesh.select_mode(type="EDGE")
        # bpy.ops.mesh.edge_split(type='EDGE')
        # bpy.ops.object.mode_set( mode="OBJECT")
        bpy.ops.wm.odent_separate_objects("EXEC_DEFAULT", SeparateMode="Loose Parts")

        bpy.data.objects.remove(cutter)
        coll = bpy.data.collections.get(OdentConstants.CUTTERS_COLL_NAME)
        if coll:
            bpy.data.collections.remove(coll)

        ODENT_GpuDrawText()

        timer.log("Separate step complete")

        return

    def add_curve_cutter(self, context):

        override, area3D, space3D, region3D = context_override(context)
        context.scene.tool_settings.use_snap = True
        context.scene.tool_settings.snap_elements = {"FACE"}

        bpy.ops.curve.primitive_bezier_curve_add(
            radius=1, enter_editmode=False, align="CURSOR"
        )
        self.cutter = context.object
        MoveToCollection(self.cutter, OdentConstants.CUTTERS_COLL_NAME)
        hide_collection(False, OdentConstants.CUTTERS_COLL_NAME)

        bpy.ops.object.select_all(action="DESELECT")
        self.cutter.select_set(True)
        context.view_layer.objects.active = self.cutter

        cutter_idx = get_incremental_idx(
            data=bpy.data.objects, odent_type=OdentConstants.CONNECT_PATH_CUTTER_TYPE
        )
        cutter_name = get_incremental_name(
            OdentConstants.CONNECT_PATH_CUTTER_NAME, cutter_idx
        )
        self.cutter.name = cutter_name
        # self.cutter[OdentConstants.ODENT_TYPE_TAG] = OdentConstants.CONNECT_PATH_CUTTER_TYPE
        # self.cutter["odent_target"] = self.base_mesh.name
        # self.cutter["odent_close_curve"] = self.closeCurve

        # CurveCutter settings :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        bpy.ops.curve.dissolve_verts()
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

        self.cutter.data.bevel_depth = 0.1

        mat = bpy.data.materials.get(
            OdentConstants.CONNECT_PATH_CUTTER_MAT.get("name")
        ) or bpy.data.materials.new(OdentConstants.CONNECT_PATH_CUTTER_MAT.get("name"))
        mat.diffuse_color = OdentConstants.CONNECT_PATH_CUTTER_MAT.get("diffuse_color")
        mat.roughness = OdentConstants.CONNECT_PATH_CUTTER_MAT.get("roughness")

        self.cutter.active_material = mat

        with bpy.context.temp_override(**override):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

        space3D.overlay.show_outline_selected = False
        self.shrinkwrap = self.cutter.modifiers.new(
            name="Shrinkwrap", type="SHRINKWRAP"
        )
        self.shrinkwrap.target = self.base_mesh
        self.shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        self.shrinkwrap.use_apply_on_spline = True

    def add_cutter_point(self, context):

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.extrude(mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.curve.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        points = self.cutter.data.splines[0].bezier_points[-1].select_control_point = (
            True
        )

    def del_cutter_point(self, context):
        if len(self.cutter.data.splines[0].bezier_points) > 1:
            if not context.object:
                context.view_layer.objects.active = self.cutter
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            self.cutter.select_set(True)
            context.view_layer.objects.active = self.cutter
            try:
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.curve.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")
                self.cutter.data.splines[0].bezier_points[
                    -1
                ].select_control_point = True

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.curve.delete(type="VERT")  # Deletes selected points

                bpy.ops.curve.select_all(action="SELECT")
                bpy.ops.curve.handle_type_set(type="AUTOMATIC")
                bpy.ops.curve.select_all(action="DESELECT")

                bpy.ops.object.mode_set(mode="OBJECT")
                self.cutter.data.splines[0].bezier_points[
                    -1
                ].select_control_point = True
            except Exception as e:
                odent_log(["error delete curve point :", e])
                pass

    def modal(self, context, event):

        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}
        elif event.type == "RET" and self.counter == 0:
            return {"PASS_THROUGH"}
        elif event.type == "DEL" and self.counter == 0:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                area3D, space3D, region_3d = CtxOverride(context)
                with bpy.context.temp_override(
                    area=area3D, space_data=space3D, region=region_3d
                ):
                    bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                ODENT_GpuDrawText(message)
                sleep(2)
                ODENT_GpuDrawText()
                return {"CANCELLED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                # print(_is_valid)
                if _is_valid:
                    self.add_cutter_point(context)
                    return {"RUNNING_MODAL"}
                else:
                    return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:

            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                # print(_is_valid)
                if _is_valid:
                    self.add_curve_cutter(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point(context)
                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:

            if event.value == ("PRESS"):
                self.perform_cut_bvhtree(context)
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object

            # return self.execute(context)
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.odent.message_box("INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        self.scn = context.scene
        self.counter = 0
        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        override, area3D, space3D, region3D = context_override(context)
        with bpy.context.temp_override(**override):
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

        context.window_manager.modal_handler_add(self)
        txt = [
            "Mouse left : draw curve",
            "DEL : Roll back",
            "ESC : Cancell operation",
            "ENTER : Finalise",
        ]
        ODENT_GpuDrawText(txt)
        return {"RUNNING_MODAL"}


class ODENT_OT_add_asset_modal(bpy.types.Operator):
    """Modal operator to link selected assets interactively"""

    bl_idname = "wm.odent_add_asset_modal"
    bl_label = "Add Selected Assets (Modal)"
    bl_options = {"REGISTER", "UNDO"}

    # Props
    collection_name: bpy.props.StringProperty()  # type: ignore
    new_collection_name: bpy.props.StringProperty()  # type: ignore
    use_new_collection: bpy.props.BoolProperty(default=False)  # type: ignore
    component_mode: bpy.props.EnumProperty(
        name="Component Mode",
        items=[("ADD", "Add", ""), ("CUT", "Cut", "")],
        default="ADD",
    )  # type: ignore
    align_to_active: bpy.props.BoolProperty(default=False)  # type: ignore

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.asset_browser_area, self.asset_browser_space = open_asset_browser()
            self.can_update = False

            bpy.app.timers.register(self.defer)
            # Step 1: Choose collection
            return context.window_manager.invoke_props_dialog(self, width=600)
        else:
            self.report({"WARNING"}, "Active space must be a View3d")
            return {"CANCELLED"}

    def draw(self, context):
        layout = self.layout
        layout.label(text="Select Target Collection:")

        if not self.use_new_collection:
            layout.prop_search(
                self, "collection_name", bpy.data, "collections", text=""
            )
        layout.prop(self, "use_new_collection")
        if self.use_new_collection:
            layout.prop(self, "new_collection_name")

        target = (
            self.new_collection_name
            if self.use_new_collection
            else self.collection_name
        )
        if target == OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME:
            layout.separator()
            layout.label(text="GUIDE COMPONENTS Mode:")
            layout.prop(self, "component_mode", expand=True)

        layout.separator()
        layout.prop(self, "align_to_active")

    def get_selected_assets(self):
        result = {
            "success": 0,
            "message": "",
            "directory": None,
            "filename": None,
        }
        asset_area = self.asset_browser_area
        space = asset_area.spaces.active
        current_library_name = space.params.asset_library_reference
        if not current_library_name == OdentConstants.ODENT_LIB_NAME:
            result["message"] = [
                f"Warning : The selected asset is not part of {OdentConstants.ODENT_LIB_NAME}",
                "<ENTER> : retry  <ESC> : cancel.",
            ]
            return result

        active_asset_file = space.params.filename
        if not active_asset_file:
            result["message"] = [
                f"Warning : Please select asset from {OdentConstants.ODENT_LIB_NAME}",
                "<ENTER> : retry  <ESC> : cancel.",
            ]
            return result
        library_path_root = bpy.context.preferences.filepaths.asset_libraries.get(
            OdentConstants.ODENT_LIB_NAME
        ).path
        head, filename = split(active_asset_file)
        directory = join(library_path_root, head)
        result = {
            "success": 1,
            "message": "",
            "directory": directory,
            "filename": filename,
        }
        return result

    def execute(self, context):
        self.target_collection_name = (
            self.new_collection_name
            if self.use_new_collection
            else self.collection_name
        )
        if not self.target_collection_name:
            self.report({"WARNING"}, "No collection selected or created")
            return {"CANCELLED"}

        message = [f"Using collection '{self.target_collection_name}'"]
        ODENT_GpuDrawText(message_list=message)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def defer(self):
        params = self.asset_browser_space.params
        if not params:
            return 0

        try:
            params.asset_library_reference = OdentConstants.ODENT_LIB_NAME

        except TypeError:
            # If the reference doesn't exist.
            params.asset_library_reference = "LOCAL"

        # params.import_type = 'APPEND'
        self.can_update = True

    def modal(self, context, event):
        if self.can_update:
            message = ["Select asset(s)", "<ENTER> : to confirm  <ESC> : to cancel"]
            ODENT_GpuDrawText(message_list=message)
            self.can_update = False

        if event.type in {"ESC"}:
            close_asset_browser(context)
            message = ["Cancelled!"]
            ODENT_GpuDrawText(message_list=message, sleep_time=1)
            return {"CANCELLED"}

        elif event.type == "RET" and event.value == "PRESS":
            print(bpy.context.asset)
            result = self.get_selected_assets()
            success, message, directory, filename = result.values()
            if not success:
                ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.red)
                return {"RUNNING_MODAL"}

            # --- Align logic ---
            align_to_active = self.align_to_active
            target = context.object
            if not align_to_active or not target or not target.select_get():
                align_to_active = False

            previous_existing_objects = bpy.context.scene.objects[:]
            # --- Simulate linking ---
            bpy.ops.wm.append(
                directory=directory,
                filename=filename,
                clear_asset_data=True,
                autoselect=True,
            )

            new_asset_list = [
                o
                for o in context.selected_objects
                if not o in previous_existing_objects
            ]
            # --- Align logic ---
            for new_asset in new_asset_list:
                if align_to_active:

                    new_asset.location = target.location
                    new_asset.rotation_euler = target.rotation_euler

                # --- Move linked objects to collection ---

                MoveToCollection(new_asset, self.target_collection_name)
                asset_name = new_asset.name
                # --- GUIDE_COMPONENTS logic ---
                if (
                    self.target_collection_name
                    == OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME
                    and self.component_mode == "ADD"
                ):
                    asset_name = "_ADD_" + asset_name

                # --- Unique name logic ---
                asset_name = get_unique_name(asset_name, bpy.data.objects)
                new_asset.name = asset_name

            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


################################################################
# AI
################################################################


class ODENT_OT_AiSegmentator(bpy.types.Operator):
    """ai segmentator using nnunetv2 and pytorch"""

    bl_idname = "wm.ai_segmentator"
    bl_label = " AI Segmentation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if not context.object.select_get():
            return False

        return context.object.get(OdentConstants.ODENT_TYPE_TAG) in [
            OdentConstants.VOXEL_OBJECT_TYPE,
            OdentConstants.PCD_OBJECT_TYPE,
        ]

    def execute(self, context):
        global IMAGE3D
        message = ["AI segmentation : processing ..."]
        ODENT_GpuDrawText(message_list=message)
        self.vis_object = context.object
        self.props = context.scene.ODENT_Props
        self.source_mha = context.object.get(OdentConstants.MHA_PATH_TAG)
        if not self.source_mha:
            message = [
                "AI segmentation : please select a valid vizualisation and retry !"
            ]
            ODENT_GpuDrawText(message_list=message, rect_color=OdentColors.red)
            return {"CANCELLED"}
        self.idx = self.vis_object["idx"]
        self.uid = self.vis_object["uid"]

        self.set_image3d(context)
        return {"FINISHED"}

    def set_image3d(self, context):
        global IMAGE3D
        if not (
            IMAGE3D
            and context.scene.get(OdentConstants.MHA_PATH_TAG) == self.source_mha
        ):
            context.scene[OdentConstants.MHA_PATH_TAG] = self.source_mha
            IMAGE3D = sitk.ReadImage(AbsPath(self.source_mha))

    def ensure_modules():
        required_modules_dict = None
        return


#################################################################################################
# Registration :
#################################################################################################
classes = [
    ODENT_OT_NewProject,
    ODENT_OT_ReloadStartup,
    ODENT_OT_Dicom_Reader,
    ODENT_OT_VolumeSlicer,
    ODENT_OT_DicomToMesh,
    ODENT_OT_DicomToMesh2,
    ODENT_OT_MultiTreshSegment,
    ODENT_OT_AutoAlign,
    ODENT_OT_AlignPoints,
    ODENT_OT_AlignPointsInfo,
    ODENT_OT_align_to_front,
    ODENT_OT_to_center,
    ODENT_OT_center_cursor,
    ODENT_OT_AlignToActive,
    ODENT_OT_AddImplant,
    ODENT_OT_RemoveImplant,
    ODENT_OT_AlignObjectsAxes,
    ODENT_OT_AddFixingPin,
    ODENT_OT_LockObjectToPointer,
    ODENT_OT_UnlockObjectFromPointer,
    ODENT_OT_ObjectToPointer,
    ODENT_OT_PointerToActive,
    ODENT_OT_FlyToImplantOrFixingSleeve,
    ODENT_OT_GuideFinaliseGeonodes,
    ODENT_OT_AddSplint,
    ODENT_OT_SplintGuide,
    ODENT_OT_AddTube,
    ODENT_OT_GuideAddComponent,
    ODENT_OT_GuideSetComponents,
    ODENT_OT_GuideSetCutters,
    ODENT_OT_SplintGuideGeom,
    ODENT_OT_guide_3d_text,
    ODENT_OT_ConnectPathCutter,
    ODENT_OT_RibbonCutterAdd,
    ODENT_OT_RibbonCutter_Perform_Cut,
    ODENT_OT_CurveCutter1_New,
    ODENT_OT_CurveCutter1_New_Perform_Cut,
    ODENT_OT_CurveCutter2_Cut_New,
    ODENT_OT_AddCustomSleeveCutter,
    ODENT_OT_CurveCutterAdd,
    ODENT_OT_CurveCutterAdd2,
    ODENT_OT_CurveCutterAdd3,
    ODENT_OT_CurveCutterCut,
    ODENT_OT_CurveCutterCut3,
    ODENT_OT_CurveCutter2_ShortPath,
    ODENT_OT_AddSquareCutter,
    ODENT_OT_square_cut_confirm,
    ODENT_OT_square_cut_exit,
    ODENT_OT_Survey,
    ODENT_OT_ModelBase,
    ODENT_OT_BlockModel,
    ODENT_OT_hollow_model,
    ODENT_OT_add_offset,
    ODENT_OT_AddColor,
    ODENT_OT_RemoveColor,
    ODENT_OT_JoinObjects,
    ODENT_OT_SeparateObjects,
    ODENT_OT_Parent,
    ODENT_OT_Unparent,
    ODENT_OT_decimate,
    ODENT_OT_clean_mesh2,
    ODENT_OT_fill,
    ODENT_OT_retopo_smooth,
    ODENT_OT_VoxelRemesh,
    ODENT_OT_LockObjects,
    ODENT_OT_UnlockObjects,
    ODENT_OT_UndercutsPreview,
    ODENT_OT_BlockoutNew,
    ODENT_OT_ImportMesh,
    ODENT_OT_ExportMesh,
    ODENT_OT_add_3d_text,
    ODENT_OT_Text3d,
    ODENT_OT_NormalsToggle,
    ODENT_OT_FlipNormals,
    ODENT_OT_SlicesPointerSelect,
    ODENT_OT_FilpCameraAxial90Plus,
    ODENT_OT_FilpCameraAxial90Minus,
    ODENT_OT_FilpCameraAxialUpDown,
    ODENT_OT_FilpCameraAxialLeftRight,
    ODENT_OT_FilpCameraCoronal90Plus,
    ODENT_OT_FilpCameraCoronal90Minus,
    ODENT_OT_FilpCameraCoronalUpDown,
    ODENT_OT_FilpCameraCoronalLeftRight,
    ODENT_OT_FilpCameraSagittal90Plus,
    ODENT_OT_FilpCameraSagittal90Minus,
    ODENT_OT_FilpCameraSagittalUpDown,
    ODENT_OT_FilpCameraSagittalLeftRight,
    ODENT_OT_AssetBrowserToggle,
    ODENT_OT_add_asset_modal,
    ODENT_OT_OverhangsPreview,
    ODENT_OT_AddReferencePlanes,
    ODENT_OT_AddMarkupPoint,
    ODENT_OT_OcclusalPlane,
    ODENT_OT_OcclusalPlaneInfo,
    ODENT_OT_DrawTester,
]

handlers_name_list = [
    "update_slices_handler_callback"
]  # "on_voxel_vizualisation_object_select"
handlers_func_list = [
    update_slices_handler_callback
]  # on_voxel_vizualisation_object_select,


def register():

    for cls in classes:
        bpy.utils.register_class(cls)

    remove_handlers_by_names(handlers_name_list)
    add_handlers_from_func_list(handlers_func_list)


def unregister():
    remove_handlers_by_names(handlers_name_list)
    try:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
    except Exception as er:
        print("Addon unregister error : ", er)
