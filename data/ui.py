import bpy  # type: ignore
from os.path import exists
from .utils import (
    OdentConstants,
    get_odent_version,
    update_is_availible,
    get_active_object_color,
    get_icon_value,
    DRAW_HANDLERS,
    check_odent_library,
)

# remote_version = update_is_availible()


class ODENT_PT_MainPanel(bpy.types.Panel):
    """Main Panel"""

    bl_idname = "ODENT_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = ""
    bl_options = {"HIDE_HEADER", "DEFAULT_CLOSED"}

    def draw(self, context):

        global remote_version, DRAW_HANDLERS
        odent_logo_id = 0
        odent_icons = OdentConstants.ODENT_ICONS
        if odent_icons :
            odent_logo = odent_icons.get("odent_logo")
            if odent_logo :
                odent_logo_id = odent_logo.icon_id
        
                
        save_alert = False
        if bpy.data.is_dirty or not bpy.data.filepath:
            save_alert = True
        
        layout = self.layout
        box = layout.box()
        b = box.box()
        g = b.grid_flow(columns=2, align=True)
        scale_y = 1.4
        
        g.scale_y = scale_y
        if odent_logo_id:
            g.template_icon(icon_value=odent_logo_id, scale=scale_y)
        g.operator("odent.welcome_dialog", text=f"{OdentConstants.ADDON_NAME} - ver. {int(OdentConstants.ADDON_VER_DATE)}")
        
        # r = box.row()
        # r.alignment = 'EXPAND'
        # r.scale_y = rscale_y
        # if odent_icon_id:
        #     r.template_icon(icon_value=odent_icon_id, scale=1.5)
        # r.operator("odent.welcome_dialog", text=f"{OdentConstants.ADDON_NAME}, (ver. {OdentConstants.ADDON_VER_DATE})")
        
        # split = box.split(factor=0.2, align=True)
        
        # if odent_icon_id:
        #     coll = split.column()
        #     coll.alignment = 'LEFT'
        #     coll.template_icon(icon_value=odent_icon_id, scale=1.5)
        
        # coll = split.column()
        # coll.scale_y = 1.5
        # coll.operator("odent.welcome_dialog", text=f"{OdentConstants.ADDON_NAME}, (ver. {OdentConstants.ADDON_VER_DATE})")
               

        # if OdentConstants.REMOTE_VERSION :
        #     grid = box.grid_flow(columns=1, align=True)
        #     grid.alert = True
        #     grid.label(text=f"ODent update is availible :{remote_version} ")
        #     grid.operator("wm.odent_checkupdate", text="Update", icon="FILE_REFRESH")

        if not bpy.data.filepath:
            # box = layout.box()
            
            r = box.row()
            r.alignment = "CENTER"
            r.scale_y = scale_y
            r.alert = True
            r.label(text="Please add new Project, or open existing one!")
            
            b = box.box()
            split = b.split(factor=0.5, align=True)

            coll = split.column(align=True)
            coll.alert = True
            coll.operator("wm.odent_new_project")

            coll = split.column(align=True)
            coll.operator("wm.open_mainfile")
            
            # g = box.grid_flow(columns=1, align=True)
            # g.alert = True
            # g.scale_y = scale_y
            # g.label(text="Please add new Project, or open existing one!")
            # split = box.split(factor=0.5, align=True)

            # coll = split.column(align=True)
            # coll.alert = True
            # coll.operator("wm.odent_new_project")

            # coll = split.column(align=True)
            # coll.operator("wm.open_mainfile")

        else:
            g = box.grid_flow(columns=1, align=True)
            g.operator("wm.open_mainfile")

            g = box.grid_flow(columns=2, align=True)
            g.alert = save_alert
            g.operator("wm.save_mainfile", text="Save", icon="FILE_TICK")
            g.operator("wm.save_as_mainfile", text="Save As...", icon="FILE_BLEND")

            g = box.grid_flow(columns=2, align=True)
            g.operator("wm.odent_import_mesh", icon="IMPORT")
            g.operator("wm.odent_export_mesh", icon="EXPORT")

        if DRAW_HANDLERS:
            g = box.grid_flow(columns=1, align=True)
            g.operator("wm.odent_remove_info_footer", icon="CANCEL")

        # layout.operator("wm.odent_draw_tester")

        # box = layout.box()
        # box.operator("wm.odent_connect_path_cutter")


class ODENT_PT_DicomPanel(bpy.types.Panel):
    """Dicom Panel"""

    bl_idname = "ODENT_PT_DicomPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = "DICOM"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):

        ODENT_Props = context.scene.ODENT_Props
        layout = self.layout

        if not bpy.data.filepath:
            box = layout.box()
            g = box.grid_flow(columns=1, align=True)
            g.alert = True
            g.label(text="Please add new Project, or open existing one.")

        else:
            box = layout.box()
            g = box.grid_flow(columns=2, align=True)
            g.prop(ODENT_Props, "dicomDataType", text="")
            if ODENT_Props.dicomDataType == "DICOM Series":
                g.prop(ODENT_Props, "UserDcmDir", text="")
            elif ODENT_Props.dicomDataType == "3D Image File":
                g.prop(ODENT_Props, "UserImageFile", text="")

            split = box.split(factor=0.4, align=True)
            coll = split.column(align=True)
            coll.label(text="Visualisation options:")
            coll = split.column(align=True)
            coll.prop(ODENT_Props, "visualisation_mode", text="")

            if ODENT_Props.visualisation_mode == OdentConstants.VISUALISATION_MODE_PCD:

                split = box.split(factor=0.4, align=True)
                coll = split.column(align=True)
                coll.label(text="Point cloud sampling:")
                coll = split.column(align=True)
                coll.prop(ODENT_Props, "pcd_sampling_method", text="")

                split = box.split(factor=0.4, align=True)
                coll = split.column(align=True)
                coll.label(text="Points max (millions):")
                coll = split.column(align=True)
                coll.prop(ODENT_Props, "pcd_points_max", text="")

            g = box.grid_flow(columns=1, align=False)
            g.operator("wm.odent_dicom_reader", text="Read DICOM", icon="IMPORT")

            if (
                context.object
                and context.object.get(OdentConstants.ODENT_TYPE_TAG)
                == OdentConstants.PCD_OBJECT_TYPE
            ):
                box = layout.box()
                # g = box.grid_flow(columns=1, align=True)
                # g.prop(ODENT_Props, "ThresholdMin", text="threshold", slider=True)

                box.label(text="Point cloud settings :")
                g = box.grid_flow(columns=2, align=True)
                g.prop(
                    ODENT_Props, "pcd_point_radius", text="Point radius", slider=True
                )
                g.prop(ODENT_Props, "pcd_point_auto_resize", text="Auto resize")

                g = box.grid_flow(columns=1, align=True)

                g.prop(ODENT_Props, "pcd_points_opacity", text="Opacity", slider=True)
                g.prop(ODENT_Props, "pcd_points_emission", text="Emission", slider=True)

            box = layout.box()
            g = box.grid_flow(columns=1, align=True)
            g.prop(ODENT_Props, "ThresholdMin", text="threshold", slider=True)

            g = box.grid_flow(columns=1, align=True)
            g.label(text="Mesh segmentation:")

            g = box.grid_flow(columns=3, align=True)
            g.prop(ODENT_Props, "SegmentColor", text="")
            g.prop(
                ODENT_Props, "slicesColorThresholdBool", text="Show Segmentation Label"
            )
            g.operator(
                "wm.odent_dicom_to_mesh", text="DICOM to Mesh", icon="MESH_CUBE"
            )


class ODENT_PT_SlicesPanel(bpy.types.Panel):
    """Slices Panel"""

    bl_idname = "ODENT_PT_SlicesPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = "SLICES"
    bl_options = {"DEFAULT_CLOSED"}

    # @classmethod
    # def poll(cls, context):
    #     if context.screen.name == "Odent Slicer" :
    #         return True
    #     else:
    #         return False

    def draw(self, context):
        # if context.screen.name != "Odent Slicer" :
        #     return
        ODENT_Props = context.scene.ODENT_Props
        layout = self.layout

        if not bpy.data.filepath:
            box = layout.box()
            g = box.grid_flow(columns=1, align=True)
            g.alert = True
            g.label(text="Please add new Project, or open existing one.")

        else:

            box = layout.box()
            g = box.grid_flow(columns=1, align=True)
            g.prop(ODENT_Props, "ThresholdMin", text="threshold", slider=True)

            g = box.grid_flow(columns=2, align=True)
            g.prop(ODENT_Props, "SegmentColor", text="")
            g.prop(
                ODENT_Props, "slicesColorThresholdBool", text="Show Segmentation Label"
            )

            g = box.grid_flow(columns=2, align=True)
            g.operator(
                "wm.odent_slices_pointer_select",
                text="Select Pointer",
                icon="EMPTY_AXIS",
            )
            g.operator(
                "wm.odent_volume_slicer", text="Update Dicom Slices", icon="EMPTY_AXIS"
            )

            slices_gn = bpy.data.node_groups.get(OdentConstants.SLICES_SHADER_NAME)
            if slices_gn:
                row = box.row()
                row.alert = True
                row.prop(ODENT_Props, "slices_brightness", text="Brightness")
                row.prop(ODENT_Props, "slices_contrast", text="Contrast")

            row = box.row()
            row.label(text="Axial Slice Flip :")

            row = box.row()
            row.operator("wm.odent_flip_camera_axial_90_plus", icon="PLUS")
            row.operator("wm.odent_flip_camera_axial_90_minus", icon="REMOVE")
            row.operator("wm.odent_flip_camera_axial_up_down", icon="TRIA_UP")
            row.operator("wm.odent_flip_camera_axial_left_right", icon="TRIA_RIGHT")

            row = box.row()
            row.label(text="Coronal Slice Flip :")

            row = box.row()
            row.operator("wm.odent_flip_camera_coronal_90_plus", icon="PLUS")
            row.operator("wm.odent_flip_camera_coronal_90_minus", icon="REMOVE")
            row.operator("wm.odent_flip_camera_coronal_up_down", icon="TRIA_UP")
            row.operator("wm.odent_flip_camera_coronal_left_right", icon="TRIA_RIGHT")

            row = box.row()
            row.label(text="Sagittal Slice Flip :")

            row = box.row()
            row.operator("wm.odent_flip_camera_sagittal_90_plus", icon="PLUS")
            row.operator("wm.odent_flip_camera_sagittal_90_minus", icon="REMOVE")
            row.operator("wm.odent_flip_camera_sagittal_up_down", icon="TRIA_UP")
            row.operator("wm.odent_flip_camera_sagittal_left_right", icon="TRIA_RIGHT")


class ODENT_PT_ToolsPanel(bpy.types.Panel):
    """Tools Panel"""

    bl_idname = "ODENT_PT_ToolsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = "TOOLS"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        ODENT_Props = context.scene.ODENT_Props
        ob = context.object
        layout = self.layout

        is_valid_object, mat_info = get_active_object_color()
        if is_valid_object:

            try:
                box = layout.box()
                grid = box.grid_flow(columns=2, align=True)
                if mat_info:
                    grid.prop(mat_info[0], mat_info[1], text="")
                    grid.label(text="COLOR")
                else:
                    grid.template_icon(get_icon_value("COLORSET_12_VEC"), scale=1.5)
                    grid.label(text="COLOR")  # icon=yellow_point

                grid = box.grid_flow(columns=2, align=True)
                grid.operator("wm.odent_add_color", text="Add Color", icon="MATERIAL")
                grid.operator("wm.odent_remove_color", text="Remove Color")

            except:
                pass

        box = layout.box()
        grid = box.grid_flow(columns=2, align=True)
        grid.label(text="RELATIONS")
        grid.template_icon(get_icon_value("LINKED"), scale=1.5)

        grid = box.grid_flow(columns=2, align=True)
        grid.operator("wm.odent_parent_object", text="Parent", icon="LINKED")
        grid.operator("wm.odent_join_objects", text="Join", icon="SNAP_FACE")
        grid.operator("wm.odent_lock_objects", text="Lock", icon="LOCKED")

        grid.operator(
            "wm.odent_unparent_objects",
            text="Un-Parent",
            icon="LIBRARY_DATA_OVERRIDE",
        )
        grid.operator("wm.odent_separate_objects", text="Separate", icon="SNAP_VERTEX")
        grid.operator("wm.odent_unlock_objects", text="Un-Lock", icon="UNLOCKED")

        # Model Repair Tools :
        box = layout.box()
        grid = box.grid_flow(columns=2, align=True)
        grid.label(text="REPAIR")
        grid.template_icon(get_icon_value("TOOL_SETTINGS"), scale=1.5)

        grid = box.grid_flow(columns=2, align=True)

        grid.operator("wm.odent_decimate", text="Decimate", icon="MOD_DECIM")
        grid.operator("wm.odent_clean_mesh2", text="Clean Mesh", icon="BRUSH_DATA")
        grid.operator(
            "wm.odent_retopo_smooth", text="Retopo-Smooth", icon="SMOOTHCURVE"
        )
        grid.operator("wm.odent_normals_toggle")

        grid.prop(ODENT_Props, "decimate_ratio", text="")
        grid.operator("wm.odent_voxelremesh", text="Remesh", icon="MOD_REMESH")
        if ob and ob.mode == "SCULPT":
            try:
                grid.operator("sculpt.sample_detail_size", text="", icon="EYEDROPPER")
            except:
                grid.operator(
                    "wm.odent_fill", text="Fill", icon="OUTLINER_OB_LIGHTPROBE"
                )
        else:
            grid.operator("wm.odent_fill", text="Fill", icon="OUTLINER_OB_LIGHTPROBE")
        grid.operator("wm.odent_flip_normals")

        # Cutting Tools :
        box = layout.box()
        g = box.grid_flow(columns=2, align=True)
        g.label(text="CUT")
        g.template_icon(get_icon_value("COLOR"), scale=1.5)

        g = box.grid_flow(columns=1, align=True)
        g.prop(ODENT_Props, "Cutting_Tools_Types_Prop", text="Cutters")
        if ODENT_Props.Cutting_Tools_Types_Prop == "Path Split":
            g = box.grid_flow(columns=2, align=True)
            g.operator(
                "wm.odent_curvecutteradd2",
                text="Add Cutter",
                icon="GP_SELECT_STROKES",
            )
            g.operator(
                "wm.odent_curvecutter2_cut_new",
                text="Perform Cut",
                icon="GP_MULTIFRAME_EDITING",
            )

        elif ODENT_Props.Cutting_Tools_Types_Prop == "Ribbon Split":
            g = box.grid_flow(columns=2, align=True)
            g.operator(
                "wm.odent_curvecutter1_new",
                text="Add Cutter",
                icon="GP_SELECT_STROKES",
            )

            g.operator(
                "wm.odent_curvecutter1_new_perform_cut",
                text="Perform Cut",
                icon="GP_MULTIFRAME_EDITING",
            )

        elif ODENT_Props.Cutting_Tools_Types_Prop == "Ribbon Cutter":
            g = box.grid_flow(columns=2, align=True)
            g.operator(
                "wm.odent_ribboncutteradd",
                text="Add Cutter",
                icon="GP_SELECT_STROKES",
            )
            g.operator(
                "wm.odent_ribboncutter_perform_cut",
                text="Perform Cut",
                icon="GP_MULTIFRAME_EDITING",
            )
        elif ODENT_Props.Cutting_Tools_Types_Prop == "Frame Cutter":

            g = box.grid_flow(columns=2, align=True)
            g.prop(ODENT_Props, "cutting_mode", text="Cutting Mode")
            g.operator("wm.odent_add_square_cutter", text="Frame Cutter")

        if context.object and context.object.get("odent_type"):

            obj = context.object
            if obj.get("odent_type") in ["curvecutter1", "curvecutter2"]:
                g = box.grid_flow(columns=2, align=True)
                g.prop(obj.data, "extrude", text="Extrude")
                g.prop(obj.data, "offset", text="Offset")
            elif obj.get("odent_type") == "curvecutter3":
                g = box.grid_flow(columns=3, align=True)
                g.prop(obj.data, "extrude", text="Extrude")
                g.prop(obj.data, "bevel_depth", text="Bevel")
                g.prop(obj.data, "offset", text="Offset")

        box = layout.box()
        grid = box.grid_flow(columns=2, align=True)
        grid.label(text="MODEL")
        grid.template_icon(get_icon_value("FILE_VOLUME"), scale=1.5)

        grid = box.grid_flow(columns=2, align=True)

        grid.operator("wm.odent_model_base", text="Make Model Base")
        grid.operator("wm.odent_undercuts_preview", text="Preview Undercuts")

        grid.operator("wm.odent_add_offset", text="Add Offset")
        grid.operator("wm.odent_blockout_new", text="Blocked Model")

        box = layout.box()
        grid = box.grid_flow(columns=2, align=True)
        grid.label(text="TEXTE")
        grid.template_icon(get_icon_value("SMALL_CAPS"), scale=1.5)

        grid = box.grid_flow(columns=2, align=True)
        grid.prop(ODENT_Props, "text", text="")
        grid.operator("wm.odent_add_3d_text", text="Add 3D Text")


class ODENT_PT_ODentLibrary(bpy.types.Panel):
    """Odent Library Panel"""

    bl_idname = "ODENT_PT_ODentLibrary"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = "LIBRARY"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        grid = box.grid_flow(columns=1, align=True)
        if not OdentConstants.ODENT_LIB_IS_OK :
            grid.alert = True
            grid.label(text="Odent library is not installed")
            grid.operator("wm.odent_add_odent_library", text="Install Odent Library", icon="TOOL_SETTINGS")
        
        else :
            grid.operator("wm.odent_asset_browser_toggle")
            grid.operator("wm.odent_add_asset_modal")


class ODENT_PT_Align(bpy.types.Panel):
    """ALIGN Panel"""

    bl_idname = "ODENT_PT_Main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = "ALIGN"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        ODENT_Props = context.scene.ODENT_Props
        layout = self.layout

        box = layout.box()

        row = box.row()
        row.operator("wm.odent_auto_align")

        g = box.grid_flow(columns=2, align=True)
        g.operator("wm.odent_alignpoints", text="ALIGN Points")
        g.operator("wm.odent_alignpointsinfo", text="INFO", icon="INFO")

        is_ready = (
            context.object
            and context.object in context.selected_objects
            and len(context.selected_objects) == 2
        )
        txt = []
        if ODENT_Props.AlignModalState:
            txt = ["WAITING FOR ALIGNEMENT..."]

        elif is_ready:
            target_name = context.object.name
            src_name = [
                obj for obj in context.selected_objects if not obj is context.object
            ][0].name

            txt = [
                "READY FOR ALIGNEMENT.",
                f"{src_name} will be aligned to, {target_name}",
            ]

        else:
            txt = ["STANDBY MODE"]

        #########################################
        if txt:
            b2 = box.box()
            b2.alert = True

            for t in txt:
                b2.label(text=t)

        # Align Tools :
        box = layout.box()
        g = box.grid_flow(columns=3, align=True)
        g.operator("wm.odent_align_to_front", text="Align To Me", icon="AXIS_FRONT")
        g.operator("wm.odent_to_center", text="Move To Center", icon="SNAP_FACE_CENTER")
        g.operator("wm.odent_align_to_active", text="Align To Active")

        g = box.grid_flow(columns=2, align=True)
        g.operator("wm.odent_occlusalplane", text="OCCLUSAL PLANE")
        g.operator("wm.odent_occlusalplaneinfo", text="INFO", icon="INFO")

        g = box.grid_flow(columns=2, align=True)
        g.operator("wm.odent_add_reference_planes", text="Ref planes")


class ODENT_PT_ImplantPanel(bpy.types.Panel):
    """Implant panel"""

    bl_idname = "ODENT_PT_ImplantPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = "IMPLANT"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):

        layout = self.layout
        props = context.scene.ODENT_Props
        ####################################################################            
        box = layout.box()
        g = box.grid_flow(columns=3, align=True)

        g.operator(
            "wm.odent_slices_pointer_select",
            text="Select Pointer",
            icon="EMPTY_AXIS",
        )
        g.operator(
            "wm.odent_fly_to_implant_or_fixing_sleeve", text="", icon="TRIA_LEFT"
        ).direction = "previous"
        g.operator(
            "wm.odent_fly_to_implant_or_fixing_sleeve", text="", icon="TRIA_RIGHT"
        ).direction = "next"

        g = box.grid_flow(columns=2, align=True)

        g.operator("wm.odent_add_implant", text="Add Implant", icon="ADD")
        g.operator("wm.odent_remove_implant", text="Remove Implant", icon="REMOVE")

        g = box.grid_flow(columns=1, align=True)
        g.operator("wm.add_fixing_pin")

        g = box.grid_flow(columns=2, align=True)
        g.operator("wm.odent_lock_object_to_pointer")
        g.operator("wm.odent_unlock_object_from_pointer")

        g = box.grid_flow(columns=2, align=True)
        g.operator("wm.odent_object_to_pointer", text="Active to Pointer")
        g.operator("wm.odent_pointer_to_active", text="Pointer to Active")

        g = box.grid_flow(columns=1, align=True)
        g.operator("wm.odent_align_objects_axes", text="Align Axes")
        ###########################################################################
        #implant gn parameters :
        if context.object and context.object.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.ODENT_IMPLANT_TYPE:
                
            mod = context.object.modifiers.get(OdentConstants.ODENT_IMPLANT_MODIFIER_NAME)
            if mod :
                box = layout.box()
                box.label(text="Implant:")
                
                group = mod.node_group
                items = group.interface.items_tree
                tooth_socket = next((i for i in items if "tooth" in i.name.lower()), None)
                
                if tooth_socket:
                    b = box.box()
                    b.label(text="tooth number:")
                    upper,lower= props.get_upper_and_lower_teeth(context)
                    
                    g = b.grid_flow(columns=14, align=True)
                    for t_id in upper:
                        g.prop_enum(mod, f'["{tooth_socket.identifier}"]', value=t_id, text=t_id)
                    
                    g = b.grid_flow(columns=14, align=True)
                    for t_id in lower:
                        g.prop_enum(mod, f'["{tooth_socket.identifier}"]', value=t_id, text=t_id)
                
                
        
                
                
                # b.prop(mod,  '["Socket_23"]', text="tooth number:")
                b.prop(mod,  '["Socket_39"]', text="Rotate implant")
                b.prop(mod,  '["Socket_3"]', text="Implant diameter")
                b.prop(mod,  '["Socket_2"]', text="Implant length:")
                
                b = box.box()
                b.label(text="Abutment :")
                
                split = b.split(factor=0.3, align=True)
                coll = split.column(align=True)
                coll.label(text="Custom abutment")
                coll = split.column(align=True)
                coll.prop(mod,  '["Socket_41"]', text="")
                
                
                
                
                # r = b.row()
                # r.prop(mod,  '["Socket_42"]', text=" ",expand=True)
                # if mod["Socket_42"] == 1:
                #     split = b.split(factor=0.3, align=True)
                #     coll = split.column(align=True)
                #     coll.label(text="Custom abutment")
                #     coll = split.column(align=True)
                #     coll.prop(mod,  '["Socket_41"]', text="")
                    # r = b.row()
                    # r.alignment = "EXPAND"
                    # r.label(text="Custom abutment")
                    # r.prop(mod,  '["Socket_41"]', text="")
                
                b = box.box()
                b.label(text="Sleeve :")
                b.prop(mod,  '["Socket_44"]', text="Add sleeve bevel")
                b.prop(mod,  '["Socket_6"]', text="External diameter")
                b.prop(mod,  '["Socket_7"]', text="Height (offset)")
                b.prop(mod,  '["Socket_17"]', text="Cut sleeve bottom")
                
                b.label(text="Sleeve drill hole cutter :")
                r = b.row()
                r.prop(mod,  '["Socket_32"]', text=" ", expand=True)
                r = b.row()
                if mod["Socket_32"] == 1: #custom cutter (0 is cylinder cutter)
                    r.prop(mod,  '["Socket_37"]', text="Custom cutter")
                else :
                    r.prop(mod,  '["Socket_14"]', text="Hole diameter")
                
                
                # r = b.row()
                # r.prop(mod,  '["Socket_32"]', text="Sleeve internal cutter :", expand=True)
                # if mod["Socket_32"] == 1: #custom cutter (0 is cylinder cutter)
                #     r = b.row()
                #     r.prop(mod,  '["Socket_37"]', text="Custom Sleeve:")
                # else :
                #     b.prop(mod,  '["Socket_14"]', text="Internal diameter:")
                
                
                b = box.box()
                b.label(text="Show parts :")
                # r = b.row()
                # r.prop(props, 'show_all_parts', text="Show all")
                # r.prop(props, 'hide_all_parts', text="Hide all")
                cf = b.column_flow(columns=2, align=True)
                cf.prop(mod,  '["Socket_16"]', text="Sleeve")
                cf.prop(mod,  '["Socket_15"]', text="Drill hole cutter")
                cf.prop(mod,  '["Socket_22"]', text="Implant axis")
                cf.prop(mod,  '["Socket_29"]', text="Implant info")
                cf.prop(mod,  '["Socket_40"]', text="Abutement")  
                cf.prop(mod,  '["Socket_36"]', text="Safe zone")
                if mod["Socket_36"] == 1:
                    cf.prop(mod,  '["Socket_34"]', text="Safe zone height")
                    cf.prop(mod,  '["Socket_33"]', text="Safe zone width")
        


class ODENT_PT_Guide(bpy.types.Panel):
    """Guide Panel"""

    bl_idname = "ODENT_PT_Guide"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = OdentConstants.ADDON_NAME
    bl_label = "SURGICAL GUIDE"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        ODENT_Props = context.scene.ODENT_Props
        layout = self.layout

        box = layout.box()
        row = box.row()
        row.operator("wm.odent_add_guide_splint_geom")
        row = box.row()
        row.operator("wm.odent_add_tube")
        row.prop(ODENT_Props, "TubeCloseMode", text="")

        if context.active_object:
            if (
                "ODENT_GuideTube" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):
                obj = context.active_object
                row = box.row()
                row.prop(obj.data, "bevel_depth", text="Radius")
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        row = box.row()
        row.operator("wm.odent_guide_add_component")
        row = box.row()
        row.operator("wm.odent_guide_finalise_geonodes")


####################################################################

##########################################################################################
# Registration :
##########################################################################################

classes = [
    ODENT_PT_MainPanel,
    ODENT_PT_DicomPanel,
    ODENT_PT_SlicesPanel,
    ODENT_PT_ToolsPanel,
    ODENT_PT_ODentLibrary,
    ODENT_PT_Align,
    ODENT_PT_ImplantPanel,
    ODENT_PT_Guide,
    
    
]


def register():
    global ADDON_VER_DATE
    ADDON_VER_DATE = get_odent_version(filepath=OdentConstants.ADDON_VER_PATH)

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
