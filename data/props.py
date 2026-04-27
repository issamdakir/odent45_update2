import bpy # type: ignore
from os.path import abspath, basename, exists, join
import numpy as np # type: ignore
from time import sleep
from .Operators.ODENT_Operators import force_update_slices

from bpy.props import ( # type: ignore
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)
from .utils import (
    OdentConstants,
    set_enum_items,
    HuTo255,
)


def on_segmentation_color_update(self, context) :
    if self.slicesColorThresholdBool:
        force_update_slices()
def on_slices_color_threshold_bool_update(self, context) :
    force_update_slices()

def on_threshold_min_update(self, context):
    
    vis_coll = bpy.data.collections.get(OdentConstants.DICOM_VIZ_COLLECTION_NAME)
    if not vis_coll or not vis_coll.objects:
        return
    vis_objects = [obj for obj in vis_coll.objects if obj.get(OdentConstants.ODENT_TYPE_TAG) in (
        OdentConstants.PCD_OBJECT_TYPE,
        OdentConstants.VOXEL_OBJECT_TYPE
    )]
    if not vis_objects:
        return
    targets_list = []
    active_obj = context.object
    if active_obj and active_obj.select_get() and active_obj in vis_objects:
        targets_list.append(active_obj)
    else:
        targets_list = vis_objects
    for target in targets_list:
        if target.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.PCD_OBJECT_TYPE :
            pcd_node_group = target.modifiers[OdentConstants.PCD_GEONODE_MODIFIER_NAME].node_group
            pcd_theshold = pcd_node_group.nodes.get(OdentConstants.PCD_THRESHOLD_NODE_NAME)
            if pcd_theshold:
                threshold_255 = HuTo255(self.ThresholdMin)
                pcd_theshold.integer = threshold_255
                # print(f"pcd threshold updated = {pcd_theshold.integer}")
                # target.data.update()

        elif target.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.VOXEL_OBJECT_TYPE :
            voxel_node_group = bpy.data.node_groups[target[OdentConstants.VOXEL_NODE_NAME_TAG]]
            voxel_node_group.nodes["Low_Treshold"].outputs[0].default_value = self.ThresholdMin
            
    if self.slicesColorThresholdBool:
        force_update_slices()
    
def on_pcd_point_radius_update(self, context):
    vis_coll = bpy.data.collections.get(OdentConstants.DICOM_VIZ_COLLECTION_NAME)
    if not vis_coll or not vis_coll.objects:
        return
    pcd_vis_check = [obj for obj in vis_coll.objects if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.PCD_OBJECT_TYPE]
    if not pcd_vis_check:
        return
    
    targets_list = []
    active_obj = context.object
    if active_obj and active_obj.select_get() and active_obj in pcd_vis_check:
        targets_list.append(active_obj)
    else:
        targets_list = pcd_vis_check

    for target in targets_list: 
        pcd_node_group = target.modifiers[OdentConstants.PCD_GEONODE_MODIFIER_NAME].node_group
        pcd_opacity = pcd_node_group.nodes.get(OdentConstants.PCD_POINT_RADIUS_NODE_NAME)
        if pcd_opacity:
            pcd_opacity.outputs[0].default_value = self.pcd_point_radius

def on_pcd_point_auto_resize_update(self, context):
    vis_coll = bpy.data.collections.get(OdentConstants.DICOM_VIZ_COLLECTION_NAME)
    if not vis_coll or not vis_coll.objects:
        return
    pcd_vis_check = [obj for obj in vis_coll.objects if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.PCD_OBJECT_TYPE]
    if not pcd_vis_check:
        return
    
    targets_list = []
    active_obj = context.object
    if active_obj and active_obj.select_get() and active_obj in pcd_vis_check:
        targets_list.append(active_obj)
    else:
        targets_list = pcd_vis_check

    for target in targets_list: 
        pcd_node_group = target.modifiers[OdentConstants.PCD_GEONODE_MODIFIER_NAME].node_group
        pcd_point_auto_resize = pcd_node_group.nodes.get(OdentConstants.PCD_POINT_AUTO_RESIZE_NODE_NAME)
        if pcd_point_auto_resize:
            pcd_point_auto_resize.boolean = self.pcd_point_auto_resize

def on_pcd_points_opacity_update(self, context):
    vis_coll = bpy.data.collections.get(OdentConstants.DICOM_VIZ_COLLECTION_NAME)
    if not vis_coll or not vis_coll.objects:
        return
    pcd_vis_check = [obj for obj in vis_coll.objects if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.PCD_OBJECT_TYPE]
    if not pcd_vis_check:
        return
    
    targets_list = []
    active_obj = context.object
    if active_obj and active_obj.select_get() and active_obj in pcd_vis_check:
        targets_list.append(active_obj)
    else:
        targets_list = pcd_vis_check

    for target in targets_list: 
        pcd_node_group = target.modifiers[OdentConstants.PCD_GEONODE_MODIFIER_NAME].node_group
        pcd_points_opacity = pcd_node_group.nodes.get(OdentConstants.PCD_OPACITY_NODE_NAME)
        if pcd_points_opacity:
            pcd_points_opacity.integer = self.pcd_points_opacity

def on_pcd_points_emission_update(self, context):
    vis_coll = bpy.data.collections.get(OdentConstants.DICOM_VIZ_COLLECTION_NAME)
    if not vis_coll or not vis_coll.objects:
        return
    pcd_vis_check = [obj for obj in vis_coll.objects if obj.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.PCD_OBJECT_TYPE]
    if not pcd_vis_check:
        return
    
    targets_list = []
    active_obj = context.object
    if active_obj and active_obj.select_get() and active_obj in pcd_vis_check:
        targets_list.append(active_obj)
    else:
        targets_list = pcd_vis_check

    for target in targets_list: 
        pcd_mat = bpy.data.materials.get(OdentConstants.PCD_MAT_NAME)
        if pcd_mat :
            pcd_points_emission = pcd_mat.node_tree.nodes[OdentConstants.PCD_POINT_EMISSION_NODE_NAME].outputs[0]
            pcd_points_emission.default_value = self.pcd_points_emission
         

def get_dicom_series_items_callback(self, context):
    enum_items = set_enum_items(["EMPTY"])

    if self.dicomDataType == "DICOM Series" :
        dicom_series_dictionary = eval(self.dicom_series_dictionary)
        if dicom_series_dictionary :
            enum_txt_list = list(dicom_series_dictionary.keys())
            if enum_txt_list :
                enum_items = set_enum_items(enum_txt_list)
            
    return enum_items

def on_dicom_series_update(self, context):
    if self.dicomDataType == "DICOM Series" :
        dicom_series_dictionary = eval(self.dicom_series_dictionary)
        if dicom_series_dictionary and dicom_series_dictionary.get(self.Dicom_Series) :
            spacing = dicom_series_dictionary[self.Dicom_Series].get("spacing")
            if spacing :
                self.scan_resolution = spacing
                        
def update_text(self, context):
    if context.object :
        if context.object.type == "FONT" :
            if context.object.get("odent_type") == "odent_text" :
                context.object.data.body = self.text
    return None

def brightness_update(self, context):
    slices_gn = bpy.data.node_groups.get(OdentConstants.SLICES_SHADER_NAME)
    if slices_gn :
        bright_contrast_node = slices_gn.nodes["Bright/Contrast"]
        bright_contrast_node.inputs[1].default_value = self.slices_brightness

def contrast_update(self, context):
    slices_gn = bpy.data.node_groups.get(OdentConstants.SLICES_SHADER_NAME)
    if slices_gn :
        bright_contrast_node = slices_gn.nodes["Bright/Contrast"]
        bright_contrast_node.inputs[2].default_value = self.slices_contrast

     
class ODENT_Props(bpy.types.PropertyGroup):
    def get_tooth_items(self, context):
        upper = ["17","16","15","14","13","12","11","21","22","23","24","25","26","27"]
        lower = ["47","46","45","44","43","42","41","31","32","33","34","35","36","37"]

        items = set_enum_items(upper + lower)
        return items
    
    def get_upper_and_lower_teeth(self, context):
        upper = ["17","16","15","14","13","12","11","21","22","23","24","25","26","27"]
        lower = ["47","46","45","44","43","42","41","31","32","33","34","35","36","37"]
        return upper,lower
    
    # def  on_show_all_parts_update(self, context):
    #     s_identifiers = [f"Socket_{n}"for n in  [16, 15, 22, 29, 40, 36]]
    #     if context.object and context.object.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.ODENT_IMPLANT_TYPE:
    #         if self.show_all_parts:
    #             mod = context.object.modifiers.get(OdentConstants.ODENT_IMPLANT_MODIFIER_NAME)
    #             if mod :
    #                 for s in mod.node_group.interface.items_tree:
    #                     if isinstance(s, bpy.types.NodeTreeInterfaceSocket) and s.identifier in s_identifiers:
    #                         s.default_value = True
                        
    #                     # mod[f"Socket_{socket_num}"] = True
    #                 self.hide_all_parts = False
                    
    #     return
    # def  on_hide_all_parts_update(self, context):
    #     if context.object and context.object.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.ODENT_IMPLANT_TYPE:
    #         if self.hide_all_parts:
    #             mod = context.object.modifiers.get(OdentConstants.ODENT_IMPLANT_MODIFIER_NAME)
    #             if mod :
    #                 for socket_num in (
    #                     16, 15, 22, 29, 40, 36,
    #                 ):
    #                     mod[f"Socket_{socket_num}"] = False
    #                 self.show_all_parts = False
    #                 bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP')
    #     return
    
    
    # show_all_parts :  BoolProperty(
    #     name="Show All Parts",
    #     default=False,
    #     description="Show All Parts",
    #     update = on_show_all_parts_update,
    # )
    
    # hide_all_parts :  BoolProperty(
    #     name="Hide All Parts",
    #     default=False,
    #     description="Show All Parts",
    #     update = on_hide_all_parts_update,
    # )
    
    
    tooth_number: EnumProperty(
        name="Tooth Number",
        items=get_tooth_items,
        description="Tooth Number",
        default=0,
    )

    ProjectNameProp: StringProperty(
        name="Project Name",
        default='',
        description="Project Name",
    ) # type: ignore
    UserProjectDir: StringProperty(
        name="UserProjectDir",
        default="",
        description="User project directory",
        subtype="DIR_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"}
        # update=update_user_project_dir,
    ) # type: ignore

    UserDcmDir: StringProperty(
        name="DICOM Path",
        default="",
        description="DICOM Directory Path",
        subtype="DIR_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"}
    ) # type: ignore

    UserImageFile: StringProperty(
        name="User 3D Image File Path",
        default="",
        description="User Image File Path",
        subtype="FILE_PATH",
        options={"PATH_SUPPORTS_BLEND_RELATIVE"}
    ) # type: ignore

    Dicom_Series_mode: EnumProperty(items = set_enum_items(["Simple Mode", "Advanced Mode"]), description="Dicom Read Mode", default="Advanced Mode") # type: ignore

    Dicom_Series: EnumProperty(items = get_dicom_series_items_callback, description="Dicom series",update=on_dicom_series_update) # type: ignore
    
    OrganizeInfoProp: StringProperty(
        name="OrganizeInfo",
        default='{}',
        description="Organize Information",
    ) # type: ignore
    
    dicom_dictionary: StringProperty(
        name="Dicom Dictionary",
        default='{}',
        description="Dicom Dictionary",
    ) # type: ignore
    
    dicom_series_dictionary: StringProperty(
        name="Dicom Series Dictionary",
        default='{}',
        description="Dicom Series Dictionary",
    ) # type: ignore
    
    current_dicom_dictionary: StringProperty(
        name="Current Dicom Dictionary",
        default='{}',
        description="Current Dicom Dictionary",
    ) # type: ignore

    scan_resolution: FloatProperty(
        name="Scan Resolution",
        description="Scan spacing parameter",
        default=0.0,
        step=1,
        precision=3,
    ) # type: ignore

    visualisation_mode : EnumProperty(
        items=set_enum_items([
            OdentConstants.VISUALISATION_MODE_PCD,
            OdentConstants.VISUALISATION_MODE_TEXTURED]),
        name="Visualization Mode",
        default="Point cloud 3D",
        description="3D Dicom Visualisation Mode",
    ) # type: ignore
    
    

    pcd_sampling_method : EnumProperty(
        items=set_enum_items([
            OdentConstants.PCD_SAMPLING_METHOD_RANDOM,
            OdentConstants.PCD_SAMPLING_METHOD_GRID]),

        name="Point cloud sampling",
        default="Random sampling",
        description="point cloud sampling method",
    ) # type: ignore

    pcd_points_max : IntProperty(
        name="Max points count",
        description="Maximum total pcd points count",
        default=OdentConstants.PCD_MAX_POINTS, # millions
        min=0,
    ) # type: ignore

    SlicesDir: StringProperty(
        name="SlicesDir",
        default="",
        description="Temporary Slices directory",
        subtype="DIR_PATH",
    ) # type: ignore

    dicomDataType: EnumProperty(items=set_enum_items(["DICOM Series", "3D Image File"]), description="Data type", default="DICOM Series") # type: ignore

    DcmOrganize: StringProperty(
        name="(str) Organize Dicom",
        default="{'Deffault': None}",
        description="Dicom series files list",
    ) # type: ignore

    DcmInfo: StringProperty(
        name="(str) DicomInfo",
        default="dict()",
        description="Dicom series files list",
    )# type: ignore


    PngDir: StringProperty(
        name="Png Directory",
        default="",
        description=" PNG files Sequence Directory Path",
    )# type: ignore

    Nrrd255Path: StringProperty(
        name="Nrrd255Path",
        default="",
        description="Nrrd image3D file Path",
    ) # type: ignore

    IcpVidDict: StringProperty(
            name="IcpVidDict",
            default="None",
            description="ICP Vertices Pairs str(Dict)",
        ) # type: ignore
    #######################
    Wmin: IntProperty() # type: ignore
    Wmax: IntProperty() # type: ignore

    #######################
    # SoftTissueMode = BoolProperty(description="SoftTissue Mode ", default=False)

    GroupNodeName: StringProperty(
        name="Group shader Name",
        default="",
        description="Group shader Name",
    ) # type: ignore

    #######################
    slicesColorThresholdBool: BoolProperty(
        name="Slices Color Threshold",
        description="Slices Color Threshold",
        default=False,
        update=on_slices_color_threshold_bool_update,
    ) # type: ignore

    ThresholdMin: IntProperty(
        name="Treshold Min",
        description="Volume Treshold",
        default=600,
        min=OdentConstants.WMIN,
        max=OdentConstants.WMAX,
        soft_min=OdentConstants.WMIN,
        soft_max=OdentConstants.WMAX,
        step=1,
        update=on_threshold_min_update) # type: ignore
    
    pcd_point_radius : FloatProperty(
        name="Point radius",
        description="pcd point radius (group geometry node attribute)",
        default=0.5,
        min=0.0,
        max=1.0,
        update=on_pcd_point_radius_update,
    ) # type: ignore
    
    pcd_point_auto_resize : BoolProperty(
        name="Points auto resize",
        description="pcd point radius auto resizing toggle",
        default=True,
        update=on_pcd_point_auto_resize_update,
    ) # type: ignore
    
    pcd_points_opacity : IntProperty(
        name="Opacity",
        description="pcd points opacity",
        default=50,
        subtype='PERCENTAGE',
        min=0,
        max=100,
        update=on_pcd_points_opacity_update,
    ) # type: ignore
    
    pcd_points_emission : IntProperty(
        name="Emission",
        description="pcd points emission",
        default=30,
        subtype='PERCENTAGE',
        min=0,
        max=100,
        update=on_pcd_points_emission_update,
    ) # type: ignore
    
    
    SegmentColor: FloatVectorProperty(
        name="Segmentation Color",
        description="Segmentation mesh material Color",
        default=OdentConstants.SLICE_SEGMENT_COLOR_RGB,  # (0.8, 0.46, 0.4, 1.0),
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
        update=on_segmentation_color_update,
    ) # type: ignore

    Progress_Bar: FloatProperty(
        name="Progress_Bar",
        description="Progress_Bar",
        subtype="PERCENTAGE",
        default=0.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    ) # type: ignore
    #######################
    SoftTreshold: IntProperty(
        name="SOFT TISSU",
        description="Soft Tissu Treshold",
        default=-300,
        min=OdentConstants.WMIN,
        max=OdentConstants.WMAX,
        soft_min=OdentConstants.WMIN,
        soft_max=OdentConstants.WMAX,
        step=1,
    ) # type: ignore

    BoneTreshold: IntProperty(
        name="BONE",
        description="Bone Treshold",
        default=600,
        min=OdentConstants.WMIN,
        max=OdentConstants.WMAX,
        soft_min=OdentConstants.WMIN,
        soft_max=OdentConstants.WMAX,
        step=1,
    ) # type: ignore
    TeethTreshold: IntProperty(
        name="Teeth Treshold",
        description="Teeth Treshold",
        default=1400,
        min=OdentConstants.WMIN,
        max=OdentConstants.WMAX,
        soft_min=OdentConstants.WMIN,
        soft_max=OdentConstants.WMAX,
        step=1,
    ) # type: ignore
    
    SoftBool: BoolProperty(description="", default=False) # type: ignore
    BoneBool: BoolProperty(description="", default=False) # type: ignore
    TeethBool: BoolProperty(description="", default=False) # type: ignore

    SoftSegmentColor: FloatVectorProperty(
        name="Soft Segmentation Color",
        description="Soft Color",
        default=[0.8, 0.5, 0.38, 1.000000],  # [0.8, 0.46, 0.4, 1.0],[0.63, 0.37, 0.30, 1.0]
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    ) # type: ignore
    BoneSegmentColor: FloatVectorProperty(
        name="Bone Segmentation Color",
        description="Bone Color",
        default=[0.44, 0.4, 0.5, 1.0],  # (0.8, 0.46, 0.4, 1.0),
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    ) # type: ignore
    TeethSegmentColor: FloatVectorProperty(
        name="Teeth Segmentation Color",
        description="Teeth Color",
        default=[0.55, 0.645, 0.67, 1.000000],  # (0.8, 0.46, 0.4, 1.0),
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    ) # type: ignore

    #######################

    CT_Loaded: BoolProperty(description="CT loaded ", default=False) # type: ignore
    CT_Rendered: BoolProperty(description="CT Rendered ", default=False) # type: ignore
    sceneUpdate: BoolProperty(description="scene update ", default=True) # type: ignore
    AlignModalState: BoolProperty(description="Align Modal state ", default=False) # type: ignore

    #######################
    ActiveOperator: StringProperty(
        name="Active Operator",
        default="None",
        description="Active_Operator",
    ) # type: ignore
    #######################
    # Guide Components :

    TeethLibList = ["Christian Brenes Teeth Library"]
    items = []
    for i in range(len(TeethLibList)):
        item = (str(TeethLibList[i]), str(TeethLibList[i]), str(""), int(i))
        items.append(item)

    TeethLibrary: EnumProperty(
        items=items,
        description="Teeth Library",
        default="Christian Brenes Teeth Library",
    ) # type: ignore

    ImplantLibList = ["NEOBIOTECH_IS_II_ACTIVE"]
    items = []
    for i in range(len(ImplantLibList)):
        item = (str(ImplantLibList[i]), str(ImplantLibList[i]), str(""), int(i))
        items.append(item)

    ImplantLibrary: EnumProperty(
        items=items,
        description="Implant Library",
        default="NEOBIOTECH_IS_II_ACTIVE",
    ) # type: ignore
    #######################
    SleeveDiameter: FloatProperty(
        name="Sleeve Diameter",
        description="Sleeve Diameter",
        default=5.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    ) # type: ignore
    #######################
    SleeveHeight: FloatProperty(
        name="Sleeve Height",
        description="Sleeve Height",
        default=5.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    ) # type: ignore
    #######################
    HoleDiameter: FloatProperty(
        name="Hole Diameter",
        description="Hole Diameter",
        default=2.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    ) # type: ignore
    #######################
    HoleOffset: FloatProperty(
        name="Hole Offset",
        description="Sleeve Offset",
        default=0.1,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    ) # type: ignore

    #########################################################################################
    # Mesh Tools Props :
    #########################################################################################

    # Decimate ratio prop :
    #######################
    no_material_prop: StringProperty(
        name="No Material",
        default="No Color",
        description="No active material found for active object",
    ) # type: ignore
    decimate_ratio: FloatProperty(
        description="Enter decimate ratio ", default=0.5, step=1, precision=2
    ) # type: ignore
    #########################################################################################

    CurveCutterNameProp: StringProperty(
        name="Cutter Name",
        default="",
        description="Current Cutter Object Name",
    ) # type: ignore

    #####################

    CuttingTargetNameProp: StringProperty(
        name="Cutting Target Name",
        default="",
        description="Current Cutting Target Object Name",
    ) # type: ignore

    #####################
    items = [
        
        "Path Split",
        "Ribbon Split",
        "Ribbon Cutter",
        "Frame Cutter",
        # "Curve Cutter 3",
        
        # "Paint Cutter",
        # "Path Cutter",
        
    ]

    Cutting_Tools_Types_Prop: EnumProperty(
        items=set_enum_items(items), description="Select a cutting tool", default="Path Split"
    ) # type: ignore
    #####################
    CurveCutCloseModeList = ["Open Curve", "Close Curve"]
    CurveCutCloseMode: EnumProperty(items=set_enum_items(CurveCutCloseModeList), description="", default="Close Curve") # type: ignore
    #####################
    cutting_mode_list = ["Cut inner", "Keep inner"]
    cutting_mode: EnumProperty(items=set_enum_items(cutting_mode_list), description="", default="Cut inner") # type: ignore
    #####################
    TubeWidth: FloatProperty(description="Tube Width ", default=2, step=1, precision=2) # type: ignore
    ######################
    TubeCloseModeList = ["Open Tube", "Close Tube"]
    TubeCloseMode: EnumProperty(items=set_enum_items(TubeCloseModeList), description="", default="Close Tube") # type: ignore

    BaseHeight: FloatProperty(
        description="Base Height ", default=10, step=1, precision=2
    ) # type: ignore
    SurveyInfo: StringProperty(
        name="Models Survey Local Z",
        default="{}",
        description="Models Survey Local Z",
    ) # type: ignore

    #############################################################################################
    # ODENT Align Properties :
    #############################################################################################
    IcpVidDict: StringProperty(
        name="IcpVidDict",
        default="None",
        description="ICP Vertices Pairs str(Dict)",
    ) # type: ignore

    #######################
    AlignModalState: BoolProperty(description="Align Modal state ", default=False) # type: ignore
    text : StringProperty( name = "3D text", default = "Odent", update=update_text) # type: ignore
    slices_brightness: FloatProperty(
        description="Slices Brightness ", default=0.0, step=1, precision=3, update=brightness_update
    ) # type: ignore
    slices_contrast: FloatProperty(
        description="Slices Contrast ", default=0.2, step=1, precision=3, update=contrast_update
    ) # type: ignore

    # axial_slice_flip_vertical : BoolProperty(name="Axial Flip Vertical", default=True, description="Axial Flip Vertical")
    # coronal_slice_flip_vertical : BoolProperty(name="Coronal Flip Vertical", default=True, description="Coronal Flip Vertical")
    # sagital_slice_flip_vertical : BoolProperty(name="Sagital Flip Vertical", default=True, description="Sagital Flip Vertical")
    # axial_slice_flip_horizontal : BoolProperty(name="Axial Flip Horizontal", default=False, description="Axial Flip Horizontal")
    # coronal_slice_flip_horizontal : BoolProperty(name="Coronal Flip Horizontal", default=False, description="Coronal Flip Horizontal")
    # sagital_slice_flip_horizontal : BoolProperty(name="Sagital Flip Horizontal", default=False, description="Sagital Flip Horizontal")
#################################################################################################
# Registration :
#################################################################################################

classes = [
    ODENT_Props,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.ODENT_Props = bpy.props.PointerProperty(type=ODENT_Props)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.ODENT_Props


