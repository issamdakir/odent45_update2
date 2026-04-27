
bl_info = {
    "name": "Odent45",  
    "author": "Dr. Essaid Issam Dakir DMD\n,Dr. Ilya Fomenco DMD\n, Dr. Krasouski Dmitry DMD",
    "version": (4, 5, 0),
    "blender": (4, 5, 0),  
    "location": "3D View -> UI SIDE PANEL ",
    "description": "3D Digital Dentistry",  
    "warning": "",
    "doc_url": "",
    "category": "Dental",  
}
##############################################################################
# common imports:
##############################################################################
# Detect whether this addon has been loaded before
if "utils" in locals():
    utils = importlib.reload(utils)
else :
    import bpy # type: ignore
    import bpy.utils.previews # type: ignore
    import importlib
    import os
    from . import utils

##############################################################################
#get needed utils constants and functions :
##############################################################################

odent_cts = utils.OdentConstants
addon_name = odent_cts.ADDON_NAME
addon_dir_path = odent_cts.ADDON_DIR
odent_version_date = odent_cts.ADDON_VER_DATE

odent_log = utils.odent_log
resources_path = odent_cts.RESOURCES
odent_modules_path = odent_cts.ODENT_MODULES_PATH
odent_new_modules = os.path.join(resources_path, "odent_modules")
required_3rdparty_modules = odent_cts.REQ_DICT
missing_modules_info_txt_list = [
        f"Some {addon_name} 3rd party python modules need to be installed online!",
        "check support link for help.",
    ]

##############################################################################
#Console : print addon name and version (version is date based year/month/day)
##############################################################################

utils.odent_log([f"{addon_name} version : {odent_version_date}"])

##############################################################################
#ensure odent_modules_path is in sys.modules
##############################################################################

utils.set_modules_path() 
utils.OdentConstants.ODENT_LIB_IS_OK = utils.check_odent_library()

##############################################################################
#in case we need to update the 3rd party modules like vtk, SimpleITK, scipy...
#the pre_update action will store new modules under resources folder
#utils.replace_dir will replace old odent modules folder by the new one 
# before importing 3rd party modules (ODent_Operators, ODent_Utils)./
##############################################################################

utils.replace_dir(odent_new_modules, odent_modules_path)

##############################################################################
#check if 3rd party modules are loadable : 
##############################################################################

missing_modules, imports = utils.import_required_modules(required_3rdparty_modules)


##############################################################################
#common classes :
##############################################################################

class ODENT_PT_MissingModulesPanel(bpy.types.Panel):
    """ Missing 3rd party modules panel"""

    bl_idname = "ODENT_PT_MissingModulesPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI" 
    bl_category = addon_name
    bl_label = ""
    bl_options = {"HIDE_HEADER"}

    def draw(self, context):
        odent_logo_id = 0
        odent_icons = odent_cts.ODENT_ICONS
        if odent_icons :
            odent_logo = odent_icons.get("odent_logo")
            if odent_logo :
                odent_logo_id = odent_logo.icon_id
                
        layout = self.layout
        box = layout.box()
        b = box.box()
        g = b.grid_flow(columns=2, align=True)
        scale_y = 1.4
        
        g.scale_y = scale_y
        if odent_logo_id:
            g.template_icon(icon_value=odent_logo_id, scale=scale_y)
        g.operator("odent.welcome_dialog", text=f"{addon_name} - ver. {int(odent_version_date)}")
        
        b = box.box()
        for txt_line in missing_modules_info_txt_list :
            r = b.row()
            r.alert = True
            r.label(text=txt_line)
        # box = layout.box()
        # r = box.row()
        # r.operator(utils.ODENT_OT_OdentModulesPipInstaller.bl_idname)
        # r = box.row()
        # r.operator(utils.ODENT_OT_SupportTelegram.bl_idname)
        
class OdentAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    show_odent_welcome: bpy.props.BoolProperty(
        name="Show ODent welcome popup on startup",
        default=True,
    )
    def draw(self, context):
        global missing_modules
        layout = self.layout
        
        odent_icons = odent_cts.ODENT_ICONS
        odent_logo_id = 0
        telegram_logo_id = 0
        
        if odent_icons :
            odent_logo = odent_icons.get("odent_logo")
            odent_logo_id = odent_logo.icon_id if odent_logo else 0
            telegram_logo = odent_icons.get("telegram_logo")
            telegram_logo_id = telegram_logo.icon_id if telegram_logo else 0
                
        if odent_logo_id :     
            r = layout.row()
            r.alignment = 'CENTER'
            r.template_icon(icon_value=odent_logo.icon_id, scale=15)
        
        
        
        for txt_line, icon in [
            (f"Welcome to {addon_name} - ver. {int(odent_version_date)}", "NONE"),
            ("The all-in-one 3D Digital Dentistry addon for Blender", "NONE"),
            ("for assistance and updates", "NONE")
        ] :
            r = layout.row()
            r.alignment = 'CENTER'
            r.scale_y = 0.5
            r.label(text=txt_line, icon=icon)
        
        r = layout.row()
        r.scale_y = 1.5
        r.alert = True
        r.alignment = 'CENTER'
        r.operator(
            "wm.url_open", 
            text="Join the Telegram community",
            icon_value=telegram_logo_id).url = odent_cts.TELEGRAM_LINK
        
        
        layout.separator()
        
        if missing_modules :
            b = layout.box()
            for txt_line in missing_modules_info_txt_list :
                r = b.row()
                r.alert = True
                r.label(text=txt_line)
            
            r = layout.row()
            r.operator(utils.ODENT_OT_OdentModulesPipInstaller.bl_idname)
        
        r = layout.row()
        r.alignment = 'LEFT'
        r.label(text="Odent preferences :")
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_AddOdentLibrary.bl_idname, text="Install Odent Library", icon="SETTINGS")
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_AddAppTemplate.bl_idname, text="Odent as template",icon="SETTINGS")
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_SetConfig.bl_idname, text="Odent as default", icon="SETTINGS")
        # Create Desktop Icon button
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_create_desktop_shortcut.bl_idname, text="Create ODENT Desktop Icon", icon='DESKTOP')

        r = layout.row()
        r.prop(self, "show_odent_welcome")
        
class ODENT_OT_welcome_dialog(bpy.types.Operator):
    bl_idname = "odent.welcome_dialog"
    bl_label = f"{odent_cts.ADDON_NAME} - ver. {int(odent_cts.ADDON_VER_DATE)}"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        # utils.browse(odent_cts.TELEGRAM_LINK)
        return {'FINISHED'}

    def invoke(self, context, event):
        
        # return context.window_manager.invoke_props_dialog(self, width=600)
        # return context.window_manager.invoke_popup(self, width=1280)
        return context.window_manager.invoke_props_dialog(self, width=800)
    def draw(self, context):
        global missing_modules
        
        layout = self.layout
        remote_version = utils.update_is_availible()
        prefs = context.preferences.addons[__name__].preferences
        
        odent_icons = odent_cts.ODENT_ICONS
        odent_logo_id = 0
        telegram_logo_id = 0
        
        if odent_icons :
            odent_logo = odent_icons.get("odent_logo")
            odent_logo_id = odent_logo.icon_id if odent_logo else 0
            telegram_logo = odent_icons.get("telegram_logo")
            telegram_logo_id = telegram_logo.icon_id if telegram_logo else 0
                
        if odent_logo_id :     
            r = layout.row()
            r.alignment = 'CENTER'
            r.template_icon(icon_value=odent_logo.icon_id, scale=15)
        
        
        
        for txt_line, icon in [
            (f"Welcome to {addon_name} - ver. {int(odent_version_date)}", "NONE"),
            ("The all-in-one 3D Digital Dentistry addon for Blender", "NONE"),
            ("for assistance and updates", "NONE")
        ] :
            r = layout.row()
            r.alignment = 'CENTER'
            r.scale_y = 0.5
            r.label(text=txt_line, icon=icon)
        
        r = layout.row()
        r.scale_y = 1.5
        r.alert = True
        r.alignment = 'CENTER'
        r.operator(
            "wm.url_open", 
            text="Join the Telegram community",
            icon_value=telegram_logo_id).url = odent_cts.TELEGRAM_LINK
        
        
        layout.separator()
        box = layout.box()
        box.alignment = 'EXPAND'
        if missing_modules :
            b = box.box()
            b.alert = True
            for txt_line in missing_modules_info_txt_list :
                b.label(text=txt_line)
            r = box.row()
            r.alignment = 'EXPAND'
            r.scale_y = 2
            r.operator(utils.ODENT_OT_OdentModulesPipInstaller.bl_idname)
        
        layout.separator()
        r = layout.row()
        r.alignment = 'LEFT'
        r.label(text="Odent preferences :")
        
        r = layout.row()
        r.alignment = 'EXPAND'
        
        if remote_version :
            r.alert = True
            r.label(text=f"ODent update is availible :{remote_version} ")
            r = layout.row()
            r.alignment = 'EXPAND'
            r.alert = True
            r.operator("wm.odent_checkupdate", text="Update", icon="FILE_REFRESH")
        
        
        # r.operator(utils.ODENT_OT_checkUpdate.bl_idname, text="Check for addon updates", icon="SETTINGS")
        
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_AddOdentLibrary.bl_idname, text="Install Odent Library", icon="SETTINGS")
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_AddAppTemplate.bl_idname, text="Odent as template",icon="SETTINGS")
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_SetConfig.bl_idname, text="Odent as default", icon="SETTINGS")
        # Create Desktop Icon button
        r = layout.row()
        r.alignment = 'EXPAND'
        r.operator(utils.ODENT_OT_create_desktop_shortcut.bl_idname, text="Create ODENT Desktop Icon", icon='DESKTOP')

        r = layout.row()
        r.prop(prefs, "show_odent_welcome")


################### timers ################################
def run_popup_timer():
    """Timer callback to show popup after Blender starts"""
    try:
        # Use a more reliable way to get addon preferences
        addon_name = __name__.split('.')[0]  # Get main addon module name
        addon_prefs = bpy.context.preferences.addons.get(addon_name)
        
        if addon_prefs and hasattr(addon_prefs, 'preferences') and addon_prefs.preferences:
            if hasattr(addon_prefs.preferences, 'show_odent_welcome') and addon_prefs.preferences.show_odent_welcome:
                bpy.ops.odent.welcome_dialog('INVOKE_DEFAULT')
            
    except AttributeError as e:
        odent_log(f"Attribute error in run_popup_timer: {e}")
    except Exception as e:
        odent_log(f"Error in run_popup_timer: {e}")
        
    return None  # Stop the timer after showing the popup
def odent_slices_update_timer():
    tick = utils.OdentConstants.SLICES_UPDATE_TIMER_PERIOD
        
    if ODENT_Operators.FORCE_SLICES_UPDATE :
        _success, ODENT_Operators.SLICES_POINTER, message = ODENT_Operators.check_context()
        if not _success :
            # if  message :
            #     utils.odent_log(message)
            return tick
        ODENT_Operators.FORCE_SLICES_UPDATE = False
        
        ODENT_Operators.SLICES_DIRTY = True
        ODENT_Operators.SLICES_UPDATING = False
            
        
    if not ODENT_Operators.SLICES_DIRTY :
        return tick
    
    if ODENT_Operators.SLICES_UPDATING :
        return tick # seconds, time interval for the timer to check for new slices and update the point cloud if needed

    ODENT_Operators.SLICES_UPDATING = True
    
    try :
        success, message = ODENT_Operators._update_slices(ODENT_Operators.SLICES_POINTER)
        if success :
            ODENT_Operators.SLICES_DIRTY = False
            ODENT_Operators.SLICES_UPDATING = False
        # if message :
        #     utils.odent_log(message)
    except Exception as e :
        # utils.odent_log([f"Error in update slices : {e}"])
        pass
        
    return tick
    

# def show_hide_all_parts_timer():
#     tick = utils.OdentConstants.SLICES_UPDATE_TIMER_PERIOD
#     try :
#         props = bpy.context.scene.ODENT_Props
#         obj = bpy.context.object
#         if obj and obj.get(odent_cts.ODENT_TYPE_TAG) == odent_cts.ODENT_IMPLANT_TYPE:
#             mod = obj.modifiers.get(odent_cts.ODENT_IMPLANT_MODIFIER_NAME)
#             if mod :
#                 socket_num_list = [16, 15, 22, 29, 40, 36]
#                 if not all(mod[f"Socket_{socket_num}"] for socket_num in socket_num_list) :
#                     props.show_all_parts = False
#                 elif any(mod[f"Socket_{socket_num}"] for socket_num in socket_num_list) :
#                     props.hide_all_parts = False
#     except Exception as e:
#        odent_log([f"show_all_parts_timer error : {e}"])
        
#     return tick

classes = [
        OdentAddonPreferences,
        ODENT_OT_welcome_dialog
        
]
modules = [
        utils,
    ]

#########################################
# Case 1 > required modules are missing :
#########################################

if missing_modules :
    """display  missing addon modules ui"""
    
    classes.append(ODENT_PT_MissingModulesPanel)
    
    def register():
        utils.register_odent_icons()
        # pcoll = bpy.utils.previews.new()
        # odent_logo_path = odent_cts.ODENT_LOGO_PATH 
        # if os.path.exists(odent_logo_path):
        #     pcoll.load("odent_logo", odent_logo_path, 'IMAGE')
        # odent_cts.PREVIEW_COLLECTIONS["main"] = pcoll
        
        for module in modules:
            module.register()
        for cl in classes:
            bpy.utils.register_class(cl)
            
        if not bpy.app.timers.is_registered(run_popup_timer):
            bpy.app.timers.register(run_popup_timer, first_interval=0.1)
            
    def unregister():
        utils.unregister_odent_icons()
        if bpy.app.timers.is_registered(run_popup_timer):
            bpy.app.timers.unregister(run_popup_timer)
            
        for cl in reversed(classes):
            bpy.utils.unregister_class(cl)
        for module in reversed(modules):
            module.unregister()
        
        
    
    
 
###########################################
# Case 2 > required modules are installed :
###########################################
else :
    """display  full addon ui"""

    if "props" in locals():
        props = importlib.reload(props)
    else :
        from . import props
    if "ui" in locals():
        ui = importlib.reload(ui)
    else :
        from . import ui
    if "ODENT_Operators" in locals():
        ODENT_Operators = importlib.reload(ODENT_Operators)
    else :
        from .Operators import ODENT_Operators
    if "looptools" in locals():
        looptools = importlib.reload(looptools)
    else :
        from .Operators import looptools
    if "ODENT_Utils" in locals():
        ODENT_Utils = importlib.reload(ODENT_Utils)
    else :
        from .Operators import ODENT_Utils
    
    
    modules.extend([
            props,
            ui,
            ODENT_Operators,
            looptools
        ])
    classes.extend([

    ])
    
    timers = [
        (odent_slices_update_timer, True, 0.1),
        (run_popup_timer, False, 0.1)
        # (show_hide_all_parts_timer, True, 0.1)
    ]
    

##############################################################################
#addon registration :
##############################################################################

    def register():
        #load custom icons
        utils.register_odent_icons()
        
        #register addon modules and classes
        for module in modules:
            module.register()
        for cl in classes:
            bpy.utils.register_class(cl)
        
        #register timers
        for t, p, f_interval in timers:
            if not bpy.app.timers.is_registered(t):
                bpy.app.timers.register(t, persistent=p, first_interval=f_interval)
        
        ODENT_Operators.force_update_slices()
        
        
        
    def unregister():
        #remove custom icons
        utils.unregister_odent_icons()
        
        #unregister timers
        for t, p, f_interval in timers:
            if bpy.app.timers.is_registered(t):
                bpy.app.timers.unregister(t)
        
        #unregister addon modules and classes   
        for cl in reversed(classes):
            bpy.utils.unregister_class(cl)
        for module in reversed(modules):
            module.unregister()
        
        
        
        

if __name__ == "__main__":
    register()
