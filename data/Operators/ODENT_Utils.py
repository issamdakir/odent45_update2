import os
import socket
import sys
import shutil
import tempfile
import zipfile
from math import acos, ceil, degrees, pi, radians
from os.path import abspath, exists, join, isdir, split
from time import perf_counter as Tcounter
from time import sleep
from os import stat
from glob import glob
import bmesh # type: ignore
# Blender Imports :
import bpy # type: ignore
import cv2 # type: ignore
import mathutils # type: ignore
import numpy as np # type: ignore
import SimpleITK as sitk # type: ignore
# import itk
import vtk # type: ignore
from vtk import vtkCommand # type: ignore
from vtk.util import numpy_support # type: ignore
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk # type: ignore
from bpy.app.handlers import persistent # type: ignore
from bpy_extras import view3d_utils # type: ignore
from mathutils import Euler, Matrix, Vector, kdtree, bvhtree, geometry as Geo # type: ignore
from numpy.linalg import svd # type: ignore
import webbrowser
from threading import Thread


from ..utils import (
    OdentConstants,
    odent_log,
    AbsPath,
    RelPath,
    HuTo255,
    activate_obj,
    get_socket_geonodes,
    set_socket_value_geonodes,
)
# Global Variables :
ProgEvent = vtkCommand.ProgressEvent
_PREFFIX = None


class Segmentator:
    def __init__(self, 
                 sitk_image: sitk.Image, 
                 threshold: float = 600.0,
                 viz_object=None):
        self.sitk_image = sitk_image
        self.vtk_image = self._sitk_to_vtk(sitk_image)
        self.viz_object = viz_object
        self.threshold = threshold

    def _sitk_to_vtk(self, sitk_image: sitk.Image) -> vtk.vtkImageData:
        """Convert SimpleITK image to vtkImageData (ignoring direction)."""
        array = sitk.GetArrayFromImage(sitk_image)  # z, y, x
        depth, height, width = array.shape

        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetNumberOfTuples(array.size)
        vtk_array.SetVoidArray(array.ravel(order="C"), array.size, 1)

        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(width, height, depth)
        vtk_img.SetSpacing(sitk_image.GetSpacing())
        vtk_img.SetOrigin(sitk_image.GetOrigin())
        vtk_img.GetPointData().SetScalars(vtk_array)

        return vtk_img

    def _get_direction_transform(self) -> vtk.vtkTransform:
        """Convert SITK direction to vtkTransform (to apply later)."""
        direction = np.array(self.sitk_image.GetDirection()).reshape(3, 3)
        origin = np.array(self.sitk_image.GetOrigin())

        mat = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                mat.SetElement(i, j, direction[i, j])
            mat.SetElement(i, 3, origin[i])

        mat.SetElement(3, 0, 0)
        mat.SetElement(3, 1, 0)
        mat.SetElement(3, 2, 0)
        mat.SetElement(3, 3, 1)

        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)
        return transform

    def segment(self, threshold: float = 100.0) -> vtk.vtkPolyData:
        """Run marching cubes with SITK coordinates preserved."""
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(self.vtk_image)
        mc.SetValue(0, threshold)
        mc.Update()

        # Apply SITK direction/origin as a transform
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(mc.GetOutputPort())
        transform_filter.SetTransform(self._get_direction_transform())
        transform_filter.Update()

        return transform_filter.GetOutput()



class Segmentator_Fast:

    def __init__(self, sitk_image: sitk.Image, threshold: float = 600.0, viz_object=None, segmented_mesh=None):

        self.sitk_image = sitk_image
        self.threshold = HuTo255(threshold, OdentConstants.WMIN, OdentConstants.WMAX)
        self.threshold = max(1, min(254, self.threshold))
        print(f"Segmentator initialized with threshold (HU) = {threshold}, converted to {self.threshold} in 0-255 scale.")
        
        self.viz_object = viz_object
        self.segmented_mesh = segmented_mesh

        # Convert once
        self.vtk_image = self.sitk_to_vtk_image(sitk_image)

        # Build extraction pipeline once (FAST)
        self.extractor = vtk.vtkFlyingEdges3D()
        self.extractor.ComputeNormalsOn()
        self.extractor.SetInputData(self.vtk_image)
        self.extractor.SetValue(0, self.threshold)
        self.extractor.Update()  # Initial extraction to build the pipeline
        
        # self.extractor = vtk.vtkMarchingCubes()
        # self.extractor.ComputeNormalsOn()
        # self.extractor.SetValue(0, self.threshold)
        # self.extractor.SetInputData(self.vtk_image)
        # self.extractor.Update()

        
        self.debug = OdentConstants.DEBUG
        

    def sitk_to_vtk_image(self,sitk_image):
        """Convert sitk image to a VTK image"""
    
        sitk_array = sitk.GetArrayFromImage(sitk_image)  # .astype(np.uint8)
        spacing = sitk_image.GetSpacing()
        size = sitk_image.GetSize()
        origin = np.array(sitk_image.GetOrigin())
        direction = np.array(sitk_image.GetDirection()).reshape(3,3)
        vtk_direction = vtk.vtkMatrix3x3()
        for i in range(3):
            for j in range(3):
                vtk_direction.SetElement(i, j, direction[i, j])
        

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(size)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(origin)
        vtk_image.SetDirectionMatrix(vtk_direction)
        vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

        vtk_array = numpy_to_vtk(
            sitk_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_INT
        )
        vtk_array.SetNumberOfComponents(1)
        vtk_image.GetPointData().SetScalars(vtk_array)
        vtk_image.Modified()
        return vtk_image


    def segment(self, threshold: float = None) -> vtk.vtkPolyData:
        """Extract surface mesh using FlyingEdges (interactive safe)."""

        if threshold is not None:
            self.threshold = HuTo255(threshold, OdentConstants.WMIN, OdentConstants.WMAX)
            self.threshold = max(1, min(254, self.threshold))
            self.extractor.SetValue(0, self.threshold)
            
            self.extractor.Modified()
            self.extractor.Update()

        return self.extractor.GetOutput()
    
    def _vtk_polydata_to_numpy(self, polydata):
        """Convert vtkPolyData to numpy verts/faces."""
        error_message = []
        vtk_points = polydata.GetPoints()
        if not vtk_points :
            error_message = ["can't extract geometry from image !"]
            return error_message, vertices, faces
        
        vertices = vtk_to_numpy(vtk_points.GetData()).astype(np.float32)

        # Get triangle faces (all faces from marching cubes are triangles)
        vtk_polys = polydata.GetPolys()
        if vtk_polys:
            faces = vtk_to_numpy(vtk_polys.GetData()).reshape(-1, 4)[:, 1:].astype(np.int32)  # drop the leading '3'
    

        # points = numpy_support.vtk_to_numpy(
        #     polydata.GetPoints().GetData()
        # ).astype(np.float32)

        # # VTK stores faces like: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        # faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(np.int32)  # drop the leading '3'

        return vertices, faces

    
    def generate_segmented_mesh(self, threshold=None, object_name="SegmentedMesh"):
        """
        Generate or update the segmented mesh in Blender from the current threshold.
        Uses the in-memory VTK pipeline.
        
        If `self.segmented_mesh` is None, creates a new Blender object.
        Otherwise updates the existing mesh.
        """
        
        polydata = self.segment(threshold)
        # self.vtk_view3d(polydata)  # Optional: visualize in VTK before creating Blender mesh
    
                
        vertices, faces = self._vtk_polydata_to_numpy(polydata)
        if self.debug:
            print(f"Extracted {len(vertices)} vertices and {len(faces)} faces from VTK.")
        
        num_verts = vertices.shape[0]
        num_faces = faces.shape[0]
        verts_per_face = faces.shape[1]
        if self.segmented_mesh:
            if self.debug :
                print("Update existing mesh object")
            mesh = self.segmented_mesh.data

            # Clear old geometry
            mesh.clear_geometry()

        else:
            if self.debug :
                print("Create new mesh object")
            mesh = bpy.data.meshes.new(object_name)
            self.segmented_mesh = bpy.data.objects.new(object_name, mesh)
            bpy.context.collection.objects.link(self.segmented_mesh)
            with bpy.context.temp_override(active_object=self.segmented_mesh, selected_objects=[self.segmented_mesh], selected_editable_objects=[self.segmented_mesh]):
                bpy.ops.wm.odent_lock_objects()
            MoveToCollection(self.segmented_mesh, OdentConstants.SEGMENTS_COLLECTION_NAME)
            
        mesh.vertices.add(num_verts)
        mesh.vertices.foreach_set("co", vertices.ravel())

        mesh.loops.add(num_faces * verts_per_face)
        mesh.polygons.add(num_faces)
        mesh.loops.foreach_set("vertex_index", faces.ravel())
        mesh.polygons.foreach_set("loop_start", np.arange(0, num_faces * verts_per_face, verts_per_face))
        mesh.polygons.foreach_set("loop_total", np.full(num_faces, verts_per_face, dtype=np.int32))

        mesh.validate()
        mesh.update()

    

        return self.segmented_mesh

    def visualize_polydata(self,polydata: vtk.vtkPolyData):
        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()  # Use solid color

        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.6, 0.3)  # Orange

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)  # Dark grey

        # Render Window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)

        # Interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # Optional: trackball camera controls
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        # Start
        render_window.Render()
        interactor.Initialize()
        interactor.Start()

    def vtk_view3d(self, polydata):
        def vtk_viewer(polydata):
            self.visualize_polydata(polydata)
        t = Thread(target=vtk_viewer, args=(polydata,), daemon=True)
        t.start()
        t.join()
######################################################################
def get_locked_to_pointer_objects():
    objects = []
    for obj in bpy.context.scene.objects:
        if not obj.get(OdentConstants.ODENT_TYPE_TAG) in [
            OdentConstants.SLICES_POINTER_TYPE,
            OdentConstants.SLICE_PLANE_TYPE,]:
            constraints = [c for c in obj.constraints if c.type ==
            "CHILD_OF"and c.target is not None and c.target.get(OdentConstants.ODENT_TYPE_TAG) == OdentConstants.SLICES_POINTER_TYPE]
            if constraints:
                objects.append((obj, constraints))
    return objects
def get_fly_to_objects():
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.get(OdentConstants.ODENT_TYPE_TAG) in [
            OdentConstants.ODENT_IMPLANT_TYPE,
            OdentConstants.FIXING_SLEEVE_TYPE
        ]
    ]

def remove_pointer_lock():
    locked_objects = get_locked_to_pointer_objects()
    for obj, constraints in locked_objects:
        odent_log(f"Unlocking {obj.name} from pointer lock.")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        bpy.ops.wm.odent_unlock_object_from_pointer()
       


def get_direction_origin_from_visobj(visobj):
    M = visobj.matrix_world        # 4x4
    # Get pure rotation (remove scaling)
    rot3 = M.to_3x3().normalized()
    # Convert to numpy
    rot_np = np.array(rot3)
    # Flatten to 1D
    sitk_direction = np.array(rot3).flatten().tolist()
    
    origin = np.array(M.translation).tolist()
    
    vtk_direction = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            vtk_direction.SetElement(i, j, rot_np[i, j])
    
    return vtk_direction, sitk_direction, origin
def sitk_to_vtk_image(sitk_image, visobj=None):
    """Convert sitk image to a VTK image"""
   
    sitk_array = sitk.GetArrayFromImage(sitk_image)  # .astype(np.uint8)
    spacing = sitk_image.GetSpacing()
    size = sitk_image.GetSize()
    origin = np.array(sitk_image.GetOrigin())
    direction = np.array(sitk_image.GetDirection()).reshape(3,3)
    vtk_direction = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            vtk_direction.SetElement(i, j, direction[i, j])
    

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetDirectionMatrix(vtk_direction)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    vtk_array = numpy_to_vtk(
        sitk_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_INT
    )
    vtk_array.SetNumberOfComponents(1)
    vtk_image.GetPointData().SetScalars(vtk_array)
    vtk_image.Modified()
    return vtk_image

def vtk_marching_cubes(vtk_image, threshold_255):
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.ComputeNormalsOn()
    marching_cubes.SetValue(0, threshold_255)
    marching_cubes.SetInputData(vtk_image)
      # Example threshold
    marching_cubes.Update()
    return marching_cubes

def clear_bmesh(bm):
    # Delete all faces
    bmesh.ops.delete(bm, geom=bm.faces[:], context='FACES')
    # Delete all edges
    bmesh.ops.delete(bm, geom=bm.edges[:], context='EDGES')
    # Delete all vertices
    bmesh.ops.delete(bm, geom=bm.verts[:], context='VERTS')

def create_blender_mesh(name, vertices, faces):
    # Create Blender mesh and object
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Use BMesh
    bm = bmesh.new()
    bm_verts = [bm.verts.new(tuple(v)) for v in vertices]

    for face in faces:
        try:
            bm.faces.new([bm_verts[i] for i in face])
        except ValueError:
            pass

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    return obj
   
def polydata_to_blender_mesh(name="DicomMesh", marching_cubes=None,obj=None, transform=None):

    vertices, faces = extract_vertices_faces_from_marching_cubes(marching_cubes, transform)
    if obj:
        thresh = bpy.context.scene.ODENT_Props.ThresholdMin
        last = obj.name.split("#")[-1]
        obj.name = obj.name.replace(last, str(thresh))
        # Update the mesh
        bm = bmesh.new()
        bm.from_mesh(obj.data)

        # Clear existing geometry
        clear_bmesh(bm)

        # Add new geometry
        bm_verts = [bm.verts.new(v) for v in vertices]

        # Add faces to the BMesh
        for face in faces:
            try:
                bm.faces.new([bm_verts[i] for i in face])
            # except ValueError:
            #     print(f"Skipping duplicate face: {face}")
            except ValueError:
                pass
        # Ensure all vertices are linked properly
        bm.verts.ensure_lookup_table()
        # Ensure proper updates
        bm.faces.ensure_lookup_table()

        bm.normal_update()
        bm.to_mesh(obj.data)
        bm.free()
    else :
        # Create a new mesh
        obj = create_blender_mesh(name, vertices, faces)
    return obj

# def get_polydata_from_marching_cubes(marching_cubes, transform=None):
#     polydata = vtk.vtkPolyData()
#     polydata.DeepCopy(marching_cubes.GetOutput())
#     if transform is not None:
#         polydata = vtk_set_transform(polydata, transform)
#     return polydata

def vtk_set_transform(polydata, vtk_transform):
    """Transform a mesh using VTK's vtkTransformPolyData filter."""
    Transform = vtk.vtkTransform()
    Transform.SetMatrix(vtk_transform)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polydata)
    transformFilter.SetTransform(Transform)
    transformFilter.Update()
    polydata.DeepCopy(transformFilter.GetOutput())
    return polydata

# def extract_geometry(polydata):
#     # Extract vertices
#     vtk_points = polydata.GetPoints()
#     vertices = vtk_to_numpy(vtk_points.GetData()) if vtk_points else []

#     # Extract faces
#     vtk_polys = polydata.GetPolys()
#     faces = vtk_to_numpy(vtk_polys.GetData()).reshape(-1, 4)[:, 1:] if vtk_polys else []
#     return vertices, faces

def extract_vertices_faces_from_marching_cubes(marching_cubes, transform=None):
    error_message = []
    vertices, faces = None, None
    polydata = marching_cubes.GetOutput()
    if transform is not None:
        polydata = vtk_set_transform(polydata, transform)
    # Extract vertices
    vtk_points = polydata.GetPoints()
    if not vtk_points :
        error_message = ["can't extract geometry from image !"]
        return error_message, vertices, faces
    
    vertices = vtk_to_numpy(vtk_points.GetData()).astype(np.float32)

    # Get triangle faces (all faces from marching cubes are triangles)
    vtk_polys = polydata.GetPolys()
    if vtk_polys:
        faces = vtk_to_numpy(vtk_polys.GetData()).reshape(-1, 4)[:, 1:].astype(np.int32)  # drop the leading '3'
    
    return error_message, vertices, faces

def create_blender_mesh_from_marching_cubes_fast(
    name="DicomMesh", 
    marching_cubes=None,
    visobj=None, 
    transform=None):
    
    error_message = []
    obj = None
    if not marching_cubes:
        error_message = ["can't extract geometry from image !"]
        return error_message, obj
    error_message, vertices, faces = extract_vertices_faces_from_marching_cubes(marching_cubes,transform)
    if error_message:
        return error_message, obj

    num_verts = vertices.shape[0]
    num_faces = faces.shape[0]
    verts_per_face = faces.shape[1]

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(num_verts)
    mesh.vertices.foreach_set("co", vertices.ravel())

    mesh.loops.add(num_faces * verts_per_face)
    mesh.polygons.add(num_faces)
    mesh.loops.foreach_set("vertex_index", faces.ravel())
    mesh.polygons.foreach_set("loop_start", np.arange(0, num_faces * verts_per_face, verts_per_face))
    mesh.polygons.foreach_set("loop_total", np.full(num_faces, verts_per_face, dtype=np.int32))

    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    with bpy.context.temp_override(active_object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
        bpy.ops.wm.odent_lock_objects()
    MoveToCollection(obj, OdentConstants.SEGMENTS_COLLECTION_NAME)


    return error_message, obj

def get_sitk_image_center(sitk_image):
    size = sitk_image.GetSize()
    o_coordinate = sitk_image.TransformContinuousIndexToPhysicalPoint((0, 0, 0))
    max_coordinate = sitk_image.TransformContinuousIndexToPhysicalPoint(
        (size[0] - 1, size[1] - 1, size[2] - 1)
    )
    sitk_image_center_coordinate = (Vector(o_coordinate) + Vector(max_coordinate)) * 0.5
    return sitk_image_center_coordinate

def get_vtk_transform_from_vis(vis_obj):
    # Example: get matrix_world from the active object
    M_blender = vis_obj.matrix_world

    # Convert to numpy (row-major)
    M_np = np.array(M_blender)

    # Transpose to column-major for VTK
    M_vtk = M_np.transpose()

    # Create vtkTransform and set matrix
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, M_vtk[i, j])

    vtk_transform = vtk.vtkTransform()
    vtk_transform.SetMatrix(vtk_matrix)
    return vtk_transform

def get_vtk_transform_from_sitk(sitk_image):
    O = sitk_image.GetOrigin()
    D = sitk_image.GetDirection()
    rotation_matrix_4x4 = Matrix((
                                    (D[0], D[1], D[2], 0.0),
                                    (D[3], D[4], D[5], 0.0),
                                    (D[6], D[7], D[8], 0.0),
                                    (0.0, 0.0, 0.0, 1.0)))

    translation_matrix_4x4 = Matrix((
                                    (1.0, 0.0, 0.0, O[0]),
                                    (0.0, 1.0, 0.0, O[1]),
                                    (0.0, 0.0, 1.0, O[2]),
                                    (0.0, 0.0, 0.0, 1.0)))
    composed_4x4 = translation_matrix_4x4 @ rotation_matrix_4x4
    C = get_sitk_image_center(sitk_image) #sitk_image center coordinate

    blender_transform = Matrix(
        (
            (D[0], D[1], D[2], C[0]),
            (D[3], D[4], D[5], C[1]),
            (D[6], D[7], D[8], C[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    vtk_transform = (
            blender_transform.inverted() @ composed_4x4
        )

    # vtk_transform = list(np.array(vtk_transform).ravel())
    vtk_transform = list(np.array(composed_4x4).ravel())
    # print(f"vtk_transform : {vtk_transform}")
    return vtk_transform, blender_transform


def create_blender_image_from_np(name: str, np_array: np.ndarray):
    """
    Create a Blender image from a NumPy array with shape (height, width, 4).
    Values should be in [0, 255] (uint8) or [0.0, 1.0] (float32).
    
    :param name: Name of the new Blender image.
    :param np_array: NumPy array of shape (height, width, 4).
    :return: The created bpy.types.Image object.
    """
    if np_array.ndim != 3 or np_array.shape[2] != 4:
        print(f"Invalid shape: {np_array.shape}")
        raise ValueError(f"Invalid shape: {np_array.shape}\nNumPy array must have shape (height, width, 4)")
    # Convert to float32 if needed and normalize to [0.0, 1.0]
    if np_array.dtype == np.uint8:
        pixels = (np_array.astype(np.float32) / 255.0).reshape(-1)
    elif np_array.dtype == np.float32 or np_array.dtype == np.float64:
        pixels = np_array.astype(np.float32).reshape(-1)
    else:
        raise TypeError("Unsupported dtype: must be uint8 or float32/64")
    image = bpy.data.images.get(name)
    if image:
        bpy.data.images.remove(image)
    height, width, _ = np_array.shape
    image = bpy.data.images.new(name=name,width=width, height=height, alpha=True, float_buffer=True)
    # Create image in Blender    # image.pixels[:] = pixels.tolist()  # Blender expects a flat list
    image.pixels.foreach_set(pixels)
    image.pack()  # Optional: pack the image into the .blend file
    return image

def parse_dicom_series(self, dicom_directory):
    
    sorted_data_dict = {}
    error_msg = []
    series_ids = []

    self.reader = sitk.ImageSeriesReader()
    try :
        series_ids = self.reader.GetGDCMSeriesIDs(AbsPath(dicom_directory))
    except Exception as e:
        error_msg.append("Can not read DICOM series")
        odent_log(txt_list=["Error parsing dicom series :\n", str(e)])
        return error_msg, sorted_data_dict
    if not series_ids:
        error_msg.append("No DICOM series found")
        odent_log(txt_list=["No DICOM series found:\n", f"dicom directory : {dicom_directory}"])
        return error_msg, sorted_data_dict
    
    series_data = []
    for s_id in series_ids:
        
        try :
            _file_names = self.reader.GetGDCMSeriesFileNames(dicom_directory, s_id)
            count = len(_file_names)
            if count <= 2:
                continue
            self.reader.SetFileNames(_file_names)
            image = self.reader.Execute()
            
            sp_max = round(max(image.GetSpacing()[:]), 3)

            data_item = {
                "s_id": s_id,
                "count": count,
                "spacing": sp_max,
                # "files": _file_names,
                "enum_txt": f"{s_id} ({count} slices)",
            }
            series_data.append(data_item)
        except Exception as e:
            odent_log(txt_list=["Error parsing dicom series :\n", str(e)])
            continue
    _data_sorted = sorted(series_data, key=lambda x: x["count"], reverse=True)
    sorted_data_dict = {e["enum_txt"]: e for e in _data_sorted}

    # print(f"Dicom Series : {sorted_data_dict}")
    return error_msg, sorted_data_dict



def load_matrix_from_file(filename):
    try:
        matrix = np.loadtxt(filename)
        return Matrix(matrix)
    except IOError:
        print(f"Could not read file: {filename}")
        return None


def get_layerColl(colname):
    lc = bpy.context.view_layer.layer_collection.children.get(colname)
    return lc


def exclude_coll(_exclude=True, colname=""):
    coll = bpy.data.collections.get(colname)
    lc = get_layerColl(colname)
    if coll and lc:
        lc.exclude = _exclude
    return


def hide_collection(_hide=True, colname=""):
    coll = bpy.data.collections.get(colname)
    if coll :
        coll.hide_select = _hide
        coll.hide_viewport = _hide
        
        lc = get_layerColl(colname)
        if lc:
            lc.exclude = _hide
            lc.hide_viewport = _hide


def hide_object(_hide=True, obj=None):
    if obj:
        obj.hide_select = _hide
        obj.hide_viewport = _hide
        obj.hide_set(_hide)


def is_manifold(obj):
    if obj.type == "MESH":
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        is_manifold = all(edge.is_manifold for edge in bm.edges)
        bm.free()
        return is_manifold


def separate_geoimplants():

    def apply_implant_node(obj):
        mod = obj.modifiers.get(OdentConstants.ODENT_IMPLANT_MODIFIER_NAME)
        if mod is None:
            return
        ng = mod.node_group 


        cut_type = mod[get_socket_geonodes(ng,'Internal cut type')]
        ## cut_type 0 - Cylinder, 1 - Custom sleeve obj

        set_socket_value_geonodes(mod,'Show implant axe',False)
        set_socket_value_geonodes(mod,'Show text', False)
        set_socket_value_geonodes(mod,'Safe zone Show', False)
        if cut_type == 0:
            set_socket_value_geonodes(mod,'Show sleeve', True)
            set_socket_value_geonodes(mod,'Show internal cutter', True)
        else:
            set_socket_value_geonodes(mod,'Show sleeve', True),
            set_socket_value_geonodes(mod,'Show custom sleeve', True),
        mod.show_viewport = False
        mod.show_viewport = True
        bpy.ops.object.convert(target='MESH')


    def extract_vertex_group(obj, group_name, implant_name):
        vg = obj.vertex_groups.get(group_name)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        group_index = vg.index
        verts_keep = []

        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == group_index and g.weight > 0:
                    verts_keep.append(v.index)
                    break

        bm.verts.ensure_lookup_table()
        verts_delete = [v for v in bm.verts if v.index not in verts_keep]
        bmesh.ops.delete(bm, geom=verts_delete, context='VERTS')

        if group_name == "sleeve":
            new_mesh = bpy.data.meshes.new(f"_ADD_{implant_name}_{group_name}")
        else:
            new_mesh = bpy.data.meshes.new(f"{implant_name}_{group_name}")        
        bm.to_mesh(new_mesh)
        bm.free()

        new_obj = bpy.data.objects.new(new_mesh.name, new_mesh)
        new_obj.matrix_world = obj.matrix_world.copy()
        return new_obj
    
    
    impl_col = bpy.data.collections.get("Implant_Collection")
    if not impl_col:
        return    
    for implant in list(impl_col.objects):
        if implant.name.startswith('Odent_Implant'):
            temp_implant = implant.copy()
            temp_implant.data = implant.data.copy()
            bpy.context.collection.objects.link(temp_implant)
            activate_obj(temp_implant)
            apply_implant_node(temp_implant)
            cutter_obj = extract_vertex_group(temp_implant, "cutter", implant.name)
            MoveToCollection(cutter_obj, "CutColl")
            sleeve_obj = extract_vertex_group(temp_implant, "sleeve", implant.name)
            MoveToCollection(sleeve_obj, "AddColl")
            bpy.data.objects.remove(temp_implant, do_unlink=True)


def get_or_create_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def finalize_geonodes_make_dup_colls(context, guide_components, add_components):
    # obj = add_components[0]

    add_coll = get_or_create_collection('AddColl')
    cut_coll = get_or_create_collection('CutColl')
    for obj in guide_components:
        hide_object(False, obj)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        bpy.ops.object.duplicate_move()
        dup_obj = context.object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if dup_obj.constraints:
            for c in dup_obj.constraints:
                bpy.ops.constraint.apply(constraint=c.name)
        if dup_obj.type == "CURVE":
            bpy.ops.object.convert(target="MESH", keep_original=False)
            remesh = dup_obj.modifiers.new(name="Remesh", type="REMESH")

        if dup_obj.modifiers or dup_obj.type == "CURVE":
            bpy.ops.object.convert(target="MESH", keep_original=False)

        # check non manifold :
        if is_manifold(dup_obj) == False:
            remesh = dup_obj.modifiers.new(name="Remesh", type="REMESH")
            remesh.mode = "SHARP"
            remesh.octree_depth = 8
            bpy.ops.object.convert(target="MESH", keep_original=False)
        if obj in add_components:
            MoveToCollection(dup_obj, "AddColl")
        else:
            MoveToCollection(dup_obj, "CutColl")
    separate_geoimplants()
    return add_coll, cut_coll


def add_odent_libray():
    lib_archive_dir_path = join(OdentConstants.ODENT_LIBRARY_PATH, "lib_archive")
    if exists(lib_archive_dir_path):
        files = glob(join(lib_archive_dir_path, "*"))

        for f in files:
            if f.endswith(".zip"):
                with zipfile.ZipFile(f, "r") as zip_ref:
                    zip_ref.extractall(OdentConstants.ODENT_LIBRARY_PATH)
            else:
                shutil.move(f, OdentConstants.ODENT_LIBRARY_PATH)
        shutil.rmtree(lib_archive_dir_path)

    user_lib = bpy.context.preferences.filepaths.asset_libraries.get(OdentConstants.ODENT_LIB_NAME)

    if user_lib:
        user_lib.path = OdentConstants.ODENT_LIBRARY_PATH
    else:
        bpy.ops.preferences.asset_library_add(directory=OdentConstants.ODENT_LIBRARY_PATH)

    return


def close_asset_browser(context, area=None):

    if area:
        a = area
        # r = [r for r in a.regions if r.type == "WINDOW"][0]
        s = a.spaces.active
        with bpy.context.temp_override(area=a, space_data=s):
            bpy.ops.screen.area_close()
        return 1

    scr = context.screen
    areas_asset = []
    for a in scr.areas:
        if a.type == "FILE_BROWSER":
            if a.ui_type == "ASSETS":
                areas_asset.append(a)
    if areas_asset:
        for a in areas_asset:
            # r = [r for r in a.regions if r.type == "WINDOW"][0]
            s = a.spaces.active
            with bpy.context.temp_override(area=a, space_data=s):
                bpy.ops.screen.area_close()
            return 1
    return 0


def open_asset_browser():
    context = bpy.context

    scr = context.screen
    areas3d = []
    for a in scr.areas:
        if a.type == "FILE_BROWSER":
            if a.ui_type == "ASSETS":
                return a, a.spaces.active
        elif a.type == "VIEW_3D":
            areas3d.append(a)

    if areas3d:
        a3d = areas3d[0]
        r3d = [r for r in a3d.regions if r.type == "WINDOW"][0]
        s3d = a3d.spaces.active

        with bpy.context.temp_override(area=a3d, space_data=s3d, region=r3d):

            bpy.ops.screen.area_split(direction="VERTICAL", factor=1 / 3)
            scr.update_tag()

        a2 = [a for a in scr.areas if a.type == "VIEW_3D"][-1]
        a2.type = "FILE_BROWSER"
        a2.ui_type = "ASSETS"
        scr.update_tag()

        s2 = a2.spaces.active

        return a2, s2


def get_selected_odent_assets(area=None):
    result = {
        "success": 0,
        "message": "",
        "error": 0,
        "directory": None,
        "filename": None,
    }
    selected = bpy.context.selected_objects
    if not selected or not [
        o.get("odent_type") != "odent_implant" for o in selected
    ]:
        result["message"] = [
            f"Warning : Please select implants",
            "<ENTER> : retry  <ESC> : cancel.",
        ]
        result["error"] = 2
        return result

    space = area.spaces.active

    current_library_name = space.params.asset_library_ref
    if not current_library_name == OdentConstants.ODENT_LIB_NAME:
        result["message"] = [
            f"Warning : The selected asset is not part of {OdentConstants.ODENT_LIB_NAME}",
            "<ENTER> : retry  <ESC> : cancel.",
        ]
        result["error"] = 2
        return result

    asset_file = space.params.filename
    if not asset_file:
        result["message"] = [
            f"Warning : Please select asset from {OdentConstants.ODENT_LIB_NAME}",
            "<ENTER> : retry  <ESC> : cancel.",
        ]
        result["error"] = 2
        return result

    library_path_root = bpy.context.preferences.filepaths.asset_libraries.get(
        OdentConstants.ODENT_LIB_NAME
    ).path
    head, filename = split(asset_file)
    directory = join(library_path_root, head)
    result = {
        "success": 1,
        "message": "",
        "error": 0,
        "directory": directory,
        "filename": filename,
    }

    return result


def isConnected():
    try:
        sock = socket.create_connection(("www.google.com", 80))
        if sock is not None:
            sock.close
        return True
    except OSError:
        return False


def browse(url):
    success = 0
    try:
        webbrowser.open(url)
        success = 1
        return success
    except Exception as er:
        print(f"open telegram link error :\n{er}")
        return success


def raycast(context, event, obj):

    # define boundary conditions
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    mouseCoordinates = event.mouse_region_x, event.mouse_region_y
    # convert cursor location and view direction
    viewVector = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouseCoordinates)
    rayOrigin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouseCoordinates)
    rayTarget = rayOrigin + viewVector

    # convert to object space
    matrixInverted = obj.matrix_world.inverted()
    rayOriginObject = matrixInverted @ rayOrigin
    rayTargetObject = matrixInverted @ rayTarget
    rayVectorObject = rayTargetObject - rayOriginObject

    # raycast procedure
    success, hitLocation, _, _ = obj.ray_cast(rayOriginObject, rayVectorObject)

    return success


def click_is_in_view3d(context, event):
    scr = context.screen
    a = 0
    r = 0
    for a in scr.areas:
        if a.type == "VIEW_3D":
            break
    if not a:
        print("not found : area VIEW_3D")
        return None

    for r in a.regions:
        if r.type == "WINDOW":
            break
    if not r:
        print("not found : region WINDOW")
        return None

    mouseCoordinates = event.mouse_region_x, event.mouse_region_y
    regionCoordinates = r.x, r.y
    regionSize = r.width, r.height
    _is_valid = (
        regionCoordinates[0]
        <= mouseCoordinates[0]
        <= regionCoordinates[0] + regionSize[0]
    ) and (
        regionCoordinates[1]
        <= mouseCoordinates[1]
        <= regionCoordinates[1] + regionSize[1]
    )

    return _is_valid



def start_blender_session():
    os.system(f'"{bpy.app.binary_path}"')

def bmesh_copy_from_object(obj):
    """Returns bmesh copy of the mesh"""
    assert obj.type == "MESH"
    me = obj.data
    if obj.mode == "EDIT":
        bm_orig = bmesh.from_edit_mesh(me)
        bm = bm_orig.copy()
    else:
        bm = bmesh.new()
        bm.from_mesh(me)

    return bm


def select_non_manifold_verts(
    obj,
    use_wire=False,
    use_boundary=False,
    use_multi_face=False,
    use_non_contiguous=False,
    use_verts=False,
):
    """select non-manifold vertices"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_non_manifold(
        extend=False,
        use_wire=use_wire,
        use_boundary=use_boundary,
        use_multi_face=use_multi_face,
        use_non_contiguous=use_non_contiguous,
        use_verts=use_verts,
    )


def mesh_count(obj):
    bm = bmesh_copy_from_object(obj)
    results = len(bm.verts), len(bm.edges), len(bm.faces)
    bm.free()
    return results


def count_non_manifold_verts(obj):
    """return non manifold verts count"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    select_non_manifold_verts(obj, use_wire=True, use_boundary=True, use_verts=True)
    bpy.ops.object.mode_set(mode="OBJECT")

    bm = bmesh_copy_from_object(obj)
    return sum((1 for v in bm.verts if v.select))


def get_parts(obj):

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.object.mode_set(mode="OBJECT")

    face_ids = [f.index for f in obj.data.polygons]
    data = {}
    counter = 1
    while face_ids:
        # print(counter)
        idx = face_ids[0]
        obj.data.polygons[idx].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_linked()
        bpy.ops.object.mode_set(mode="OBJECT")
        selected = [f.index for f in obj.data.polygons if f.select]
        data.update({f"PART{counter}": {"COUNT": len(selected), "IDS": selected}})
        face_ids = [f.index for f in obj.data.polygons if not f.select]
        counter += 1
    n = len(data)
    return data, n


def delete_loose(obj):
    """delete loose vertices/edges/faces"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=True)
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")


def delete_interior_faces(obj):
    """delete interior faces"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_interior_faces()
    bpy.ops.object.mode_set(mode="OBJECT")
    selected = [f.index for f in obj.data.polygons if f.select]
    if selected:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.fill_holes(sides=4)
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode="OBJECT")


def merge_verts(obj, threshold=0.001, all=False):
    """merge vertices with respect to a distance threshold"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    select_non_manifold_verts(obj, use_wire=True, use_boundary=True, use_verts=True)
    if all:
        bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode="OBJECT")


def degenerate_dissolve(obj, threshold):
    """dissolve zero area faces and zero length edges"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_degenerate(threshold=threshold)
    bpy.ops.object.mode_set(mode="OBJECT")


def fill_holes(obj, _all=True, hole_size=4):
    """fill holes"""
    bpy.ops.object.mode_set(mode="EDIT")
    if _all:
        bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.fill_holes(hole_size)
    bpy.ops.object.mode_set(mode="OBJECT")


class clean_Mesh:

    def __init__(self, remesh):
        self.thresh_1 = 0.0001
        self.thresh_2 = 0.05
        self.hole_size = 4
        self.remesh = remesh
        self.info_dict = {}
        self.target = bpy.context.object
        self.current_count = None
        self.step = 0

        ################ Prepare scene : ################################
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        verts_count, edges_count, polygons_count = mesh_count(self.target)
        self.current_count = count_non_manifold_verts()
        message = [
            "\t- Starting state :",
            f"\t\t- Total number of Verts : {verts_count}",
            f"\t\t- Total number of Edges : {edges_count}",
            f"\t\t- Total number of Polygons : {polygons_count}",
            f"\t\t- Total number of non manifold Verts : {self.current_count}",
        ]
        self.log(message)

        self.info_dict[self.file_name] = {
            "Starting State": {
                "Total number of Verts": verts_count,
                "Total number of Edges": edges_count,
                "Total number of Polygons": polygons_count,
                "Total number of non manifold Verts": self.current_count,
            },
            "Mesh Processing": {},
            "Mesh Export": {},
        }

    def process(self):

        self.step += 1
        ########### Get Intersecting faces ###############

        message = [f"\n\t- STEP {self.step} : Checking Intersecting Faces : "]
        self.log(message)
        self.overlaping_pairs, self.overlaping_faces, n = self.find_intersections(
            self.target
        )
        message = [
            f"\t\t- Total number of intersecting faces : {n}",
        ]
        self.log(message)
        self.info_dict[self.target.name]["Mesh Processing"][
            "Checking Intersecting Faces"
        ] = f"{n}  Intersecting Faces found"

        ############################# Get Number of bodies ################################
        message = [f"\n\t- STEP {self.step} : Checking multiple Body parts : "]
        self.log(message)
        self.parts_data, self.n_parts = self.get_parts()
        if n > 1:
            message = [
                f"\t\t- Multiple Body parts found : {self.n_parts}",
            ]
        else:
            message = [
                f"\t\t- Mesh is one Body part",
            ]
        self.log(message)
        self.info_dict[self.file_name]["Mesh Processing"][
            "Checking mesh multiple Body parts"
        ] = f"{self.n_parts}  Body part(s) found"

        self.step += 1

        if not self.current_count:
            self.log(self.cancel_message)
            self.info_dict[self.file_name]["Mesh Processing"][
                "Mesh cleaning"
            ] = "CANCELLED"
            ################ Fix Normals : ################################
            message = [
                f"\n\t- Fixing Normals : ",
            ]
            self.log(message)

            fixed_normals = self.make_normals_consistent()

            message = [
                f"\t\t- DONE : (Fixed Normals : {len(fixed_normals)})",
            ]
            self.log(message)

            self.info_dict[self.file_name]["Mesh Processing"][
                "Fixing Normals"
            ] = f"(fixed Normals : {len(fixed_normals)})"

            ################ case : <SUCCESS> --> Export Mesh : ################################
            out_path = join(
                self.export_path,
                f"({self.file_name})_Cleaned_Non_Destructive_(Printable).stl",
            )

            message = [f"\n\t- STEP {self.step} Exporting Mesh STL ... "]
            self.log(message)
            self.export_stl(out_path)
            message = [
                f"\t\t- File exported : {RelPath(out_path)}",
            ]
            self.log(message)

            self.info_dict[self.file_name]["Mesh Export"][
                "Cleaned Non Destructive Printable Mesh STL"
            ] = RelPath(out_path)

        else:
            ###################### Run clean pipeline #########################
            self.mean_edge_lenght = self.get_mean_edge_lenght()
            self.clean()

            ################ Fix Normals : ################################
            message = [
                f"\n\t- Fixing Normals : ",
            ]
            self.log(message)

            fixed_normals = self.make_normals_consistent()

            message = [
                f"\t\t- DONE : (Fixed Normals : {len(fixed_normals)})",
            ]
            self.log(message)

            self.info_dict[self.file_name]["Mesh Processing"][
                "Fixing Normals"
            ] = f"(fixed Normals : {len(fixed_normals)})"

            self.step += 1

        ################ END : ################################
        self.current_count = self.count_non_manifold_verts()
        if not self.current_count:
            message = [
                "\t- RESULT : Mesh Processing <SUCCESS>",
            ]
            self.info_dict[self.file_name]["Mesh Processing"]["RESULT"] = "SUCCESS"
        else:
            message = [
                "\t- RESULT : Mesh Processing <FAILED>",
            ]
            self.log(message)
            self.info_dict[self.file_name]["Mesh Processing"]["RESULT"] = "FAILED"

        return self.info_dict

    def clean(self):
        ################  Delete Loose Geometry : ################################
        message = [f"\n\t- STEP {self.step} : Removing Loose Geometry ..."]
        self.log(message)

        self.delete_loose()

        self.current_count = self.count_non_manifold_verts()
        message = [
            f"\t\t- DONE : (Non Manifold verts : {self.current_count})",
        ]
        self.log(message)

        self.info_dict[self.file_name]["Mesh Processing"][
            "Remove Loose Geometry"
        ] = f"Non Manifold vertices count = ({self.current_count})"
        self.step += 1

        ################  Merge Verts 1rst Pass : ################################

        if not self.current_count:
            self.info_dict[self.file_name]["Mesh Processing"][
                f"Merge Vertices 1rst Pass threshold{self.thresh_1}"
            ] = "CANCELLED"
        else:

            message = [
                f"\n\t- STEP {self.step} : Merging Vertices 1rst Pass (Threshold = {self.thresh_1})..."
            ]
            self.log(message)

            self.merge_verts(self.thresh_1, all=True)
            self.current_count = self.count_non_manifold_verts()

            message = [
                f"\t\t- DONE : (Non Manifold verts : {self.current_count})",
            ]
            self.log(message)

            self.info_dict[self.file_name]["Mesh Processing"][
                f"Merge Vertices 1rst Pass threshold ({self.thresh_1})"
            ] = f"Non Manifold vertices count = ({self.current_count})"
            self.step += 1
        ################ Merge Verts 2nd Pass : ################################

        if not self.current_count:
            self.info_dict[self.file_name]["Mesh Processing"][
                f"Merge Vertices 2nd Pass threshold{self.thresh_2}"
            ] = "CANCELLED"
        else:
            message = [
                f"\n\t- STEP {self.step} : Merging Vertices 2nd Pass (Threshold = {self.thresh_2})..."
            ]
            self.log(message)

            self.merge_verts(self.thresh_2)
            self.current_count = self.count_non_manifold_verts()

            message = [
                f"\t\t- DONE : (Non Manifold verts : {self.current_count})",
            ]

            self.log(message)
            self.info_dict[self.file_name]["Mesh Processing"][
                f"Merge Vertices 2nd Pass threshold ({self.thresh_2})"
            ] = f"Non Manifold vertices count = ({self.current_count})"
            self.step += 1
        ################ STEP 7 Fill holes : ################################
        if not self.current_count:
            self.info_dict[self.file_name]["Mesh Processing"][
                f"Fill Holes {self.thresh_2}"
            ] = "CANCELLED"
        else:
            message = [f"\n\t- STEP {self.step} : Filling Holes ..."]
            self.log(message)

            self.fill_holes()
            self.current_count = self.count_non_manifold_verts()

            message = [
                f"\t\t- DONE : (Non Manifold verts : {self.current_count})",
            ]

            self.log(message)
            self.info_dict[self.file_name]["Mesh Processing"][
                f"Fill Holes"
            ] = f"Non Manifold vertices count = ({self.current_count})"
            self.step += 1

        ################ case : <SUCCESS> --> Export non destructive Repair : ################################
        if not self.current_count:
            out_path = join(
                self.export_path,
                f"({self.file_name})_Cleaned_Non_Destructive_(Printable).stl",
            )

            message = [f"\n\t- STEP {self.step} Exporting Mesh STL ... "]
            self.log(message)
            self.export_stl(out_path)
            message = [
                f"\t\t- File exported : {RelPath(out_path)}",
            ]
            self.log(message)

            self.info_dict[self.file_name]["Mesh Export"][
                "Cleaned Non Destructive Printable Mesh STL"
            ] = RelPath(out_path)
            self.step += 1

        else:

            ################ case : <Fail> --> Export non destructive Repair : ################################
            out_path = join(
                self.export_path,
                f"({self.file_name})_Repaired_Non_Destructive_(Non Manifold).stl",
            )

            message = [f"\n\t- STEP {self.step} Exporting Mesh STL ... "]
            self.log(message)
            self.export_stl(out_path)
            message = [
                f"\t\t- File exported : {RelPath(out_path)}",
            ]
            self.log(message)

            self.info_dict[self.file_name]["Mesh Export"][
                "Repaired Non Destructive (Non Manifold) Mesh STL"
            ] = RelPath(out_path)
            self.step += 1

        if self.remesh:
            if not self.current_count:
                self.info_dict[self.file_name]["Mesh Processing"][
                    "Remesh Sharp"
                ] = "CANCELLED"
            else:

                ################ Remesh sharp : ################################
                message = [f"\n\t- STEP {self.step} : Sharp Remeshing ..."]
                self.log(message)

                self.remesh_sharp()
                self.current_count = self.count_non_manifold_verts()

                message = [
                    f"\t\t- DONE : (Non Manifold verts : {self.current_count})",
                ]

                self.log(message)
                self.info_dict[self.file_name]["Mesh Processing"][
                    f"Sharp Remesh"
                ] = f"Non Manifold vertices count = ({self.current_count})"
                self.step += 1

                ################ case : <SUCCESS> --> Export Sharp Remesh Repair : ################################
                if not self.current_count:
                    self.info_dict[self.file_name]["Mesh Processing"][
                        "Remesh Voxel"
                    ] = "CANCELLED"

                    out_path = join(
                        self.export_path,
                        f"({self.file_name})_Cleaned_Sharp_Remeshed_(Printable).stl",
                    )

                    message = [f"\n\t- STEP {self.step} Exporting Mesh STL ... "]
                    self.log(message)
                    self.export_stl(out_path)
                    message = [
                        f"\t\t- File exported : {RelPath(out_path)}",
                    ]
                    self.log(message)

                    self.info_dict[self.file_name]["Mesh Export"][
                        "Cleaned Sharp Remeshed Printable Mesh STL"
                    ] = RelPath(out_path)
                    self.step += 1

                else:

                    ################ case : <Fail> --> Export Sharp Remesh Repair : ################################
                    out_path = join(
                        self.export_path,
                        f"({self.file_name})_Repaired_Sharp_Remeshed_(Non Manifold).stl",
                    )

                    message = [f"\n\t- STEP {self.step} Exporting Mesh STL ... "]
                    self.log(message)
                    self.export_stl(out_path)
                    message = [
                        f"\t\t- File exported : {RelPath(out_path)}",
                    ]
                    self.log(message)

                    self.info_dict[self.file_name]["Mesh Export"][
                        "Repaired Sharp Remeshed (Non Manifold) Mesh STL"
                    ] = RelPath(out_path)
                    self.step += 1

                    ################ Remesh Voxel : ################################
                    # self.log(['\n\t- STEP 9 : Remesh Voxel : Bypassed'])

                    message = [f"\n\t- STEP {self.step} : Voxel Remeshing ..."]
                    self.log(message)

                    self.remesh_voxel(self.mean_edge_lenght)
                    self.current_count = self.count_non_manifold_verts()

                    message = [
                        f"\t\t- DONE : (Non Manifold verts : {self.current_count})",
                    ]

                    self.log(message)
                    self.info_dict[self.file_name]["Mesh Processing"][
                        f"Voxel Remesh"
                    ] = f"Non Manifold vertices count = ({self.current_count})"
                    self.step += 1

                    ################ case : <SUCCESS> --> Export Sharp Remesh Repair : ################################
                    if not self.current_count:

                        out_path = join(
                            self.export_path,
                            f"({self.file_name})_Cleaned_Voxel_Remeshed_(Printable).stl",
                        )

                        message = [f"\n\t- STEP {self.step} Exporting Mesh STL ... "]
                        self.log(message)
                        self.export_stl(out_path)
                        message = [
                            f"\t\t- File exported : {RelPath(out_path)}",
                        ]
                        self.log(message)

                        self.info_dict[self.file_name]["Mesh Export"][
                            "Cleaned Voxel Remeshed Printable Mesh STL"
                        ] = RelPath(out_path)
                        self.step += 1

                    else:

                        ################ case : <Fail> --> Export Sharp Remesh Repair : ################################
                        out_path = join(
                            self.export_path,
                            f"({self.file_name})_Repaired_Voxel_Remeshed_(Non Manifold).stl",
                        )

                        message = [f"\n\t- STEP {self.step} Exporting Mesh STL ... "]
                        self.log(message)
                        self.export_stl(out_path)
                        message = [
                            f"\t\t- File exported : {RelPath(out_path)}",
                        ]
                        self.log(message)

                        self.info_dict[self.file_name]["Mesh Export"][
                            "Repaired Voxel Remeshed (Non Manifold) Mesh STL"
                        ] = RelPath(out_path)
                        self.step += 1

        return self.info_dict

    #####################################################
    #####################################################
    def export_intersecting_faces_fbx(self):
        D = bpy.data
        obj = self.obj
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.duplicate_move()
        obj_copy = bpy.context.active_object

        bpy.context.view_layer.objects.active = obj_copy
        bpy.ops.object.select_all(action="DESELECT")
        obj_copy.select_set(True)

        overlaping_pairs, overlaping_faces, n = self.find_intersections(
            obj_copy, reveal=True
        )

        body_mat = D.materials.get("body_mat") or D.materials.new(name="body_mat")
        obj_copy.active_material = body_mat

        intersect_mat = D.materials.get("intersecting_faces_mat") or D.materials.new(
            name="intersecting_faces_mat"
        )
        intersect_mat.diffuse_color = [1, 0, 0, 1]
        obj_copy.data.materials.append(intersect_mat)
        obj_copy.active_material_index = 1
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.material_slot_assign()

        bpy.ops.object.mode_set(mode="OBJECT")
        out_path = join(self.export_path, f"({self.file_name})_Intersecting_Faces_.fbx")
        self.export_fbx(out_path)
        bpy.data.objects.remove(obj_copy)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        return out_path

    #####################################################

    def export_fbx(self, path):
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True)

    def get_bounding_box(self):
        mu = mathutils
        obj = self.obj
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
        dims = obj.dimensions
        bb = [tuple(obj.matrix_world @ mu.Vector(i[:])) for i in obj.bound_box]
        height = round(dims[2], 2)
        a = round(dims[0], 2)
        b = round(dims[1], 2)
        lenght = max(a, b)
        width = min(a, b)

        return bb, width, lenght, height

    #####################################################
    def info_volume(self):

        obj = self.obj
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")

        bm = self.bmesh_copy_from_object(obj)
        volume = round(bm.calc_volume(), 2)
        bm.free()
        volume_message = f"Volume : {volume} mm³"
        volume_info = f"{volume} mm3"

        return volume_message, volume_info

    #####################################################

    def bmesh_calc_area(self, bm):
        """Calculate the surface area."""
        return sum(f.calc_area() for f in bm.faces)

    def info_area(self):

        obj = self.obj
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")

        bm = self.bmesh_copy_from_object(obj)
        area = round(self.bmesh_calc_area(bm), 2)
        bm.free()

        area_message = f"Surface Area : {area} mm²"
        area_info = f"{area} mm2"

        return area_message, area_info

    def bmesh_copy_from_object(self, obj):
        """Returns bmesh copy of the mesh"""

        assert obj.type == "MESH"
        me = obj.data
        if obj.mode == "EDIT":
            bm_orig = bmesh.from_edit_mesh(me)
            bm = bm_orig.copy()
        else:
            bm = bmesh.new()
            bm.from_mesh(me)

        return bm

    def log(self, log_list):
        for l in log_list:
            print(l)

    def mesh_count(self, obj):
        bm = self.bmesh_copy_from_object(obj)
        results = len(bm.verts), len(bm.edges), len(bm.faces)
        bm.free()
        return results

    def reset_scene(self):
        D = bpy.data
        for coll in D.collections:
            D.collections.remove(coll)
        for mesh in D.meshes:
            D.meshes.remove(mesh)
        for obj in D.objects:
            D.objects.remove(obj)
        for mat in D.materials:
            D.materials.remove(mat)
        for img in D.images:
            D.images.remove(img)

    def prepare_scene(self):
        C = bpy.context
        self.reset_scene()
        bpy.ops.wm.stl_import(filepath=self.file)
        self.obj = C.object
        self.obj_name = self.obj.name

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        verts_count, edges_count, polygons_count = self.mesh_count(self.obj)

        return verts_count, edges_count, polygons_count

    def get_separate_objects(self) -> list:
        """separate object to loose parts"""
        bpy.ops.mesh.separate(type="LOOSE")
        return bpy.data.objects[:]

    def count_non_manifold_verts(self):
        """return non manifold verts count"""
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        self.select_non_manifold_verts(use_wire=True, use_boundary=True, use_verts=True)
        bpy.ops.object.mode_set(mode="OBJECT")

        bm = self.bmesh_copy_from_object(self.obj)
        return sum((1 for v in bm.verts if v.select))

    def select_non_manifold_verts(
        self,
        use_wire=False,
        use_boundary=False,
        use_multi_face=False,
        use_non_contiguous=False,
        use_verts=False,
    ):
        """select non-manifold vertices"""
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_non_manifold(
            extend=False,
            use_wire=use_wire,
            use_boundary=use_boundary,
            use_multi_face=use_multi_face,
            use_non_contiguous=use_non_contiguous,
            use_verts=use_verts,
        )

    def delete_loose(self):
        """delete loose vertices/edges/faces"""
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=True)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

    def delete_interior_faces(self):
        """delete interior faces"""

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.fill_holes(sides=4)
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode="OBJECT")

    def merge_verts(self, threshold, all=False):
        """merge vertices with respect to a distance threshold"""
        bpy.ops.object.mode_set(mode="EDIT")
        self.select_non_manifold_verts(use_wire=True, use_boundary=True, use_verts=True)
        if all:
            bpy.ops.mesh.select_all(action="SELECT")

        bpy.ops.mesh.remove_doubles(threshold=threshold)
        bpy.ops.object.mode_set(mode="OBJECT")

    def degenerate_dissolve(self, threshold):
        """dissolve zero area faces and zero length edges"""
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.dissolve_degenerate(threshold=threshold)
        bpy.ops.object.mode_set(mode="OBJECT")

    def fill_non_manifold(self):
        """fill holes"""
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=self.hole_size)
        bpy.ops.object.mode_set(mode="OBJECT")

    def fill_holes(self):
        """fill holes"""
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=self.hole_size)
        bpy.ops.object.mode_set(mode="OBJECT")

    def delete_newly_generated_non_manifold_verts(self):
        """delete any newly generated vertices from the filling repair"""
        self.select_non_manifold_verts(use_wire=True, use_verts=True)
        bpy.ops.mesh.delete(type="VERT")
        # bpy.ops.object.mode_set(mode='EDIT')
        # bpy.ops.mesh.select_all(action='DESELECT')

        # bpy.ops.mesh.select_all(action='DESELECT')
        # bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=False, use_multi_face=True, use_non_contiguous=False, use_verts=False)
        # bpy.ops.mesh.delete(type='EDGE')

        # bpy.ops.mesh.select_all(action='DESELECT')
        # bpy.ops.mesh.select_non_manifold(extend=False, use_wire=True, use_boundary=False, use_multi_face=False, use_non_contiguous=False, use_verts=False)
        # bpy.ops.mesh.delete_loose()

        # bpy.ops.mesh.select_all(action='SELECT')
        # bpy.ops.mesh.fill_holes(sides=400)

        # bpy.ops.mesh.select_all(action='DESELECT')
        # bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=False, use_multi_face=False, use_non_contiguous=False, use_verts=True)
        # bpy.ops.mesh.delete(type='FACE')

    def elem_count(self):
        C = bpy.context
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(C.edit_object.data)
        return len(bm.verts), len(bm.edges), len(bm.faces)

    def iterate_fill_clean(self):
        """iterate-until-no-more approach for fixing non manifolds"""
        total_non_manifold = self.count_non_manifold_verts()

        if not total_non_manifold:
            return

        progression = set()
        bm_key = self.elem_count()
        progression.add(bm_key)

        counter = 1
        while True:
            print(
                f"\t\t-iteration : {counter}, non manifold verts count : {self.count_non_manifold_verts()}"
            )
            counter += 1
            self.fill_non_manifold()
            self.delete_newly_generated_non_manifold_verts()

            bm_key = self.elem_count()
            if bm_key in progression:
                break
            else:
                progression.add(bm_key)

    def make_normals_consistent(self):
        """repair normals"""
        C = bpy.context
        bpy.ops.object.mode_set(mode="OBJECT")
        dg = C.evaluated_depsgraph_get()
        mesh_eval_0 = self.obj.data.evaluated_get(dg)
        total = len(mesh_eval_0.polygons)
        fixed_normals = np.arange(total)

        a_start = np.zeros((total, 3)).flatten()
        mesh_eval_0.polygons.foreach_get("normal", a_start)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        dg = C.evaluated_depsgraph_get()
        mesh_eval_1 = self.obj.data.evaluated_get(dg)
        a_end = np.zeros((total, 3)).flatten()
        mesh_eval_1.polygons.foreach_get("normal", a_end)

        a = a_start.reshape((total, 3))
        b = a_end.reshape((total, 3))

        true_array = np.all(np.equal(a, -b), axis=1)

        fixed_normals = fixed_normals[true_array]

        return fixed_normals

    def push_undo_point(self):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.ed.undo_push(message="my_undo")

    def export_stl(self, path):
        bpy.ops.export_mesh.stl(filepath=path)

    def mesh_bool_union(self):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        #        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.intersect_boolean(
            operation="UNION", use_swap=False, use_self=True, solver="FAST"
        )

    #        bpy.ops.mesh.select_all(action='DESELECT')
    #        bpy.ops.mesh.select_mode(type='VERT')
    #        bpy.ops.object.mode_set(mode='OBJECT')

    def MoveToCollection(self, obj, CollName):

        OldColl = obj.users_collection  # list of all collection the obj is in
        NewColl = bpy.data.collections.get(CollName)
        if not NewColl:
            NewColl = bpy.data.collections.new(CollName)
            bpy.context.scene.collection.children.link(NewColl)
        if not obj in NewColl.objects[:]:
            NewColl.objects.link(obj)  # link obj to scene
        if OldColl:
            for Coll in OldColl:  # unlink from all  precedent obj collections
                if Coll is not NewColl:
                    Coll.objects.unlink(obj)
        return NewColl

    def get_parts(self):

        obj = self.obj

        name = obj.name
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")

        face_ids = [f.index for f in obj.data.polygons]
        data = {}
        counter = 1
        while face_ids:
            # print(counter)
            idx = face_ids[0]
            obj.data.polygons[idx].select = True
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_linked()
            bpy.ops.object.mode_set(mode="OBJECT")
            selected = [f.index for f in obj.data.polygons if f.select]
            data.update({f"PART{counter}": {"COUNT": len(selected), "IDS": selected}})
            face_ids = [f.index for f in obj.data.polygons if not f.select]
            counter += 1
        n = len(data)
        return data, n

    def separate_mesh(self):
        name = self.obj_name
        bpy.context.view_layer.objects.active = self.obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")

        face_ids = [f.index for f in self.obj.data.polygons]

        while face_ids:
            idx = face_ids.pop()
            self.obj.data.polygons[idx].select = True
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_linked()
            bpy.ops.mesh.separate(type="SELECTED")
            bpy.ops.object.mode_set(mode="OBJECT")
            face_ids = [f.index for f in self.obj.data.polygons]

        bpy.data.objects.remove(bpy.data.objects[name])
        self.obj = bpy.data.objects[0]
        self.obj.name = name
        parts = bpy.data.objects
        bpy.context.view_layer.objects.active = self.obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")

        return parts

    def find_intersections(self, obj, reveal=False):
        bpy.ops.object.mode_set(mode="OBJECT")

        if not obj.data.polygons:
            return [], [], []

        dg = bpy.context.evaluated_depsgraph_get()
        tree = bvhtree.BVHTree.FromObject(obj, dg)

        overlaping_pairs = tree.overlap(tree)
        overlaping_faces = list({i for i_pair in overlaping_pairs for i in i_pair})
        n = len(overlaping_faces)
        #    non_overlapping_faces = [ f.index for f in obj.data.polygons if not f.index in overlaping_faces ]

        # print("OVERLAPPING FACES", ":", len(overlaping_faces) )

        if reveal:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            for idx in overlaping_faces:
                obj.data.polygons[idx].select = True

            bpy.ops.object.mode_set(mode="EDIT")

        return overlaping_pairs, overlaping_faces, n

    def intersect_overlaping_faces(self):
        obj = self.obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_mode(type="FACE")

        bpy.ops.object.mode_set(mode="OBJECT")
        for i in self.overlaping_faces:
            obj.data.polygons[i].select = True
            obj.data.polygons.active = i

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.intersect(mode="SELECT", separate_mode="ALL", solver="EXACT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=4)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

    def remove_overlaping_faces(self):
        obj = self.obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_mode(type="FACE")

        bpy.ops.object.mode_set(mode="OBJECT")
        for i in self.overlaping_faces:
            obj.data.polygons[i].select = True
            obj.data.polygons.active = i

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.intersect(mode="SELECT", separate_mode="ALL", solver="EXACT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=4)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")

    def separate_bool_union(self):
        name = self.obj_name
        overlaping_pairs, overlaping_faces = self.find_intersections()
        parts = self.separate_mesh(overlaping_faces)
        n = len(parts)
        main_part = bpy.data.objects[name]

        if n > 1:

            bool_objects = [o for o in bpy.data.objects if o.name != name]

            for obj in bool_objects:
                bpy.ops.object.select_all(action="DESELECT")
                main_part.select_set(True)
                bpy.context.view_layer.objects.active = main_part
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_mode(type="VERT")
                bpy.ops.mesh.select_all(action="SELECT")

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                main_part.select_set(True)
                bpy.context.view_layer.objects.active = main_part
                obj.select_set(True)
                bpy.ops.object.join()

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_mode(type="FACE")
                bpy.ops.mesh.intersect_boolean(operation="UNION", solver="EXACT")

                bpy.ops.mesh.select_mode(type="VERT")
                bpy.ops.mesh.select_interior_faces()
                bpy.ops.mesh.delete(type="FACE")
                bpy.ops.object.mode_set(mode="OBJECT")

        self.obj = main_part
        return n

    def remesh_sharp(self, octree_depth=10):
        obj = self.obj
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.modifier_add(type="REMESH")
        obj.modifiers["Remesh"].show_viewport = False
        obj.modifiers["Remesh"].mode = "SHARP"
        obj.modifiers["Remesh"].octree_depth = octree_depth
        obj.modifiers["Remesh"].scale = 0.99
        obj.modifiers["Remesh"].use_remove_disconnected = False
        # bpy.context.object.modifiers["Remesh"].threshold = 0.1
        bpy.ops.object.modifier_apply(modifier="Remesh")
        bpy.ops.object.mode_set(mode="OBJECT")

    def remesh_voxel(self, voxel_size=0.1):
        obj = self.obj
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.modifier_add(type="REMESH")
        obj.modifiers["Remesh"].show_viewport = False
        obj.modifiers["Remesh"].mode = "VOXEL"
        obj.modifiers["Remesh"].voxel_size = round(voxel_size / 2, 2)
        bpy.ops.object.modifier_apply(modifier="Remesh")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.modifier_apply(modifier="Remesh")
        bpy.ops.object.mode_set(mode="OBJECT")

    def get_mean_edge_lenght(self):
        bpy.ops.object.mode_set(mode="EDIT")

        # make a bmesh from the mesh
        me = self.obj.data
        bm = bmesh.from_edit_mesh(me)

        edge_lengths = [e.calc_length() for e in bm.edges]

        return np.mean(edge_lengths)


##################################################################################


def add_collection(name, parent_collection=None):
    coll = bpy.data.collections.get(name)
    if not coll :
        coll = bpy.data.collections.new(name)
        p_collection = parent_collection or bpy.context.scene.collection
        p_collection.children.link(coll)

    return coll


def append_group_nodes(group_name):
    group_node = bpy.data.node_groups.get(group_name)
    if not group_node:
        filepath = join(OdentConstants.DATA_BLEND_FILE, "NodeTree", group_name)
        directory = join(OdentConstants.DATA_BLEND_FILE, "NodeTree")
        filename = group_name
        bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)
        group_node = bpy.data.node_groups.get(group_name)
    return group_node


def mesh_to_volume(obj, odent_volume_node, offset):
    gn_modifier = obj.modifiers.new(name="odent_volume_modifier", type="NODES")
    gn_modifier.node_group = odent_volume_node
    nodes = gn_modifier.node_group.nodes
    # offset_out_node = nodes.get("extrude_out").inputs["Offset Scale"]
    # offset_in_node = nodes.get("extrude_in").inputs["Offset Scale"]
    # offset_out_node.default_value = offset_out
    # offset_in_node.default_value = offset_in
    
    offset_node = nodes.get("offset")
    offset_node.outputs[0].default_value = offset


def rotate_local(obj, target, axis, angle):
    mat_rot = Matrix.Rotation(radians(angle), 4, axis)
    obj_local = target.matrix_world.inverted() @ obj.matrix_world
    obj_local_rot = mat_rot @ obj_local
    obj.matrix_world = target.matrix_world @ obj_local_rot

def translate_local(obj, amount, axis='X'):
    """
    Translate the object along its local axis.

    :param obj: The Blender object to move
    :param amount: The distance to move
    :param axis: Axis to move along: 'X', 'Y', or 'Z'
    """
    axis = axis.upper()
    if axis not in 'XYZ':
        raise ValueError("Axis must be 'X', 'Y', or 'Z'")

    # Get local axis direction in world space
    if axis == "X" : direction = obj.matrix_world.to_3x3().col[0]
    elif axis == "Y" : direction = obj.matrix_world.to_3x3().col[1]
    elif axis == "Z" : direction = obj.matrix_world.to_3x3().col[2]
    # direction = obj.matrix_world.to_3x3()[{'X': 0, 'Y': 1, 'Z': 2}[axis]]
    
    # Move object along that direction
    obj.location += direction * amount

def get_odent_implant_data_dict(scene_prop):
    odent_implant_data = {}
    for k, v in scene_prop.items():
        odent_implant_data[k] = v
    return odent_implant_data


def get_slicer_areas(_slicer_ws):
    _slicer_scr = None
    _slicer_axial = None
    _slicer_coronal = None
    _slicer_sagittal = None
    _slicer_area_3d = None
    _slicer_area_outliner = None                                                                                                    
    success = 0
    try :
        _slicer_scr = _slicer_ws.screens[0]
        #get outliner area :
        _slicer_area_outliner = [a for a in _slicer_scr.areas if a.type == "OUTLINER"][0]

        _slicer_areas_3d_all = [a for a in _slicer_scr.areas if a.type == "VIEW_3D"]
        sorted_slicer_areas_by_area_y_coordinate = sorted([a for a in _slicer_areas_3d_all], key=lambda a: a.y)

        #get 3d area :
        _slicer_area_3d = sorted_slicer_areas_by_area_y_coordinate[0]

        sorted_slicer_areas_by_area_x_coordinate = sorted([a for a in _slicer_areas_3d_all if a != _slicer_area_3d], key=lambda a: a.x)

        #get the 3 slicer areas :
        _slicer_axial = sorted_slicer_areas_by_area_x_coordinate[0]
        _slicer_coronal = sorted_slicer_areas_by_area_x_coordinate[1]
        _slicer_sagittal = sorted_slicer_areas_by_area_x_coordinate[2]

        success = 1

    except Exception as e:
        odent_log([e])
        pass
    return success, _slicer_scr, _slicer_axial, _slicer_coronal, _slicer_sagittal, _slicer_area_3d, _slicer_area_outliner

def set_slices_workspace():
    ws, scr, area_axial, area_coronal, area_sagittal, area_3d = get_slicer_areas()
    coll_names = [OdentConstants.SLICES_COLLECTION_NAME, 
                  OdentConstants.SLICES_POINTERS_COLLECTION_NAME,
                  OdentConstants.GUIDE_COMPONENTS_COLLECTION_NAME]

    for a in [area_axial, area_coronal, area_sagittal]:
        space_data = a.spaces.active
        with bpy.context.temp_override(screen=scr, area=a, space_data=space_data):
            bpy.ops.wm.tool_set_by_id(name="builtin.move")
            space_data.use_local_collections = True
            for col_name in coll_names:
                index = getLocalCollIndex(col_name)
                bpy.ops.object.hide_collection(collection_index=index, toggle=True)

    return ws, scr, area_axial, area_coronal, area_sagittal


def unlock_object(obj):
    obj.lock_location = (False, False, False)
    obj.lock_rotation = (False, False, False)
    obj.lock_scale = (False, False, False)


def lock_object(obj):
    obj.lock_location = (True, True, True)
    obj.lock_rotation = (True, True, True)
    obj.lock_scale = (True, True, True)


def AppendObject(objName, coll_name=None):
    filename = objName
    directory = join(OdentConstants.DATA_BLEND_FILE, "Object")
    bpy.ops.wm.append(directory=directory, filename=filename)
    obj = bpy.data.objects[objName]
    if coll_name:
        MoveToCollection(obj, coll_name)
    return obj


def AppendCollection(coll_name, parent_coll_name=None):
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection = view_layer.layer_collection

    directory = join(OdentConstants.DATA_BLEND_FILE, "Collection")
    bpy.ops.wm.append(directory=directory, filename=coll_name)
    coll = bpy.data.collections.get(coll_name)

    if parent_coll_name:
        parent_coll = bpy.data.collections.get(parent_coll_name)
        if parent_coll:
            parent_coll.children.link(coll)
            bpy.context.scene.collection.children.unlink(coll)
    return coll


def get_landmarks_dict(landmarks_id_prop):
    landmarks_dict = {}
    for k, v in landmarks_id_prop.items():
        landmarks_dict[k] = v
    return landmarks_dict



def GetAutoReconstructParameters(Manufacturer, ConvKernel):

    Soft, Bone, Teeth = None, None, None

    if ConvKernel != None:

        if Manufacturer == "NewTom":
            Soft, Bone, Teeth = -400, 606, 1032

        if Manufacturer == "J.Morita.Mfg.Corp." and ConvKernel == "FBP":
            Soft, Bone, Teeth = -365, 200, 455

        else:
            if (
                ("Hr40f" in ConvKernel and "3" in ConvKernel)
                or ("J30s" in ConvKernel and "3" in ConvKernel)
                or ("J30f" in ConvKernel and "2" in ConvKernel)
                or ("I31f" in ConvKernel and "3" in ConvKernel)
                or ("Br40f" in ConvKernel and "3" in ConvKernel)
                or ("Hr38h" in ConvKernel and "3" in ConvKernel)
                or ConvKernel
                in (
                    "FC03",
                    "FC04",
                    "STANDARD",
                    "H30s",
                    "SOFT",
                    "UB",
                    "SA",
                    "FC23",
                    "FC08",
                    "FC21",
                    "A",
                    "FC02",
                    "B",
                    "H23s",
                    "H20s",
                    "H31s",
                    "H32s",
                    "H40s",
                    "H31s",
                    "B41s",
                    "B70s",
                    "H22s",
                    "H20f",
                    "FC68",
                    "FC07",
                    "B30s",
                    "B41s",
                    "D10f",
                    "B45s",
                    "B26f",
                    "B30f",
                    "32",
                    "SB",
                    "FC15",
                    "FC69",
                    "UA",
                    "10",
                    "STND",
                    "H30f",
                    "B20s",
                )
            ):

                Soft, Bone, Teeth = -300, 200, 1430

            if (
                ("Hr60f" in ConvKernel and "3" in ConvKernel)
                or ("I70f" in ConvKernel and "3" in ConvKernel)
                or ("Hr64h" in ConvKernel and "3" in ConvKernel)
                or ConvKernel
                in (
                    "BONE",
                    "BONEPLUS",
                    "FC30",
                    "H70s",
                    "D",
                    "EA",
                    "FC81",
                    "YC",
                    "YD",
                    "H70h",
                    "H60s",
                    "H60f",
                    "FC35",
                    "B80s",
                    "H90s",
                    "B70f",
                    "EB",
                    "11H",
                    "C",
                    "B60s",
                )
            ):

                Soft, Bone, Teeth = -300, 400, 995

    if not ConvKernel:

        if Manufacturer == "Imaging Sciences International":
            Soft, Bone, Teeth = -400, 358, 995

        if Manufacturer == "SOREDEX":
            Soft, Bone, Teeth = -400, 410, 880

        if Manufacturer == "Xoran Technologies ®":
            Soft, Bone, Teeth = -400, 331, 1052

        if Manufacturer == "Planmeca":
            Soft, Bone, Teeth = -400, 330, 756

        if Manufacturer == "J.Morita.Mfg.Corp.":
            Soft, Bone, Teeth = -315, 487, 787

        if Manufacturer in ["Carestream Health", "Carestream Dental"]:
            Soft, Bone, Teeth = -400, 388, 1013

        if Manufacturer == "MyRay":
            Soft, Bone, Teeth = -360, 850, 1735

        if Manufacturer == "NIM":
            Soft, Bone, Teeth = -1, 1300, 1260

        if Manufacturer == "PreXion":
            Soft, Bone, Teeth = -400, 312, 1505

        if Manufacturer == "Sirona":
            Soft, Bone, Teeth = -170, 590, 780

        if Manufacturer == "Dabi Atlante":
            Soft, Bone, Teeth = -375, 575, 1080

        if Manufacturer == "INSTRUMENTARIUM DENTAL":
            Soft, Bone, Teeth = -400, 430, 995

        if Manufacturer == "Instrumentarium Dental":
            Soft, Bone, Teeth = -357, 855, 1489

        if Manufacturer == "Vatech Company Limited":
            Soft, Bone, Teeth = -328, 780, 1520

    return Soft, Bone, Teeth


#######################################################################################
# Popup message box function :
#######################################################################################


def ShowMessageBox(message=[], title="INFO", icon="INFO"):
    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.alert = True
        box.alignment = "EXPAND"

        for txt in message:
            row = box.row()
            row.label(text=txt)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


#######################################################################################
# Load CT Scan functions :
#######################################################################################
def Align_Implants(Averrage=False):
    ctx = bpy.context
    Implants = ctx.selected_objects
    n = len(Implants)
    Active_Imp = ctx.active_object
    if n < 2 or not Active_Imp:
        print("Please select at least 2 implants \nThe last selected is the active")
        return
    else:
        if Averrage:
            MeanRot = np.mean(
                [np.array(Impt.rotation_euler) for Impt in Implants], axis=0
            )
            for Impt in Implants:
                Impt.rotation_euler = MeanRot

        else:

            for Impt in Implants:
                Impt.rotation_euler = Active_Imp.rotation_euler


def CheckString(String, MatchesList, mode=all):
    if mode(x in String for x in MatchesList):
        return True
    else:
        return False


def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)


def get_all_addons(display=False):
    """
    Prints the addon state based on the user preferences.

    """
    import sys

    import bpy as _bpy  # type: ignore
    from addon_utils import check, enable, paths  # type: ignore

    # RELEASE SCRIPTS: official scripts distributed in Blender releases
    paths_list = paths()
    addon_list = []
    for path in paths_list:
        _bpy.utils._sys_path_ensure(path)
        for mod_name, mod_path in _bpy.path.module_names(path):
            is_enabled, is_loaded = check(mod_name)
            addon_list.append(mod_name)
            if display:  # for example
                print("%s default:%s loaded:%s" % (mod_name, is_enabled, is_loaded))

    return addon_list


def install_blender_extensions(ext_list=[]):
    if ext_list:
        for ext_name in ext_list:
            try:
                bpy.ops.extensions.package_install(repo_index=0, pkg_id=ext_name)
            except Exception as er:
                print(
                    f"### error while instaling {ext_name} blender extension ####\n\terror = ({er})"
                )

                ...


def Addon_Enable(AddonName, Enable=True):
    import addon_utils as AU  # type: ignore

    is_enabled, is_loaded = AU.check(AddonName)
    # for mod in AU.modules() :
    # Name = mod.bl_info["name"]
    # print(Name)
    if Enable:
        if not is_enabled:
            AU.enable(AddonName, default_set=True)
    if not Enable:
        if is_enabled:
            AU.disable(AddonName, default_set=True)

    is_enabled, is_loaded = AU.check(AddonName)
    # print(f"{AddonName} : (is_enabled : {is_enabled} , is_loaded : {is_loaded})")


def CleanScanData(Preffix):
    D = bpy.data
    Objects = D.objects
    Meshes = D.meshes
    Images = D.images
    Materials = D.materials
    NodeGroups = D.node_groups

    # Remove Voxel data :
    [Meshes.remove(m) for m in Meshes if f"{Preffix}_PLANE_" in m.name]
    [Images.remove(img) for img in Images if f"{Preffix}_img" in img.name]
    [Materials.remove(mat) for mat in Materials if "BD001_Voxelmat_" in mat.name]
    [NodeGroups.remove(NG) for NG in NodeGroups if "BD001_VGS_" in NG.name]

    # Remove old Slices :
    SlicePlanes = [
        Objects.remove(obj)
        for obj in Objects
        if Preffix in obj.name and "SLICE" in obj.name
    ]
    SliceMeshes = [
        Meshes.remove(m) for m in Meshes if Preffix in m.name and "SLICE" in m.name
    ]
    SliceMats = [
        Materials.remove(mat)
        for mat in Materials
        if Preffix in mat.name and "SLICE" in mat.name
    ]
    SliceImages = [
        Images.remove(img)
        for img in Images
        if Preffix in img.name and "SLICE" in img.name
    ]


def CtxOverride(context):
    area3D = [area for area in context.screen.areas if area.type == "VIEW_3D"][0]
    space3D = [space for space in area3D.spaces if space.type == "VIEW_3D"][0]
    region3D = [reg for reg in area3D.regions if reg.type == "WINDOW"][0]

    return area3D, space3D, region3D

def context_override(context):
    area3D = [area for area in context.screen.areas if area.type == "VIEW_3D"][0]
    space3D = [space for space in area3D.spaces if space.type == "VIEW_3D"][0]
    region3D = [reg for reg in area3D.regions if reg.type == "WINDOW"][0]
    override = {"area": area3D, "space_data": space3D, "region": region3D}
    return override, area3D, space3D, region3D





############################
# Make directory function :
############################
def make_directory(Root, DirName):

    DirPath = join(Root, DirName)
    if not DirName in os.listdir(Root):
        os.mkdir(DirPath)
    return DirPath


################################
# Copy DcmSerie To ProjDir function :
################################
def CopyDcmSerieToProjDir(DcmSerie, DicomSeqDir):
    for i in range(len(DcmSerie)):
        shutil.copy2(DcmSerie[i], DicomSeqDir)


##########################################################################################
######################### ODENT Volume Render : ########################################
##########################################################################################
def AddMaterial(Obj, matName, color, transparacy=None):

    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.diffuse_color = color
    Obj.active_material = mat

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    pbsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
    pbsdf_node.inputs[0].default_value = color
    mat.blend_method = "BLEND"

    if transparacy:
        pbsdf_node.inputs[19].default_value = 0.5


def PlaneCut(Target, Plane, inner=False, outer=False, fill=False):

    bpy.ops.object.select_all(action="DESELECT")
    Target.select_set(True)
    bpy.context.view_layer.objects.active = Target

    Pco = Plane.matrix_world.translation
    Pno = Plane.matrix_world.to_3x3().transposed()[2]

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.bisect(
        plane_co=Pco,
        plane_no=Pno,
        use_fill=fill,
        clear_inner=inner,
        clear_outer=outer,
    )
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")


def AddBooleanCube(DimX, DimY, DimZ):
    bpy.ops.mesh.primitive_cube_add(
        size=max(DimX, DimY, DimZ) * 1.5,
        enter_editmode=False,
        align="WORLD",
        location=(0, 0, 0),
        scale=(1, 1, 1),
    )

    VOI = VolumeOfInterst = bpy.context.object
    VOI.name = "VOI"
    VOI.display_type = "WIRE"
    return VOI


def AddNode(nodes, type, name):

    node = nodes.new(type)
    node.name = name
    # node.location[0] -= 200

    return node


def AddFrankfortPoint(PointsList, color, CollName):
    FrankfortPointsNames = ["R_Or", "L_Or", "R_Po", "L_Po"]
    Loc = bpy.context.scene.cursor.location
    if not PointsList:
        P = AddMarkupPoint(FrankfortPointsNames[0], color, Loc, 1, CollName)
        return P
    if PointsList:
        CurrentPointsNames = [P.name for P in PointsList]
        P_Names = [P for P in FrankfortPointsNames if not P in CurrentPointsNames]
        if P_Names:
            P = AddMarkupPoint(P_Names[0], color, Loc, 1, CollName)
            return P
    else:
        return None


def AddMarkupPoint(name, color, loc, Diameter=1, CollName=None, show_name=False):

    bpy.ops.mesh.primitive_uv_sphere_add(radius=Diameter / 2, location=loc)
    P = bpy.context.object
    P.name = name
    P.data.name = name + "_mesh"

    if CollName:
        MoveToCollection(P, CollName)

    matName = f"{name}_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.diffuse_color = color
    mat.use_nodes = False
    P.active_material = mat
    P.show_name = show_name
    return P


def ProjectPoint(Plane, Point):

    V1, V2, V3, V4 = [Plane.matrix_world @ V.co for V in Plane.data.vertices]
    Ray = Plane.matrix_world.to_3x3() @ Plane.data.polygons[0].normal

    Orig = Point
    Result = Geo.intersect_ray_tri(V1, V2, V3, Ray, Orig, False)
    if not Result:
        Ray *= -1
        Result = Geo.intersect_ray_tri(V1, V2, V3, Ray, Orig, False)
    return Result


def PointsToPlaneMatrix(Ant_Point, PointsList):
    points = np.array(PointsList)
    if not points.shape[0] >= points.shape[1]:
        print("points Array should be of shape (n,m) :")
        print("where n is the number of points and m the dimension( x,y,z = 3)  ")
        return
    C = points.mean(axis=0)
    x = points - C
    M = np.dot(x.T, x)  # Could also use np.cov(x) here.
    N = svd(M)[0][:, -1]

    Center, Normal = Vector(C), Vector(N)

    PlaneZ = Normal
    PlaneX = ((Center - Ant_Point).cross(PlaneZ)).normalized()
    PlaneY = (PlaneZ.cross(PlaneX)).normalized()

    return PlaneX, PlaneY, PlaneZ, Center


def PointsToRefPlanes(Model, RefPointsList, color, CollName=None):
    Dim = max(Model.dimensions) * 1.5
    Na, R_Or, L_Or, R_Po, L_Po = [P.location for P in RefPointsList]
    PlaneX, PlaneY, PlaneZ, Center = PointsToPlaneMatrix(
        Ant_Point=Na, PointsList=[R_Or, L_Or, R_Po, L_Po]
    )
    # Frankfort Ref Plane :
    FrankX = PlaneX
    FrankY = PlaneY
    FrankZ = PlaneZ

    FrankMtx = Matrix((FrankX, FrankY, FrankZ)).to_4x4().transposed()
    FrankMtx.translation = Center

    # Sagittal Median Plane :
    SagZ = -FrankX
    SagX = FrankZ
    SagY = FrankY

    SagMtx = Matrix((SagX, SagY, SagZ)).to_4x4().transposed()
    SagMtx.translation = Center

    # Coronal(Frontal) Plane :
    CorZ = -FrankY
    CorX = FrankX
    CorY = FrankZ

    CorMtx = Matrix((CorX, CorY, CorZ)).to_4x4().transposed()
    CorMtx.translation = Center

    # Add Planes :
    a3d, s3d, r3d = CtxOverride(bpy.context)
    with bpy.context.temp_override(area=a3d, space_data=s3d, region=r3d):
        bpy.ops.mesh.primitive_plane_add(size=Dim)
    FrankfortPlane = bpy.context.object
    name = "01-Frankfort_Plane"
    FrankfortPlane.name = name
    FrankfortPlane.data.name = f"{name}_Mesh"
    FrankfortPlane.matrix_world = FrankMtx
    matName = f"ODENT_RefPlane_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.use_nodes = False
    mat.diffuse_color = color
    FrankfortPlane.active_material = mat

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    pbsdf_node = [n for n in nodes if n.type == "BSDF_PRINCIPLED"][0]
    pbsdf_node.inputs[0].default_value = color
    pbsdf_node.inputs[21].default_value = 0.5

    # mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
    # mat.node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0.5
    mat.blend_method = "BLEND"

    if CollName:
        MoveToCollection(FrankfortPlane, CollName)
    # Add Sagittal Plane :
    bpy.ops.object.duplicate_move()
    SagPlane = bpy.context.object
    name = "02-Sagittal_Median_Plane"
    SagPlane.name = name
    SagPlane.data.name = f"{name}_Mesh"
    SagPlane.matrix_world = SagMtx
    if CollName:
        MoveToCollection(SagPlane, CollName)

    # Add Coronal Plane :
    bpy.ops.object.duplicate_move()
    CorPlane = bpy.context.object
    name = "03-Coronal_Plane"
    CorPlane.name = name
    CorPlane.data.name = f"{name}_Mesh"
    CorPlane.matrix_world = CorMtx
    if CollName:
        MoveToCollection(CorPlane, CollName)

    # Project Na to Coronal Plane :
    Na_Projection_1 = ProjectPoint(Plane=CorPlane, Point=Na)

    # Project Na_Projection_1 to frankfort Plane :
    Na_Projection_2 = ProjectPoint(Plane=FrankfortPlane, Point=Na_Projection_1)

    for Plane in (FrankfortPlane, CorPlane, SagPlane):
        Plane.location = Na_Projection_2

    return [FrankfortPlane, SagPlane, CorPlane]


def PointsToOcclusalPlane(Model, R_pt, A_pt, L_pt, color, subdiv):

    Dim = max(Model.dimensions) * 1.2

    Rco = R_pt.location
    Aco = A_pt.location
    Lco = L_pt.location

    Center = (Rco + Aco + Lco) / 3

    Z = (Rco - Center).cross((Aco - Center)).normalized()
    X = Z.cross((Aco - Center)).normalized()
    Y = Z.cross(X).normalized()

    Mtx = Matrix((X, Y, Z)).to_4x4().transposed()
    Mtx.translation = Center

    bpy.ops.mesh.primitive_plane_add(size=Dim)
    OcclusalPlane = bpy.context.object
    name = "Occlusal_Plane"
    OcclusalPlane.name = name
    OcclusalPlane.data.name = f"{name}_Mesh"
    OcclusalPlane.matrix_world = Mtx

    matName = f"{name}_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.diffuse_color = color
    mat.use_nodes = False
    OcclusalPlane.active_material = mat
    if subdiv:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=50)
        bpy.ops.object.mode_set(mode="OBJECT")

    return OcclusalPlane


##############################################
def TriPlanes_Point_Intersect(P1, P2, P3, CrossLenght):

    P1N = P1.matrix_world.to_3x3() @ P1.data.polygons[0].normal
    P2N = P2.matrix_world.to_3x3() @ P2.data.polygons[0].normal
    P3N = P3.matrix_world.to_3x3() @ P3.data.polygons[0].normal

    Condition = np.dot(np.array(P1N), np.cross(np.array(P2N), np.array(P3N))) != 0

    C1, C2, C3 = P1.location, P2.location, P3.location

    F1 = sum(list(P1.location * P1N))
    F2 = sum(list(P2.location * P2N))
    F3 = sum(list(P3.location * P3N))

    # print(Matrix((P1N,P2N,P3N)))
    if Condition:

        P_Intersect = Matrix((P1N, P2N, P3N)).inverted() @ Vector((F1, F2, F3))
        P1P2_Vec = (
            Vector(np.cross(np.array(P1N), np.array(P2N))).normalized() * CrossLenght
        )
        P2P3_Vec = (
            Vector(np.cross(np.array(P2N), np.array(P3N))).normalized() * CrossLenght
        )
        P1P3_Vec = (
            Vector(np.cross(np.array(P1N), np.array(P3N))).normalized() * CrossLenght
        )

        P1P2 = [P_Intersect + P1P2_Vec, P_Intersect - P1P2_Vec]
        P2P3 = [P_Intersect + P2P3_Vec, P_Intersect - P2P3_Vec]
        P1P3 = [P_Intersect + P1P3_Vec, P_Intersect - P1P3_Vec]

        return P_Intersect, P1P2, P2P3, P1P3
    else:
        return None


###########################################################
def AddPlaneMesh(DimX, DimY, Name):
    x = DimX / 2
    y = DimY / 2
    verts = [(-x, -y, 0.0), (x, -y, 0.0), (-x, y, 0.0), (x, y, 0.0)]
    faces = [(0, 1, 3, 2)]
    mesh_data = bpy.data.meshes.new(f"{Name}_mesh")
    mesh_data.from_pydata(verts, [], faces)
    uvs = mesh_data.uv_layers.new(name=f"{Name}_uv")
    # Returns True if any invalid geometry was removed.
    corrections = mesh_data.validate(verbose=True, clean_customdata=True)
    # Load BMesh with mesh data.
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update(calc_edges=True, calc_edges_loose=True)

    return mesh_data


def AddPlaneObject(Name, mesh, CollName):
    Plane_obj = bpy.data.objects.new(Name, mesh)
    bpy.context.scene.collection.objects.link(Plane_obj)
    return Plane_obj


def MoveToCollection(obj, CollName):

    OldColl = obj.users_collection  # list of all collection the obj is in
    if OldColl:
        for Coll in OldColl:
            if Coll:  # unlink from all  precedent obj collections
                Coll.objects.unlink(obj)
    NewColl = bpy.data.collections.get(CollName)
    if not NewColl:
        NewColl = bpy.data.collections.new(CollName)
        bpy.context.scene.collection.children.link(NewColl)
    if not obj in NewColl.objects[:]:
        NewColl.objects.link(obj)  # link obj to scene

    return NewColl


@persistent
def on_voxel_vizualisation_object_select(scene):
    voxel_viz_obj = bpy.context.object
    if voxel_viz_obj and voxel_viz_obj.get("voxel_node_name"):
        voxel_node_name = voxel_viz_obj["voxel_node_name"]
        if voxel_node_name and bpy.data.node_groups.get(voxel_node_name):
            voxel_node_group = bpy.data.node_groups[voxel_node_name]
            voxel_node_group.nodes["Low_Treshold"].outputs[0].default_value = scene.ODENT_Props.ThresholdMin
            # scene.ODENT_Props.ThresholdMin = int(voxel_node_group.nodes["Low_Treshold"].outputs[0].default_value)
            # scene.ODENT_Props.TresholdMax = int(voxel_node_group.nodes["High_Treshold"].outputs[0].default_value)


def BackupArray_To_3DSitkImage(DcmInfo):
    BackupList = eval(DcmInfo["BackupArray"])
    BackupArray = np.array(BackupList)
    Sp, Origin, Direction = DcmInfo["Spacing"], DcmInfo["Origin"], DcmInfo["Direction"]
    img3D = sitk.GetImageFromArray(BackupArray)
    img3D.SetDirection(Direction)
    img3D.SetSpacing(Sp)
    img3D.SetOrigin(Origin)
    return img3D


def Scene_Settings(s3d=None):
    scn = bpy.context.scene
    if not s3d:
        a3d, s3d, r3d = CtxOverride(bpy.context)
    # Set World Shader node :
    world_nodes = bpy.data.worlds["World"].node_tree.nodes
    world_nodes["Background"].inputs[0].default_value = [0.99, 1.0, 0.83, 1.0]
    world_nodes["Background"].inputs[1].default_value = 1.0
    # ""
    # 3DView Shading Methode : in {'WIREFRAME', 'SOLID', 'MATERIAL', 'RENDERED'}
    s3d.shading.type = "MATERIAL"
    s3d.shading.use_scene_lights = True
    s3d.shading.use_scene_world = False
    s3d.shading.studio_light = "forest.exr"
    # s3d.shading.studiolight_rotate_z = -75
    s3d.shading.studiolight_intensity = 1.0
    s3d.shading.studiolight_background_alpha = 0.0
    s3d.shading.studiolight_background_blur = 1.0
    # s3d.shading.render_pass = "COMBINED"

    # 'RENDERED' Shading Light method :
    s3d.shading.use_scene_lights_render = False
    s3d.shading.use_scene_world_render = True

    # ""
    s3d.shading.type = "SOLID"
    s3d.shading.show_specular_highlight = False
    # space3D.shading.background_type = 'WORLD'
    s3d.shading.color_type = "TEXTURE"
    s3d.shading.light = "STUDIO"
    s3d.shading.studio_light = "paint.sl"
    s3d.shading.show_cavity = True
    s3d.shading.curvature_ridge_factor = 0.5
    s3d.shading.curvature_valley_factor = 0.5

    ########################################################
    # EEVEE settings :
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.use_raytracing = True

    scn.eevee.use_gtao = True
    scn.eevee.gtao_distance = 50  # ambient occlusion distance
    scn.eevee.gtao_quality = 0.0 
    scn.eevee.taa_samples = 128
    scn.eevee.taa_render_samples = 128
    
    
    scn.display_settings.display_device = "sRGB"# "Non-Color"
    scn.view_settings.look = "None"
    scn.view_settings.view_transform = "Raw"

    scn.eevee.ray_tracing_options.use_denoise = True
    scn.eevee.ray_tracing_options.trace_max_roughness = 0.0
    scn.eevee.use_fast_gi = True
    scn.eevee.fast_gi_distance = 50
    scn.eevee.fast_gi_quality = 1.0
    scn.eevee.fast_gi_thickness_near = 5

    scn.view_settings.exposure = 0.0
    scn.view_settings.gamma = 1.0

    scn.unit_settings.scale_length = 0.001
    scn.unit_settings.length_unit = "MILLIMETERS"

    # set view curve mapping :
    scn.view_settings.use_curve_mapping = True
    cm = scn.view_settings.curve_mapping
    set_curve_mapping(OdentConstants.COLOR_CURVE_MAP, cm)

    


def set_curve_mapping(info, cm):

    cm.initialize()
    cm.update()

    curve = cm.curves[-1]
    for k, v in info.items():
        p = curve.points.new(0, 0)
        p.location = v
        cm.update()


#################################################################################################
# Add Slices :
#################################################################################################
####################################################################


def AddSlices(Preffix, DcmInfo, SlicesDir):

    slice_names = ["1_AXIAL_SLICE", "2_CORONAL_SLICE", "3_SAGITTAL_SLICE"]
    Sp, Sz, Origin, Direction, VC = (
        DcmInfo["RenderSp"],
        DcmInfo["RenderSz"],
        DcmInfo["Origin"],
        DcmInfo["Direction"],
        DcmInfo["VolumeCenter"],
    )

    DimX, DimY, DimZ = (Sz[0] * Sp[0], Sz[1] * Sp[1], Sz[2] * Sp[2])
    slice_planes = []
    cams = []

    for s_name in slice_names:
        bpy.ops.mesh.primitive_plane_add()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=100)
        bpy.ops.object.mode_set(mode="OBJECT")
        Plane = bpy.context.object
        Plane.matrix_world = Matrix.Identity(4)
        Plane.name = s_name
        Plane.data.name = f"{s_name}_mesh"
        Plane["odent_type"] = "slice_plane"
        Plane.rotation_mode = "XYZ"
        MoveToCollection(obj=Plane, CollName="SLICES")
        # The ability to select a plane is unnecessary. But it creates additional complications with pointer selection.
        Plane.hide_select = True
        slice_planes.append(Plane)

    ODENT_SliceUpdate(bpy.context.scene)
    ######################################################
    SlicesShader = "VGS_Dakir_Slices"
    # VGS = bpy.data.node_groups.get(f"{Preffix}_{SlicesShader}")
    VGS = bpy.data.node_groups.get(SlicesShader)

    if not VGS:
        filepath = join(OdentConstants.DATA_BLEND_FILE, "NodeTree", SlicesShader)
        directory = join(OdentConstants.DATA_BLEND_FILE, "NodeTree")
        filename = SlicesShader
        bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)
        VGS = bpy.data.node_groups.get(SlicesShader)
        # VGS.name = f"{Preffix}_{SlicesShader}"
        # VGS = bpy.data.node_groups.get(f"{Preffix}_{SlicesShader}")

    for p in slice_planes:

        p.dimensions = Vector((DimX, DimY, 0.0))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        Cam = Add_Cam_To_Plane(p, CamDistance=100, ClipOffset=OdentConstants.CAM_CLIP_OFFSET)
        cams.append(Cam)
        MoveToCollection(obj=Cam, CollName="SLICES")

        if "CORONAL" in p.name:
            rotation_euler = Euler((pi / 2, 0.0, 0.0), "XYZ")
            mat = rotation_euler.to_matrix().to_4x4()
            p.matrix_world = mat @ p.matrix_world
            Cam.matrix_world = mat @ Cam.matrix_world

        elif "SAGITTAL" in p.name:
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

        # bpy.ops.object.material_slot_add()
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

        ImageName = f"{p.name}.png"
        ImagePath = join(SlicesDir, ImageName)
        BlenderImage = bpy.data.images.get(ImageName)

        BlenderImage = bpy.data.images.get(ImageName)
        if not BlenderImage:
            bpy.data.images.load(ImagePath)
            BlenderImage = bpy.data.images.get(ImageName)

        else:
            BlenderImage.filepath = ImagePath
            BlenderImage.reload()

        TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
        ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")

        ImageTexture.image = BlenderImage
        BlenderImage.colorspace_settings.name = "Non-Color"
        # materialOutput = nodes["Material Output"]

        GroupNode = nodes.new("ShaderNodeGroup")
        GroupNode.node_tree = VGS

        links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])
        links.new(ImageTexture.outputs[0], GroupNode.inputs[0])
        links.new(GroupNode.outputs[0], materialOutput.inputs["Surface"])

    ##########################################################
    bpy.context.scene.transform_orientation_slots[0].type = "LOCAL"
    bpy.context.scene.transform_orientation_slots[1].type = "LOCAL"

    AxialPlane, CoronalPlane, SagittalPlane = slice_planes
    AxialCam, CoronalCam, SagittalCam = cams

    return AxialPlane, CoronalPlane, SagittalPlane, AxialCam, CoronalCam, SagittalCam



@persistent
def ODENT_SliceUpdate(scene):
    global _IMAGE3D
    global _PREFFIX
    ActiveObject = bpy.context.object

    ODENT_Props = scene.ODENT_Props
    Preffix = scene.get("volume_preffix")
    if Preffix:
        volumes = [
            obj
            for obj in scene.objects
            if (Preffix in obj.name and "CTVolume" in obj.name)
        ]
        _need_update = (
            ActiveObject
            and ActiveObject.get("odent_type") in ("slice_plane", "slices_pointer")
            and volumes
        )

        if _need_update:

            if not _IMAGE3D or _PREFFIX != Preffix:
                DcmInfoDict = eval(ODENT_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]
                _IMAGE3D = sitk.ReadImage(AbsPath(DcmInfo["Nrrd255Path"]))
                _PREFFIX = Preffix
            Image3D_255 = _IMAGE3D

            SlicesDir = ODENT_Props.SlicesDir
            if not exists(SlicesDir):
                SlicesDir = tempfile.mkdtemp()
                ODENT_Props.SlicesDir = SlicesDir

            CTVolume = volumes[0]
            TransformMatrix = CTVolume.matrix_world

            #########################################
            #########################################

            # Image3D_255 = sitk.ReadImage(ImageData)
            Sp = Spacing = Image3D_255.GetSpacing()
            Sz = Size = Image3D_255.GetSize()
            Ortho_Origin = -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
            Image3D_255.SetOrigin(Ortho_Origin)
            Image3D_255.SetDirection(np.identity(3).flatten())

            # Output Parameters :
            Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
            Out_Direction = Vector(np.identity(3).flatten())
            Out_Size = (Sz[0], Sz[1], 1)
            Out_Spacing = Sp

            Planes = [
                obj for obj in scene.objects if obj.get("odent_type") == "slice_plane"
            ]
            for Plane in Planes:
                ImageName = f"{Plane.name}.png"
                ImagePath = join(SlicesDir, ImageName)

                ######################################
                # Get Plane Orientation and location :
                PlanMatrix = TransformMatrix.inverted() @ Plane.matrix_world
                Rot = PlanMatrix.to_euler()
                Trans = PlanMatrix.translation
                Rvec = (Rot.x, Rot.y, Rot.z)
                Tvec = Trans

                ##########################################
                # Euler3DTransform :
                Euler3D = sitk.Euler3DTransform()
                Euler3D.SetCenter((0, 0, 0))
                Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
                Euler3D.SetTranslation(Tvec)
                Euler3D.ComputeZYXOn()
                #########################################

                Image2D = sitk.Resample(
                    Image3D_255,
                    Out_Size,
                    Euler3D,
                    sitk.sitkLinear,
                    Out_Origin,
                    Out_Spacing,
                    Out_Direction,
                    0,
                )

                #############################################
                # Write Image 1rst solution:
                Array = sitk.GetArrayFromImage(Image2D)
                Array = Array.reshape(Array.shape[1], Array.shape[2])

                Array = np.flipud(Array)
                cv2.imwrite(ImagePath, Array)

                #############################################
                # Update Blender Image data :
                BlenderImage = bpy.data.images.get(ImageName)
                if not BlenderImage:
                    bpy.data.images.load(ImagePath)
                    BlenderImage = bpy.data.images.get(ImageName)

                else:
                    BlenderImage.filepath = ImagePath
                    BlenderImage.reload()


####################################################################


def Add_Cam_To_Plane(Plane, CamDistance, ClipOffset):
    a3d, s3d, r3d = CtxOverride(bpy.context)
    with bpy.context.temp_override(area=a3d, space_data=s3d, region=r3d):
        bpy.ops.view3d.view_selected()
    bpy.ops.object.camera_add()
    Cam = bpy.context.object
    Cam.name = f"{Plane.name}_CAM"
    Cam.data.name = f"{Plane.name}_CAM_data"
    Cam["odent_type"] = "slice_cam"
    Cam.data.type = "ORTHO"
    Cam.data.ortho_scale = max(Plane.dimensions) * 1.1
    Cam.data.display_size = 10

    transform = Matrix.Identity(4)
    transform.translation = Vector((0, 0, CamDistance))

    Cam.matrix_world = transform

    Cam.data.clip_start = CamDistance - ClipOffset
    Cam.data.clip_end = CamDistance + ClipOffset

    Cam.hide_set(True)
    # Cam.select_set(False)
    return Cam


#############################################################################
# SimpleITK vtk Image to Mesh Functions :
#############################################################################

def vtkWindowedSincPolyDataFilter(q, mesh, Iterations, step, start, finish):
    def VTK_Terminal_progress(caller, event):
        ProgRatio = round(float(caller.GetProgress()), 2)
        # q.put(
        #     [
        #         "loop",
        #         f"PROGRESS : {step}...",
        #         "",
        #         start,
        #         finish,
        #         ProgRatio,
        #     ]
        # )

    SmoothFilter = vtk.vtkWindowedSincPolyDataFilter()
    SmoothFilter.SetInputData(mesh)
    SmoothFilter.SetNumberOfIterations(Iterations)
    SmoothFilter.BoundarySmoothingOff()
    SmoothFilter.FeatureEdgeSmoothingOn()
    SmoothFilter.SetFeatureAngle(60)
    SmoothFilter.SetPassBand(0.01)
    SmoothFilter.NonManifoldSmoothingOn()
    SmoothFilter.NormalizeCoordinatesOn()
    SmoothFilter.AddObserver(ProgEvent, VTK_Terminal_progress)
    SmoothFilter.Update()
    mesh.DeepCopy(SmoothFilter.GetOutput())
    return mesh

def ResizeImage(sitkImage, target_spacing=0.1):
    image = sitkImage
    Sz = image.GetSize()
    Sp = image.GetSpacing()
    if Sp[0] == Sp[1] == Sp[2] == target_spacing:
        ResizedImage = image
        return ResizedImage, Sz, Sp
    new_spacing = [target_spacing] * 3

    Ratio = [Sp[0] / target_spacing, Sp[1] / target_spacing, Sp[2] / target_spacing]
    new_size = [int(Sz[0] * Ratio[0]), int(Sz[1] * Ratio[1]), int(Sz[2] * Ratio[2])]

    ResizedImage = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
    )
    return ResizedImage, new_size, new_spacing

def VTKprogress(caller, event):
    pourcentage = int(caller.GetProgress() * 100)
    calldata = str(int(caller.GetProgress() * 100)) + " %"
    # print(calldata)
    sys.stdout.write(f"\r {calldata}")
    sys.stdout.flush()
    progress_bar(pourcentage, Delay=1)

def TerminalProgressBar(
    q,
    counter_start,
    iter=100,
    maxfill=20,
    symb1="\u2588",
    symb2="\u2502",
    periode=10,
):

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="cp65001")
        # cmd = "chcp 65001 & set PYTHONIOENCODING=utf-8"
        # subprocess.call(cmd, shell=True)

    print("\n")

    while True:
        if not q.empty():
            signal = q.get()

            if "End" in signal[0]:
                finish = Tcounter()
                line = f"{symb1*maxfill}  100% Finished.------Total Time : {round(finish-counter_start,2)}"
                # clear sys.stdout line and return to line start:
                # sys.stdout.write("\r")
                # sys.stdout.write(" " * 100)
                # sys.stdout.flush()
                # sys.stdout.write("\r")
                # write line :
                sys.stdout.write("\r" + " " * 80 + "\r" + line)  # f"{Char}"*i*2
                sys.stdout.flush()
                break

            if "GuessTime" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, periode = signal
                for i in range(iter):

                    if q.empty():

                        ratio = start + (((i + 1) / iter) * (finish - start))
                        pourcentage = int(ratio * 100)
                        symb1_fill = int(ratio * maxfill)
                        symb2_fill = int(maxfill - symb1_fill)
                        line = f"{symb1*symb1_fill}{symb2*symb2_fill}  {pourcentage}% {Uptxt}"
                        # clear sys.stdout line and return to line start:
                        # sys.stdout.write("\r"+" " * 80)
                        # sys.stdout.flush()
                        # write line :
                        sys.stdout.write("\r" + " " * 80 + "\r" + line)  # f"{Char}"*i*2
                        sys.stdout.flush()
                        sleep(periode / iter)
                    else:
                        break

            if "loop" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, progFloat = signal
                ratio = start + (progFloat * (finish - start))
                pourcentage = int(ratio * 100)
                symb1_fill = int(ratio * maxfill)
                symb2_fill = int(maxfill - symb1_fill)
                line = f"{symb1*symb1_fill}{symb2*symb2_fill}  {pourcentage}% {Uptxt}"
                # clear sys.stdout line and return to line start:
                # sys.stdout.write("\r")
                # sys.stdout.write(" " * 100)
                # sys.stdout.flush()
                # sys.stdout.write("\r")
                # write line :
                sys.stdout.write("\r" + " " * 80 + "\r" + line)  # f"{Char}"*i*2
                sys.stdout.flush()

        else:
            sleep(0.1)

def sitkTovtk(sitkImage):
    """Convert sitk image to a VTK image"""
    sitkArray = sitk.GetArrayFromImage(sitkImage)  # .astype(np.uint8)
    vtkImage = vtk.vtkImageData()

    Sp = Spacing = sitkImage.GetSpacing()
    Sz = Size = sitkImage.GetSize()

    vtkImage.SetDimensions(Sz)
    vtkImage.SetSpacing(Sp)
    vtkImage.SetOrigin(0, 0, 0)
    vtkImage.SetDirectionMatrix(1, 0, 0, 0, 1, 0, 0, 0, 1)
    vtkImage.SetExtent(0, Sz[0] - 1, 0, Sz[1] - 1, 0, Sz[2] - 1)

    VtkArray = numpy_support.numpy_to_vtk(
        sitkArray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_INT
    )
    VtkArray.SetNumberOfComponents(1)
    vtkImage.GetPointData().SetScalars(VtkArray)

    vtkImage.Modified()
    return vtkImage

def vtk_MC_Func(vtkImage, Treshold):
    MCFilter = vtk.vtkMarchingCubes()
    MCFilter.ComputeNormalsOn()
    MCFilter.SetValue(0, Treshold)
    MCFilter.SetInputData(vtkImage)
    MCFilter.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(MCFilter.GetOutput())
    return mesh

def vtkMeshReduction(q, mesh, reduction, step, start, finish):
    """Reduce a mesh using VTK's vtkQuadricDecimation filter."""

    def VTK_Terminal_progress(caller, event):
        ProgRatio = round(float(caller.GetProgress()), 2)
        # q.put(
        #     [
        #         "loop",
        #         f"PROGRESS : {step}...",
        #         "",
        #         start,
        #         finish,
        #         ProgRatio,
        #     ]
        # )

    decimatFilter = vtk.vtkQuadricDecimation()
    decimatFilter.SetInputData(mesh)
    decimatFilter.SetTargetReduction(reduction)

    decimatFilter.AddObserver(ProgEvent, VTK_Terminal_progress)
    decimatFilter.Update()

    mesh.DeepCopy(decimatFilter.GetOutput())
    return mesh

def vtkSmoothMesh(q, mesh, Iterations, step, start, finish):
    """Smooth a mesh using VTK's vtkSmoothPolyData filter."""

    def VTK_Terminal_progress(caller, event):
        ProgRatio = round(float(caller.GetProgress()), 2)
        # q.put(
        #     [
        #         "loop",
        #         f"PROGRESS : {step}...",
        #         "",
        #         start,
        #         finish,
        #         ProgRatio,
        #     ]
        # )

    SmoothFilter = vtk.vtkSmoothPolyDataFilter()
    SmoothFilter.SetInputData(mesh)
    SmoothFilter.SetNumberOfIterations(int(Iterations))
    SmoothFilter.SetFeatureAngle(45)
    SmoothFilter.SetRelaxationFactor(0.05)
    SmoothFilter.AddObserver(ProgEvent, VTK_Terminal_progress)
    SmoothFilter.Update()
    mesh.DeepCopy(SmoothFilter.GetOutput())
    return mesh

def vtkTransformMesh(mesh, Matrix):
    """Transform a mesh using VTK's vtkTransformPolyData filter."""

    Transform = vtk.vtkTransform()
    Transform.SetMatrix(Matrix)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(Transform)
    transformFilter.Update()
    mesh.DeepCopy(transformFilter.GetOutput())
    return mesh

def vtkfillholes(mesh, size):
    FillHolesFilter = vtk.vtkFillHolesFilter()
    FillHolesFilter.SetInputData(mesh)
    FillHolesFilter.SetHoleSize(size)
    FillHolesFilter.Update()
    mesh.DeepCopy(FillHolesFilter.GetOutput())
    return mesh

def vtkCleanMesh(mesh, connectivityFilter=False):
    """Clean a mesh using VTK's CleanPolyData filter."""

    ConnectFilter = vtk.vtkPolyDataConnectivityFilter()
    CleanFilter = vtk.vtkCleanPolyData()

    if connectivityFilter:

        ConnectFilter.SetInputData(mesh)
        ConnectFilter.SetExtractionModeToLargestRegion()
        CleanFilter.SetInputConnection(ConnectFilter.GetOutputPort())

    else:
        CleanFilter.SetInputData(mesh)

    CleanFilter.Update()
    mesh.DeepCopy(CleanFilter.GetOutput())
    return mesh

def sitkToContourArray(sitkImage, HuMin, HuMax, Wmin, Wmax, Thikness):
    """Convert sitk image to a VTK image"""

    def HuTo255(Hu, Wmin, Wmax):
        V255 = ((Hu - Wmin) / (Wmax - Wmin)) * 255
        return V255

    Image3D_255 = sitk.Cast(
        sitk.IntensityWindowing(
            sitkImage,
            windowMinimum=Wmin,
            windowMaximum=Wmax,
            outputMinimum=0.0,
            outputMaximum=255.0,
        ),
        sitk.sitkUInt8,
    )
    Array = sitk.GetArrayFromImage(Image3D_255)
    ContourArray255 = Array.copy()
    for i in range(ContourArray255.shape[0]):
        Slice = ContourArray255[i, :, :]
        ret, binary = cv2.threshold(
            Slice,
            HuTo255(HuMin, Wmin, Wmax),
            HuTo255(HuMax, Wmin, Wmax),
            cv2.THRESH_BINARY,
        )
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        SliceContour = np.ones(binary.shape, dtype="uint8")
        cv2.drawContours(SliceContour, contours, -1, 255, Thikness)

        ContourArray255[i, :, :] = SliceContour

    return ContourArray255

def vtkContourFilter(vtkImage, isovalue=0.0):
    """Extract an isosurface from a volume."""

    ContourFilter = vtk.vtkContourFilter()
    ContourFilter.SetInputData(vtkImage)
    ContourFilter.SetValue(0, isovalue)
    ContourFilter.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(ContourFilter.GetOutput())
    return mesh

def GuessTimeLoopFunc(signal, q):
    _, Uptxt, Lowtxt, start, finish, periode = signal
    i = 0
    iterations = 10
    while i < iterations and q.empty():
        ProgRatio = start + (((i + 1) / iterations) * (finish - start))
        q.put(
            [
                "loop",
                Uptxt,
                "",
                start,
                finish,
                ProgRatio,
            ]
        )
        sleep(periode / iterations)
        i += 1

def CV2_progress_bar(q, iter=100):
    while True:
        if not q.empty():
            signal = q.get()

            if "End" in signal[0]:
                pourcentage = 100
                Uptxt = "Finished."
                progress_bar(pourcentage, Uptxt)
                break
            if "GuessTime" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, periode = signal
                i = 0
                iterations = 10
                Delay = periode / iterations
                while i < iterations:
                    if q.empty():
                        ProgRatio = start + (
                            round(((i + 1) / iterations), 2) * (finish - start)
                        )
                        pourcentage = int(ProgRatio * 100)
                        progress_bar(
                            pourcentage, Uptxt, Delay=int(Delay * 1000)
                        )  # , Delay = int(Delay*1000)
                        sleep(Delay)
                        i += 1
                    else:
                        break
                # t = threading.Thread(target=GuessTimeLoopFunc, args=[signal, q], daemon=True)
                # t.start()
                # t.join()
                # while i < iterations and q.empty() :
                #     ratio = start + (((i + 1) / iter) * (finish - start))
                #     pourcentage = int(ratio * 100)
                #     progress_bar(pourcentage, Uptxt)
                #     sleep(periode / iter)

                # iter = 5
                # _, Uptxt, Lowtxt, start, finish, periode = signal
                # for i in range(iter):

                #     if q.empty():

                #         ratio = start + (((i + 1) / iter) * (finish - start))
                #         pourcentage = int(ratio * 100)
                #         progress_bar(pourcentage, Uptxt)
                #         sleep(periode / iter)
                #     else:
                #         break

            if "loop" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, progFloat = signal
                ratio = start + (progFloat * (finish - start))
                pourcentage = int(ratio * 100)
                progress_bar(pourcentage, Uptxt)

        else:
            sleep(0.01)

def progress_bar(pourcentage, Uptxt, Lowtxt="", Title="ODENT", Delay=1):

    X, Y = WindowWidth, WindowHeight = (500, 100)
    BackGround = np.ones((Y, X, 3), dtype=np.uint8) * 255
    # Progress bar Parameters :
    maxFill = X - 70
    minFill = 40
    barColor = (50, 200, 0)
    BarHeight = 20
    barUp = Y - 60
    barBottom = barUp + BarHeight
    # Text :
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThikness = 1
    fontColor = (0, 0, 0)
    lineStyle = cv2.LINE_AA

    chunk = (maxFill - 40) / 100

    img = BackGround.copy()
    fill = minFill + int(pourcentage * chunk)
    img[barUp:barBottom, minFill:fill] = barColor

    img = cv2.putText(
        img,
        f"{pourcentage}%",
        (maxFill + 10, barBottom - 8),
        # (fill + 10, barBottom - 10),
        font,
        fontScale,
        fontColor,
        fontThikness,
        lineStyle,
    )

    img = cv2.putText(
        img,
        Uptxt,
        (minFill, barUp - 10),
        font,
        fontScale,
        fontColor,
        fontThikness,
        lineStyle,
    )
    cv2.imshow(Title, img)

    cv2.waitKey(Delay)

    if pourcentage == 100:
        img = BackGround.copy()
        img[barUp:barBottom, minFill:maxFill] = (50, 200, 0)
        img = cv2.putText(
            img,
            "100%",
            (maxFill + 10, barBottom - 8),
            font,
            fontScale,
            fontColor,
            fontThikness,
            lineStyle,
        )

        img = cv2.putText(
            img,
            Uptxt,
            (minFill, barUp - 10),
            font,
            fontScale,
            fontColor,
            fontThikness,
            lineStyle,
        )
        cv2.imshow(Title, img)
        cv2.waitKey(Delay)
        sleep(4)
        cv2.destroyAllWindows()

######################################################
# ODENT Meshes Tools Operators...........
######################################################
def AddCurveSphere(Name, Curve, i, CollName):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bezier_points = Curve.data.splines[0].bezier_points[:]
    Bpt = bezier_points[i]
    loc = Curve.matrix_world @ Bpt.co
    AddMarkupPoint(
        name=Name, color=(0, 1, 0, 1), loc=loc, Diameter=0.5, CollName=CollName
    )
    Hook = bpy.context.object
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    Hook.select_set(True)
    Curve.select_set(True)
    bpy.context.view_layer.objects.active = Curve
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    bezier_points = Curve.data.splines[0].bezier_points[:]
    Bpt = bezier_points[i]
    Bpt.select_control_point = True
    bpy.ops.object.hook_add_selob(use_bone=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    Curve.select_set(True)
    bpy.context.view_layer.objects.active = Curve

    return Hook

def CuttingCurveAdd():

    # Prepare scene settings :
    bpy.ops.transform.select_orientation(orientation="GLOBAL")
    bpy.context.scene.tool_settings.use_snap = True
    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
    bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

    # Get CuttingTarget :
    CuttingTargetName = bpy.context.scene.ODENT_Props.CuttingTargetNameProp
    CuttingTarget = bpy.data.objects[CuttingTargetName]
    # ....Add Curve ....... :
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    # Set cutting_tool name :
    CurveCutter = bpy.context.view_layer.objects.active
    CurveCutter.name = "ODENT_Curve_Cut1"
    curve = CurveCutter.data
    curve.name = "ODENT_Curve_Cut1"
    bpy.context.scene.ODENT_Props.CurveCutterNameProp = CurveCutter.name
    MoveToCollection(CurveCutter, "ODENT-4D Cutters")
    # CurveCutter settings :
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    curve.splines[0].bezier_points[-1].select_control_point = True
    bpy.ops.curve.dissolve_verts()
    B0_Point = curve.splines[0].bezier_points[0]
    B0_Point.select_control_point = True
    # bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

    bpy.context.object.data.dimensions = "3D"
    bpy.context.object.data.twist_smooth = 4
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    bpy.context.object.data.bevel_depth = 0.2

    bpy.context.object.data.bevel_resolution = 10
    bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
    bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
    # bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 1.5708
    bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
    bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
    bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

    # Add color material :
    CurveCutterMat = bpy.data.materials.get(
        "ODENT_Curve_Cut1_Mat"
    ) or bpy.data.materials.new("ODENT_Curve_Cut1_Mat")
    CurveCutterMat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
    CurveCutterMat.roughness = 0.3

    curve.materials.append(CurveCutterMat)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
    bpy.context.space_data.overlay.show_outline_selected = False

    # bpy.ops.object.modifier_add(type="SHRINKWRAP")
    # bpy.context.object.modifiers["Shrinkwrap"].target = CuttingTarget
    # bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
    # bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True

#######################################################################################
def AddTube(context, CuttingTarget):

    ODENT_Props = bpy.context.scene.ODENT_Props
    # Prepare scene settings :
    bpy.ops.transform.select_orientation(orientation="GLOBAL")
    bpy.context.scene.tool_settings.use_snap = True
    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
    bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

    # ....Add Curve ....... :
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    # Set cutting_tool name :
    TubeObject = context.view_layer.objects.active
    TubeObject.name = "ODENT_Tube"
    TubeData = TubeObject.data
    TubeData.name = "ODENT_Tube"

    # Tube settings :
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    TubeData.splines[0].bezier_points[-1].select_control_point = True
    bpy.ops.curve.dissolve_verts()
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.view3d.snap_selected_to_cursor()

    TubeData.dimensions = "3D"
    TubeData.twist_smooth = 3
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    TubeData.bevel_depth = ODENT_Props.TubeWidth
    TubeData.bevel_resolution = 10
    bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
    bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
    bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
    bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
    bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

    # Add color material :
    TubeMat = bpy.data.materials.get("ODENT_Tube_Mat") or bpy.data.materials.new(
        "ODENT_Tube_Mat"
    )
    TubeMat.diffuse_color = [0.03, 0.20, 0.14, 1.0]  # [0.1, 0.4, 1.0, 1.0]
    TubeMat.roughness = 0.3

    TubeObject.active_material = TubeMat
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
    bpy.context.space_data.overlay.show_outline_selected = False

    bpy.ops.object.modifier_add(type="SHRINKWRAP")
    bpy.context.object.modifiers["Shrinkwrap"].target = CuttingTarget
    bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
    bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True

    return TubeObject

def DeleteTubePoint(TubeObject):
    bpy.ops.object.mode_set(mode="OBJECT")

    TubeData = TubeObject.data

    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        points = TubeData.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        points = TubeData.splines[0].bezier_points[:]
        if len(points) > 1:
            bpy.ops.curve.delete(type="VERT")
            points = TubeData.splines[0].bezier_points[:]
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.ops.curve.select_all(action="DESELECT")
            points = TubeData.splines[0].bezier_points[:]
            points[-1].select_control_point = True

        bpy.ops.object.mode_set(mode="OBJECT")

    except Exception:
        pass

def ExtrudeTube(TubeObject):
    a3d, s3d, r3d = CtxOverride(bpy.context)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.extrude(mode="INIT")
    with bpy.context.temp_override(area=a3d, space_data=s3d, region=r3d):
        bpy.ops.view3d.snap_selected_to_cursor()
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    bpy.ops.curve.select_all(action="DESELECT")
    points = TubeObject.data.splines[0].bezier_points[:]
    points[-1].select_control_point = True
    bpy.ops.object.mode_set(mode="OBJECT")

#######################################################################################

def CuttingCurveAdd2():
    # Prepare scene settings :
    bpy.ops.transform.select_orientation(orientation="GLOBAL")
    bpy.context.scene.tool_settings.use_snap = True
    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
    bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

    # Get CuttingTarget :
    CuttingTargetName = bpy.context.scene.ODENT_Props.CuttingTargetNameProp
    CuttingTarget = bpy.data.objects[CuttingTargetName]
    # ....Add Curve ....... :
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    # Set cutting_tool name :
    CurveCutter = bpy.context.view_layer.objects.active
    CurveCutter.name = "ODENT_Curve_Cut2"
    curve = CurveCutter.data
    curve.name = "ODENT_Curve_Cut2"
    bpy.context.scene.ODENT_Props.CurveCutterNameProp = CurveCutter.name

    # CurveCutter settings :
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    curve.splines[0].bezier_points[-1].select_control_point = True
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
    # bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 1.5708
    bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
    bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
    bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

    # Add color material :
    CurveCutterMat = bpy.data.materials.get(
        "ODENT_Curve_Cut2_Mat"
    ) or bpy.data.materials.new("ODENT_Curve_Cut2_Mat")
    CurveCutterMat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
    CurveCutterMat.roughness = 0.3

    curve.materials.append(CurveCutterMat)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
    bpy.context.space_data.overlay.show_outline_selected = False

    bpy.ops.object.modifier_add(type="SHRINKWRAP")
    bpy.context.object.modifiers["Shrinkwrap"].target = CuttingTarget
    bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
    bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True

    MoveToCollection(CurveCutter, "ODENT-4D Cutters")

    return CurveCutter

#######################################################################################
def DeleteLastCurvePoint():
    bpy.ops.object.mode_set(mode="OBJECT")
    # Get CuttingTarget :
    CuttingTargetName = bpy.context.scene.ODENT_Props.CuttingTargetNameProp
    CuttingTarget = bpy.data.objects[CuttingTargetName]

    # Get CurveCutter :
    CurveCutterName = bpy.context.scene.ODENT_Props.CurveCutterNameProp
    CurveCutter = bpy.data.objects[CurveCutterName]
    curve = CurveCutter.data
    points = curve.splines[0].bezier_points[:]
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        points = curve.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        points = curve.splines[0].bezier_points[:]
        if len(points) > 1:

            bpy.ops.curve.delete(type="VERT")
            points = curve.splines[0].bezier_points[:]
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.ops.curve.select_all(action="DESELECT")
            points = curve.splines[0].bezier_points[:]
            points[-1].select_control_point = True

        bpy.ops.object.mode_set(mode="OBJECT")

    except Exception:
        pass

#######################################################################################
def ExtrudeCurvePointToCursor(context, event):
    a3d, s3d, r3d = CtxOverride(bpy.context)
    # Get CurveCutter :
    CurveCutterName = bpy.context.scene.ODENT_Props.CurveCutterNameProp
    CurveCutter = bpy.data.objects[CurveCutterName]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.extrude(mode="INIT")
    with bpy.context.temp_override(area=a3d, space_data=s3d, region=r3d):
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    bpy.ops.curve.select_all(action="DESELECT")
    points = CurveCutter.data.splines[0].bezier_points[:]
    points[-1].select_control_point = True
    bpy.ops.object.mode_set(mode="OBJECT")

#######################################################################################
# 1st separate method function :
def SplitSeparator(CuttingTarget):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    intersect_vgroup = CuttingTarget.vertex_groups["intersect_vgroup"]
    CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
    bpy.ops.object.vertex_group_select()

    bpy.ops.mesh.edge_split()

    # Separate by loose parts :
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.mesh.separate(type="LOOSE")

    for obj in bpy.context.visible_objects:
        if obj.data:
            if not obj.data.polygons or len(obj.data.polygons) < 5:
                bpy.data.objects.remove(obj)

#######################################################################################
# 2nd separate method function :
def IterateSeparator():

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    selected_initial = bpy.context.selected_objects[:]
    bpy.ops.object.select_all(action="DESELECT")
    # VisObj = bpy.context.visible_objects[:].copy()

    for obj in selected_initial:

        try:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            # Select intesecting vgroup + more :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="DESELECT")
            intersect_vgroup = obj.vertex_groups["intersect_vgroup"]
            obj.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_select()
            # bpy.ops.mesh.select_more()

            # Get selected unselected verts :

            mesh = obj.data
            # polys = mesh.polygons
            verts = mesh.vertices
            # Polys = mesh.polygons
            # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.object.mode_set(mode="OBJECT")
            # unselected_polys = [p.index for p in Polys if p.select == False]
            unselected_verts = [v.index for v in verts if v.select == False]

            # Hide intesecting vgroup :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.hide(unselected=False)

            # select a part :
            bpy.ops.object.mode_set(mode="OBJECT")
            verts[unselected_verts[0]].select = True
            bpy.ops.object.mode_set(mode="EDIT")

            bpy.ops.mesh.select_linked(delimit=set())
            bpy.ops.mesh.reveal()

            # ....Separate by selection.... :
            bpy.ops.mesh.separate(type="SELECTED")
            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass
    resulting_parts = PartsFilter()  # all visible objects are selected after func

    if resulting_parts == len(selected_initial):
        return False
    else:
        return True

#######################################################################################
# Filter loose parts function :
def PartsFilter():

    # Filter small parts :
    VisObj = bpy.context.visible_objects[:].copy()
    ObjToRemove = []
    for obj in VisObj:
        if not obj.data.polygons:
            ObjToRemove.append(obj)
        else:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            verts = obj.data.vertices
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode="OBJECT")
            non_manifold_verts = [v for v in verts if v.select == True]

            if len(verts) < len(non_manifold_verts) * 2:
                ObjToRemove.append(obj)

    # Remove small parts :
    for obj in ObjToRemove:
        bpy.data.objects.remove(obj)

    bpy.ops.object.select_all(action="SELECT")
    # resulting_parts = len(bpy.context.selected_objects)

    # return resulting_parts

#######################################################################################
# CurveCutter 2 functions :
#######################################################################################
def CutterPointsList(cutter, obj):

    curve = cutter.data
    CurveCoList = []
    for point in curve.splines[0].bezier_points:
        p_co_global = cutter.matrix_world @ point.co
        p_co_obj_relative = obj.matrix_world.inverted() @ p_co_global
        CurveCoList.append(p_co_obj_relative)

    return CurveCoList

def ClosestVerts(i, CurveCoList, obj):

    # initiate a KDTree :
    size = len(obj.data.vertices)
    kd = kdtree.KDTree(size)

    for v_id, v in enumerate(obj.data.vertices):
        kd.insert(v.co, v_id)

    kd.balance()
    v_co, v_id, dist = kd.find(CurveCoList[i])

    return v_id

def ClosestVertToPoint(Point, obj):

    # initiate a KDTree :
    size = len(obj.data.vertices)
    kd = kdtree.KDTree(size)

    for v_id, v in enumerate(obj.data.vertices):
        kd.insert(v.co, v_id)

    kd.balance()
    v_co, v_id, dist = kd.find(Point)

    return v_id, v_co, dist

# Add square cutter function :
def add_square_cutter(context):

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

    Model = bpy.context.view_layer.objects.active
    loc = Model.location.copy()  # get model location
    view_rotation = context.space_data.region_3d.view_rotation

    view3d_rot_matrix = (
        view_rotation.to_matrix().to_4x4()
    )  # get v3d rotation matrix 4x4

    # Add cube :
    bpy.ops.mesh.primitive_cube_add(size=120, enter_editmode=False)

    frame = bpy.context.view_layer.objects.active
    for obj in bpy.data.objects:
        if obj.name == "my_frame_cutter":
            obj.name = "my_frame_cutter_old"
    frame.name = "my_frame_cutter"

    # Reshape and align cube :

    frame.matrix_world[:3] = view3d_rot_matrix[:3]

    frame.location = loc

    bpy.context.object.display_type = "WIRE"
    bpy.context.object.scale[1] = 0.5
    bpy.context.object.scale[2] = 2

    # Subdivide cube 10 iterations 3 times :

    bpy.ops.object.select_all(action="DESELECT")
    frame.select_set(True)
    bpy.context.view_layer.objects.active = frame

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.mesh.subdivide(number_cuts=6)

    # Make cube normals consistent :

    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.select_all(action="DESELECT")

    bpy.ops.object.mode_set(mode="OBJECT")

    # Select frame :

    bpy.ops.object.select_all(action="DESELECT")
    frame.select_set(True)
    bpy.context.view_layer.objects.active = frame

    return frame

###########################################################################
# Add ODENT MultiView :
def getLocalCollIndex(collName):
    """Get the index of a collection in the current scene's collection."""
    collNames = [col.name for col in bpy.context.scene.collection.children]

    if collName in collNames:
        index = collNames.index(collName)
        return index + 1
    else:
        return None

def ODENT_MultiView_Toggle(Preffix):
    COLLS = bpy.context.view_layer.layer_collection.children
    collectionState = {col: col.hide_viewport for col in COLLS}
    for col in COLLS:
        col.hide_viewport = False

    for col in bpy.data.collections:
        col.hide_viewport = False

    WM = bpy.context.window_manager

    # Duplicate Area3D to new window :
    MainWindow = WM.windows[0]
    LayoutScreen = bpy.data.screens["Layout"]
    LayoutArea3D = [area for area in LayoutScreen.areas if area.type == "VIEW_3D"][0]

    with bpy.context.temp_override(
        window=MainWindow, screen=LayoutScreen, area=LayoutArea3D
    ):

        bpy.ops.screen.area_dupli("INVOKE_DEFAULT")

    # Get MultiView (Window, Screen, Area3D, Space3D, Region3D) and set prefferences :
    MultiView_Window = WM.windows[-1]
    MultiView_Screen = MultiView_Window.screen

    MultiView_Area3D = [
        area for area in MultiView_Screen.areas if area.type == "VIEW_3D"
    ][0]
    MultiView_Space3D = [
        space for space in MultiView_Area3D.spaces if space.type == "VIEW_3D"
    ][0]
    MultiView_Region3D = [
        reg for reg in MultiView_Area3D.regions if reg.type == "WINDOW"
    ][0]

    MultiView_Area3D.type = (
        "CONSOLE"  # change area type for update : bug dont respond to spliting
    )

    # 1rst Step : Vertical Split .

    with bpy.context.temp_override(
        window=MultiView_Window,
        screen=MultiView_Screen,
        area=MultiView_Area3D,
        space_data=MultiView_Space3D,
        region=MultiView_Region3D,
    ):
        bpy.ops.view3d.view_selected()

    bpy.ops.screen.area_split(direction="VERTICAL", factor=1 / 5)
    MultiView_Screen.areas[0].type = "OUTLINER"
    MultiView_Screen.areas[1].type = "OUTLINER"

    # 2nd Step : Horizontal Split .
    Active_Area = MultiView_Screen.areas[0]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]

    with bpy.context.temp_override(
        window=MultiView_Window,
        screen=MultiView_Screen,
        area=Active_Area,
        space_data=Active_Space,
        region=Active_Region,
    ):
        bpy.ops.screen.area_split(direction="HORIZONTAL", factor=1 / 2)
    MultiView_Screen.areas[0].type = "VIEW_3D"
    MultiView_Screen.areas[1].type = "VIEW_3D"
    MultiView_Screen.areas[2].type = "VIEW_3D"

    # 3rd Step : Vertical Split .
    Active_Area = MultiView_Screen.areas[0]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]

    with bpy.context.temp_override(
        window=MultiView_Window,
        screen=MultiView_Screen,
        area=Active_Area,
        space_data=Active_Space,
        region=Active_Region,
    ):
        bpy.ops.screen.area_split(direction="VERTICAL", factor=1 / 2)
    MultiView_Screen.areas[0].type = "OUTLINER"
    MultiView_Screen.areas[1].type = "OUTLINER"
    MultiView_Screen.areas[2].type = "OUTLINER"
    MultiView_Screen.areas[3].type = "OUTLINER"

    # 4th Step : Vertical Split .
    Active_Area = MultiView_Screen.areas[2]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]

    with bpy.context.temp_override(
        window=MultiView_Window,
        screen=MultiView_Screen,
        area=Active_Area,
        space_data=Active_Space,
        region=Active_Region,
    ):
        bpy.ops.screen.area_split(direction="VERTICAL", factor=1 / 2)
    MultiView_Screen.areas[0].type = "VIEW_3D"
    MultiView_Screen.areas[1].type = "VIEW_3D"
    MultiView_Screen.areas[2].type = "VIEW_3D"
    MultiView_Screen.areas[3].type = "VIEW_3D"
    MultiView_Screen.areas[4].type = "VIEW_3D"

    # 4th Step : Horizontal Split .
    Active_Area = MultiView_Screen.areas[1]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]

    with bpy.context.temp_override(
        window=MultiView_Window,
        screen=MultiView_Screen,
        area=Active_Area,
        space_data=Active_Space,
        region=Active_Region,
    ):
        bpy.ops.screen.area_split(direction="HORIZONTAL", factor=1 / 2)

    MultiView_Screen.areas[1].type = "OUTLINER"
    MultiView_Screen.areas[5].type = "PROPERTIES"

    # Set MultiView Areas 3D prefferences :
    # Hide local collections :
    collNames = [
        col.name
        for col in bpy.context.scene.collection.children
        if not (
            "SLICES" in col.name
            or "SLICES_POINTERS" in col.name
            or "GUIDE Components" in col.name
        )
    ]

    for i, MultiView_Area3D in enumerate(MultiView_Screen.areas):

        if MultiView_Area3D.type == "VIEW_3D":
            MultiView_Space3D = [
                space for space in MultiView_Area3D.spaces if space.type == "VIEW_3D"
            ][0]

            with bpy.context.temp_override(
                window=MultiView_Window,
                screen=MultiView_Screen,
                area=MultiView_Area3D,
                space_data=MultiView_Space3D,
            ):
                bpy.ops.wm.tool_set_by_id(name="builtin.move")
                MultiView_Space3D.use_local_collections = True
                if not i == 4:
                    for collName in collNames:
                        index = getLocalCollIndex(collName)
                        bpy.ops.object.hide_collection(
                            collection_index=index, toggle=True
                        )

            MultiView_Space3D.overlay.show_text = True
            MultiView_Space3D.show_region_ui = False
            MultiView_Space3D.show_region_toolbar = True
            MultiView_Space3D.region_3d.view_perspective = "ORTHO"
            MultiView_Space3D.show_gizmo_navigate = False
            MultiView_Space3D.show_region_tool_header = False
            MultiView_Space3D.overlay.show_floor = False
            MultiView_Space3D.overlay.show_ortho_grid = False
            MultiView_Space3D.overlay.show_relationship_lines = False
            MultiView_Space3D.overlay.show_extras = True
            MultiView_Space3D.overlay.show_bones = False
            MultiView_Space3D.overlay.show_motion_paths = False

            MultiView_Space3D.shading.type = "SOLID"
            MultiView_Space3D.shading.light = "FLAT"
            MultiView_Space3D.shading.studio_light = "outdoor.sl"
            MultiView_Space3D.shading.color_type = "TEXTURE"
            MultiView_Space3D.shading.background_type = "VIEWPORT"
            MultiView_Space3D.shading.background_color = [
                0.0,
                0.0,
                0.0,
            ]  # [0.7, 0.7, 0.7]

            MultiView_Space3D.shading.type = "MATERIAL"
            # 'Material' Shading Light method :
            MultiView_Space3D.shading.use_scene_lights = True
            MultiView_Space3D.shading.use_scene_world = False

            # 'RENDERED' Shading Light method :
            MultiView_Space3D.shading.use_scene_lights_render = False
            MultiView_Space3D.shading.use_scene_world_render = True

            MultiView_Space3D.shading.studio_light = "forest.exr"
            MultiView_Space3D.shading.studiolight_rotate_z = 0
            MultiView_Space3D.shading.studiolight_intensity = 1.5
            MultiView_Space3D.shading.studiolight_background_alpha = 0.0
            MultiView_Space3D.shading.studiolight_background_blur = 0.0

            MultiView_Space3D.shading.render_pass = "COMBINED"
            MultiView_Space3D.shading.type = "SOLID"
            MultiView_Space3D.show_region_header = False

    OUTLINER = TopLeft = MultiView_Screen.areas[1]
    PROPERTIES = DownLeft = MultiView_Screen.areas[5]
    AXIAL = TopMiddle = MultiView_Screen.areas[3]
    CORONAL = TopRight = MultiView_Screen.areas[0]
    SAGITTAL = DownRight = MultiView_Screen.areas[2]
    VIEW_3D = DownMiddle = MultiView_Screen.areas[4]

    for col in COLLS:
        col.hide_viewport = collectionState[col]

    #    TopMiddle.header_text_set("AXIAL")
    #    TopRight.header_text_set("CORONAL")
    #    DownRight.header_text_set("SAGITTAL")
    #    DownMiddle.header_text_set("3D VIEW")

    return MultiView_Window, OUTLINER, PROPERTIES, AXIAL, CORONAL, SAGITTAL, VIEW_3D

##############################################
# Vertex Paint Cutter :
def VertexPaintCut(mode):

    #######################################################################################

    # start = time.perf_counter()
    ActiveObj = bpy.context.active_object
    paint_color = bpy.data.brushes["Draw"].color
    dict_paint_color = {
        "r_channel": bpy.data.brushes["Draw"].color.r,
        "g_channel": bpy.data.brushes["Draw"].color.g,
        "b_channel": bpy.data.brushes["Draw"].color.b,
    }
    r_channel = bpy.data.brushes["Draw"].color.r
    g_channel = bpy.data.brushes["Draw"].color.g
    b_channel = bpy.data.brushes["Draw"].color.b

    list_colored_verts_indices = []
    list_colored_verts_colors = []
    dict_vid_vcolor = {}

    # get ActiveObj, hide everything but ActiveObj :
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = ActiveObj.data
    bpy.ops.object.select_all(action="DESELECT")
    ActiveObj.select_set(True)

    if len(bpy.context.visible_objects) > 1:
        bpy.ops.object.hide_view_set(unselected=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Make dictionary : key= vertex index , value = vertex color(RGB)
    for polygon in mesh.polygons:

        for v_poly_index, v_global_index in enumerate(polygon.vertices):

            col_index = polygon.loop_indices[v_poly_index]
            v_color = mesh.vertex_colors.active.data[col_index].color[:]
            dict_vid_vcolor[v_global_index] = v_color

    # calculate averrage color :
    paint_color = tuple(bpy.data.brushes["Draw"].color)
    white_color = (1, 1, 1)
    color_offset = (1 - paint_color[0], 1 - paint_color[1], 1 - paint_color[2])
    # distance = sqrt((1-paint_color[0])**2+(1-paint_color[1])**2+pow(1-paint_color[2])**2)
    factor = 0.5
    average_color = (
        paint_color[0] + factor * color_offset[0],
        paint_color[1] + factor * color_offset[1],
        paint_color[2] + factor * color_offset[2],
    )

    # Make list : collect indices of colored vertices
    for key, value in dict_vid_vcolor.items():
        if paint_color <= value[0:3] <= average_color:
            list_colored_verts_indices.append(key)
            list_colored_verts_colors.append(value)

    # select colored verts :
    for i in list_colored_verts_indices:
        mesh.vertices[i].select = True

    # remove old vertex_groups and make new one :
    bpy.ops.object.mode_set(mode="EDIT")

    for vg in ActiveObj.vertex_groups:
        if "ODENT_PaintCutter_" in vg.name:
            ActiveObj.vertex_groups.remove(vg)

    Area_vg = ActiveObj.vertex_groups.new(name="ODENT_PaintCutter_Area_vg")
    ActiveObj.vertex_groups.active_index = Area_vg.index
    bpy.ops.object.vertex_group_assign()
    bpy.ops.mesh.region_to_loop()
    Border_vg = ActiveObj.vertex_groups.new(name="ODENT_PaintCutter_Border_vg")
    bpy.ops.object.vertex_group_assign()

    # Addon_Enable(AddonName="mesh_looptools", Enable=True) deprecated
    bpy.ops.odent.looptools_relax(
        input="selected", interpolation="cubic", iterations="5", regular=True
    )

    if mode == "Cut":
        bpy.ops.mesh.loop_to_region()
        bpy.ops.odent.separate_objects(SeparateMode="Selection")

    if mode == "Make Copy (Shell)":
        bpy.ops.mesh.loop_to_region()
        # duplicate selected verts, separate and make splint shell
        bpy.ops.mesh.duplicate_move()
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        shell = bpy.context.selected_objects[1]
        bpy.ops.object.select_all(action="DESELECT")
        shell.select_set(True)
        bpy.context.view_layer.objects.active = shell

        shell.name = "Shell"
        # Add color material :
        mat = bpy.data.materials.get("ODENT_PaintCut_mat") or bpy.data.materials.new(
            "ODENT_PaintCut_mat"
        )
        mat.diffuse_color = [paint_color[0], paint_color[1], paint_color[2], 1]
        shell.active_material = mat

    if mode == "Remove Painted":
        bpy.ops.mesh.loop_to_region()
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")

    if mode == "Keep Painted":
        bpy.ops.mesh.loop_to_region()
        bpy.ops.mesh.select_all(action="INVERT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")

def CursorToVoxelPoint(Volume, CursorMove=False):

    VoxelPointCo = 0
    CTVolume = Volume
    Preffix = Volume.name[:6]
    TransformMatrix = CTVolume.matrix_world
    ODENT_Props = bpy.context.scene.ODENT_Props
    DcmInfoDict = eval(ODENT_Props.DcmInfo)
    DcmInfo = DcmInfoDict[Preffix]
    ImageData = bpy.path.abspath(DcmInfo["Nrrd255Path"])
    Treshold = ODENT_Props.Treshold
    Wmin, Wmax = DcmInfo["Wmin"], DcmInfo["Wmax"]

    Cursor = bpy.context.scene.cursor
    CursorInitMtx = Cursor.matrix.copy()

    # Get ImageData Infos :
    Image3D_255 = sitk.ReadImage(ImageData)
    Sp = Spacing = Image3D_255.GetSpacing()
    Sz = Size = Image3D_255.GetSize()
    Ortho_Origin = -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
    Image3D_255.SetOrigin(Ortho_Origin)
    Image3D_255.SetDirection(np.identity(3).flatten())

    # Cursor shift :
    Cursor_Z = Vector((CursorInitMtx[0][2], CursorInitMtx[1][2], CursorInitMtx[2][2]))
    CT = CursorTrans = -1 * (Sz[2] - 1) * Sp[2] * Cursor_Z
    CursorTransMatrix = mathutils.Matrix(
        (
            (1.0, 0.0, 0.0, CT[0]),
            (0.0, 1.0, 0.0, CT[1]),
            (0.0, 0.0, 1.0, CT[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    # Output Parameters :
    Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
    Out_Direction = Vector(np.identity(3).flatten())
    Out_Size = Sz
    Out_Spacing = Sp

    # Get Plane Orientation and location :
    MyMatrix = TransformMatrix.inverted() @ CursorTransMatrix @ CursorInitMtx
    Rot = MyMatrix.to_euler()
    Rvec = (Rot.x, Rot.y, Rot.z)
    Tvec = MyMatrix.translation

    # Euler3DTransform :
    Euler3D = sitk.Euler3DTransform()
    Euler3D.SetCenter((0, 0, 0))
    Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
    Euler3D.SetTranslation(Tvec)
    Euler3D.ComputeZYXOn()

    #########################################

    Image3D = sitk.Resample(
        Image3D_255,
        Out_Size,
        Euler3D,
        sitk.sitkLinear,
        Out_Origin,
        Out_Spacing,
        Out_Direction,
        0,
    )

    #  # Write Image :
    # Array = sitk.GetArrayFromImage(Image3D[:,:,Sz[2]-1])#Sz[2]-1
    # Flipped_Array = np.flipud(Array.reshape(Array.shape[0], Array.shape[1]))
    # cv2.imwrite(ImagePath, Flipped_Array)

    ImgArray = sitk.GetArrayFromImage(Image3D)
    Treshold255 = int(((Treshold - Wmin) / (Wmax - Wmin)) * 255)

    RayPixels = ImgArray[:, int(Sz[1] / 2), int(Sz[0] / 2)]
    ReversedRayPixels = list(reversed(list(RayPixels)))

    for i, P in enumerate(ReversedRayPixels):
        if P >= Treshold255:
            VoxelPointCo = Cursor.location - i * Sp[2] * Cursor_Z
            break

    if CursorMove and VoxelPointCo:
        bpy.context.scene.cursor.location = VoxelPointCo
    #############################################

    return VoxelPointCo

def Metaball_Splint(shell, thikness):
    #############################################################
    # Add Metaballs :

    radius = thikness * 5 / 8
    bpy.ops.object.select_all(action="DESELECT")
    shell.select_set(True)
    bpy.context.view_layer.objects.active = shell

    vcords = [shell.matrix_world @ v.co for v in shell.data.vertices]
    mball_elements_cords = [vco - vcords[0] for vco in vcords[1:]]

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

    bpy.ops.object.metaball_add(
        type="BALL", radius=radius, enter_editmode=False, location=vcords[0]
    )

    mball_obj = bpy.context.view_layer.objects.active

    mball = mball_obj.data
    mball.resolution = 0.6
    bpy.context.object.data.update_method = "FAST"

    for i in range(len(mball_elements_cords)):
        element = mball.elements.new()
        element.co = mball_elements_cords[i]
        element.radius = radius * 2

    bpy.ops.object.convert(target="MESH")

    splint = bpy.context.view_layer.objects.active
    splint.name = "ODENT_Splint"
    splint_mesh = splint.data
    splint_mesh.name = "ODENT_Splint_mesh"

    mat = bpy.data.materials.get(OdentConstants.SPLINT_MAT_NAME) or bpy.data.materials.new(
        OdentConstants.SPLINT_MAT_NAME
    )
    mat.diffuse_color = [0.0, 0.6, 0.8, 1.0]
    splint.active_material = mat
    bpy.ops.object.select_all(action="DESELECT")

    return splint

######################################################################################
# Align Utils
######################################################################################
def AddRefPoint(name, color, CollName=None):

    loc = bpy.context.scene.cursor.location
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.2, location=loc)
    RefP = bpy.context.object
    RefP.name = name
    RefP.data.name = name + "_mesh"
    if CollName:
        MoveToCollection(RefP, CollName)
    if name.startswith("B"):
        matName = "TargetRefMat"
    if name.startswith("M"):
        matName = "SourceRefMat"

    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.use_nodes = False
    mat.diffuse_color = color
    RefP.active_material = mat
    RefP.show_name = True
    return RefP

def RefPointsToTransformMatrix(TargetRefPoints, SourceRefPoints):
    # TransformMatrix = Matrix()  # identity Matrix (4x4)

    # make 2 arrays of coordinates :
    TargetArray = np.array(
        [obj.location for obj in TargetRefPoints], dtype=np.float64
    ).T
    SourceArray = np.array(
        [obj.location for obj in SourceRefPoints], dtype=np.float64
    ).T

    # Calculate centers of Target and Source RefPoints :
    TargetCenter, SourceCenter = np.mean(TargetArray, axis=1), np.mean(
        SourceArray, axis=1
    )

    # Calculate Translation :
    ###################################

    # TransMatrix_1 : Matrix(4x4) will translate center of SourceRefPoints...
    # to origine (0,0,0) location.
    TransMatrix_1 = Matrix.Translation(Vector(-SourceCenter))

    # TransMatrix_2 : Matrix(4x4) will translate center of SourceRefPoints...
    #  to the center of TargetRefPoints location.
    TransMatrix_2 = Matrix.Translation(Vector(TargetCenter))

    # Calculate Rotation :
    ###################################

    # Home Arrays will get the Centered Target and Source RefPoints around origin (0,0,0).
    HomeTargetArray, HomeSourceArray = (
        TargetArray - TargetCenter.reshape(3, 1),
        SourceArray - SourceCenter.reshape(3, 1),
    )
    # Rigid transformation via SVD of covariance matrix :
    U, S, Vt = np.linalg.svd(np.dot(HomeTargetArray, HomeSourceArray.T))

    # rotation matrix from SVD orthonormal bases and check,
    # if it is a Reflection matrix :
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0.0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)
        print(" Reflection matrix fixed ")

    RotationMatrix = Matrix(R).to_4x4()
    TransformMatrix = TransMatrix_2 @ RotationMatrix @ TransMatrix_1

    return TransformMatrix

def KdIcpPairs(SourceVcoList, TargetVcolist, VertsLimite=5000):
    start = Tcounter()
    # print("KD processing start...")
    SourceKdList, TargetKdList, DistList, SourceIndexList, TargetIndexList = (
        [],
        [],
        [],
        [],
        [],
    )
    size = len(TargetVcolist)
    kd = kdtree.KDTree(size)

    for i, Vco in enumerate(TargetVcolist):
        kd.insert(Vco, i)

    kd.balance()

    n = len(SourceVcoList)
    if n > VertsLimite:
        step = ceil(n / VertsLimite)
        SourceVcoList = SourceVcoList[::step]

    for SourceIndex, Sco in enumerate(SourceVcoList):

        Tco, TargetIndex, dist = kd.find(Sco)
        if Tco:
            if not TargetIndex in TargetIndexList:
                TargetIndexList.append(TargetIndex)
                SourceIndexList.append(SourceIndex)
                TargetKdList.append(Tco)
                SourceKdList.append(Sco)
                DistList.append(dist)
    finish = Tcounter()
    # print(f"KD total iterations : {len(SourceVcoList)}")
    # print(f"KD Index List : {len(IndexList)}")

    # print(f"KD finshed in {finish-start} secondes")
    return SourceKdList, TargetKdList, DistList, SourceIndexList, TargetIndexList

def KdRadiusVerts(obj, RefCo, radius):

    RadiusVertsIds = []
    RadiusVertsCo = []
    RadiusVertsDistance = []
    verts = obj.data.vertices
    Vcolist = [obj.matrix_world @ v.co for v in verts]
    size = len(Vcolist)
    kd = kdtree.KDTree(size)

    for i, Vco in enumerate(Vcolist):
        kd.insert(Vco, i)

    kd.balance()

    for co, index, dist in kd.find_range(RefCo, radius):

        RadiusVertsIds.append(index)
        RadiusVertsCo.append(co)
        RadiusVertsDistance.append(dist)

    return RadiusVertsIds, RadiusVertsCo, RadiusVertsDistance

def VidDictFromPoints(TargetRefPoints, SourceRefPoints, TargetObj, SourceObj, radius):
    IcpVidDict = {TargetObj: [], SourceObj: []}

    for obj in [TargetObj, SourceObj]:
        if obj == TargetObj:
            for RefTargetP in TargetRefPoints:
                RefCo = RefTargetP.location
                RadiusVertsIds, RadiusVertsCo, RadiusVertsDistance = KdRadiusVerts(
                    TargetObj, RefCo, radius
                )
                IcpVidDict[TargetObj].extend(RadiusVertsIds)
                for idx in RadiusVertsIds:
                    obj.data.vertices[idx].select = True
        if obj == SourceObj:
            for RefSourceP in SourceRefPoints:
                RefCo = RefSourceP.location
                RadiusVertsIds, RadiusVertsCo, RadiusVertsDistance = KdRadiusVerts(
                    SourceObj, RefCo, radius
                )
                IcpVidDict[SourceObj].extend(RadiusVertsIds)
                for idx in RadiusVertsIds:
                    obj.data.vertices[idx].select = True

    bpy.ops.object.select_all(action="DESELECT")
    for obj in [TargetObj, SourceObj]:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = TargetObj

    return IcpVidDict

def KdIcpPairsToTransformMatrix(TargetKdList, SourceKdList):
    # make 2 arrays of coordinates :
    TargetArray = np.array(TargetKdList, dtype=np.float64).T
    SourceArray = np.array(SourceKdList, dtype=np.float64).T

    # Calculate centers of Target and Source RefPoints :
    TargetCenter, SourceCenter = np.mean(TargetArray, axis=1), np.mean(
        SourceArray, axis=1
    )

    # Calculate Translation :
    ###################################

    # TransMatrix_1 : Matrix(4x4) will translate center of SourceRefPoints...
    # to origine (0,0,0) location.
    TransMatrix_1 = Matrix.Translation(Vector(-SourceCenter))

    # TransMatrix_2 : Matrix(4x4) will translate center of SourceRefPoints...
    #  to the center of TargetRefPoints location.
    TransMatrix_2 = Matrix.Translation(Vector(TargetCenter))

    # Calculate Rotation :
    ###################################

    # Home Arrays will get the Centered Target and Source RefPoints around origin (0,0,0).
    HomeTargetArray, HomeSourceArray = (
        TargetArray - TargetCenter.reshape(3, 1),
        SourceArray - SourceCenter.reshape(3, 1),
    )
    # Rigid transformation via SVD of covariance matrix :
    U, S, Vt = np.linalg.svd(np.dot(HomeTargetArray, HomeSourceArray.T))

    # rotation matrix from SVD orthonormal bases :
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0.0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)
        print(" Reflection fixed ")

    RotationMatrix = Matrix(R).to_4x4()
    TransformMatrix = TransMatrix_2 @ RotationMatrix @ TransMatrix_1

    return TransformMatrix

def AddVoxelPoint(
    Name="Voxel Anatomical Point",
    Color=(1.0, 0.0, 0.0, 1.0),
    Location=(0, 0, 0),
    Radius=1.2,
):
    Active_Obj = bpy.context.view_layer.objects.active
    bpy.ops.mesh.primitive_uv_sphere_add(radius=Radius, location=Location)
    Sphere = bpy.context.object
    Sphere.name = Name
    Sphere.data.name = Name

    MoveToCollection(Sphere, "VOXELS Points")

    matName = f"VOXEL_Points_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.diffuse_color = Color
    mat.use_nodes = False
    Sphere.active_material = mat
    Sphere.show_name = True
    bpy.ops.object.select_all(action="DESELECT")
    Active_Obj.select_set(True)
    bpy.context.view_layer.objects.active = Active_Obj

def CursorToVoxelPoint(Volume, CursorMove=False):

    VoxelPointCo = 0
    Preffix = Volume.name[:6]
    ODENT_Props = bpy.context.scene.ODENT_Props
    DcmInfoDict = eval(ODENT_Props.DcmInfo)
    DcmInfo = DcmInfoDict[Preffix]
    ImageData = bpy.path.abspath(DcmInfo["Nrrd255Path"])
    Treshold = ODENT_Props.ThresholdMin
    Wmin, Wmax = DcmInfo["Wmin"], DcmInfo["Wmax"]
    TransformMatrix = DcmInfo["TransformMatrix"]
    VtkTransform_4x4 = DcmInfo["VtkTransform_4x4"]

    Cursor = bpy.context.scene.cursor
    CursorInitMtx = Cursor.matrix.copy()

    # Get ImageData Infos :
    Image3D_255 = sitk.ReadImage(ImageData)
    Sp = Spacing = Image3D_255.GetSpacing()
    Sz = Size = Image3D_255.GetSize()
    Ortho_Origin = -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
    Image3D_255.SetOrigin(Ortho_Origin)
    Image3D_255.SetDirection(np.identity(3).flatten())

    # Cursor shift :
    Cursor_Z = Vector((CursorInitMtx[0][2], CursorInitMtx[1][2], CursorInitMtx[2][2]))
    CT = CursorTrans = -1 * (Sz[2] - 1) * Sp[2] * Cursor_Z
    CursorTransMatrix = mathutils.Matrix(
        (
            (1.0, 0.0, 0.0, CT[0]),
            (0.0, 1.0, 0.0, CT[1]),
            (0.0, 0.0, 1.0, CT[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    # Output Parameters :
    Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
    Out_Direction = Vector(np.identity(3).flatten())
    Out_Size = Sz
    Out_Spacing = Sp

    # Get Plane Orientation and location :
    Matrix = TransformMatrix.inverted() @ CursorTransMatrix @ CursorInitMtx
    Rot = Matrix.to_euler()
    Rvec = (Rot.x, Rot.y, Rot.z)
    Tvec = Matrix.translation

    # Euler3DTransform :
    Euler3D = sitk.Euler3DTransform()
    Euler3D.SetCenter((0, 0, 0))
    Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
    Euler3D.SetTranslation(Tvec)
    Euler3D.ComputeZYXOn()

    #########################################

    Image3D = sitk.Resample(
        Image3D_255,
        Out_Size,
        Euler3D,
        sitk.sitkLinear,
        Out_Origin,
        Out_Spacing,
        Out_Direction,
        0,
    )

    #  # Write Image :
    # Array = sitk.GetArrayFromImage(Image3D[:,:,Sz[2]-1])#Sz[2]-1
    # Flipped_Array = np.flipud(Array.reshape(Array.shape[0], Array.shape[1]))
    # cv2.imwrite(ImagePath, Flipped_Array)

    ImgArray = sitk.GetArrayFromImage(Image3D)
    Treshold255 = int(((Treshold - Wmin) / (Wmax - Wmin)) * 255)

    RayPixels = ImgArray[:, int(Sz[1] / 2), int(Sz[0] / 2)]
    ReversedRayPixels = list(reversed(list(RayPixels)))

    for i, P in enumerate(ReversedRayPixels):
        if P >= Treshold255:
            VoxelPointCo = Cursor.location - i * Sp[2] * Cursor_Z
            break

    if CursorMove and VoxelPointCo:
        bpy.context.scene.cursor.location = VoxelPointCo
    #############################################

    return VoxelPointCo

def AddHookedSegment(Points, Name, color, thikness, CollName=None):
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    Segment = bpy.context.view_layer.objects.active
    Segment.name = Name
    Segment.data.name = Name

    # Add color material :
    SegmentMat = bpy.data.materials.get(f"{Name}_Mat") or bpy.data.materials.new(
        f"{Name}_Mat"
    )
    SegmentMat.diffuse_color = color
    Segment.active_material = SegmentMat

    SegmentPoints = Segment.data.splines[0].bezier_points[:]
    SegmentPoints[0].co = Segment.matrix_world.inverted() @ Points[0].location
    SegmentPoints[1].co = Segment.matrix_world.inverted() @ Points[1].location

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.curve.handle_type_set(type="VECTOR")
    bpy.context.object.data.bevel_depth = thikness / 2

    # Hook Segment to spheres
    for i, P in enumerate(Points):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        P.select_set(True)
        Segment.select_set(True)
        bpy.context.view_layer.objects.active = Segment
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        SegmentPoints = Segment.data.splines[0].bezier_points[:]
        SegmentPoints[i].select_control_point = True
        bpy.ops.object.hook_add_selob(use_bone=False)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    Segment.hide_select = True
    MoveToCollection(Segment, CollName)

# Add Emptys
def AddEmpty(type, name, location, radius, CollName=None):
    bpy.ops.object.empty_add(type=type, radius=radius, location=location)
    Empty = bpy.context.object
    Empty.name = name
    if CollName:
        MoveToCollection(Empty, CollName)
    return Empty

###############################################################
# GPU, Blf
##########################################################
# def AddGpuPoints(PcoList, colors, Thikness):
#     import bgl # type: ignore
#     import gpu # type: ignore
#     from gpu_extras.batch import batch_for_shader # type: ignore

#     def draw(Thikness):
#         bgl.glLineWidth(Thikness)
#         shader.bind()
#         batch.draw(shader)
#         bgl.glLineWidth(1)

#     shader = gpu.shader.from_builtin("3D_SMOOTH_COLOR")
#     batch = batch_for_shader(shader, "POINTS", {"pos": PcoList, "color": colors})
#     _Handler = bpy.types.SpaceView3D.draw_handler_add(
#         draw, (Thikness,), "WINDOW", "POST_VIEW"
#     )

#     for area in bpy.context.window.screen.areas:
#         if area.type == "VIEW_3D":
#             area.tag_redraw()

#     return _Handler

# def Add_2D_BlfText(
#     Font_Path, color=[1.0, 0.1, 0.0, 1.0], horiz=20, vert=40, size=50, text="ODENT-4D"
# ):
#     import blf
#     font_id = 0

#     def draw_callback_px(self, context):

#         blf.color(font_id, color[0], color[1], color[2], color[3])
#         blf.position(font_id, horiz, vert, 0)
#         blf.size(font_id, size, 72)
#         blf.draw(font_id, text)

#     if Font_Path:
#         if os.path.exists(Font_Path):
#             font_id = blf.load(Font_Path)

#     _Handler = bpy.types.SpaceView3D.draw_handler_add(
#         draw_callback_px, (None, None), "WINDOW", "POST_PIXEL"
#     )  # 2D :'POST_PIXEL' | 3D :'POST_VIEW'

#     for area in bpy.context.window.screen.areas:
#         if area.type == "VIEW_3D":
#             area.tag_redraw()

#     return _Handler

###################################################################
def Angle(v1, v2):
    dot_product = v1.normalized().dot(v2.normalized())
    Angle = degrees(acos(dot_product))
    return Angle

def Linked_Edges_Verts(v, mesh):
    Edges = [e for e in mesh.edges if v.index in e.vertices]
    Link_Verts = [
        mesh.vertices[idx] for e in Edges for idx in e.vertices if idx != v.index
    ]
    return Edges, Link_Verts

def ShortPath2(obj, Vid_List, close=True):
    mesh = obj.data
    zipList = list(zip(Vid_List, Vid_List[1:] + [Vid_List[0]]))

    Tuples = zipList
    if not close:
        Tuples = zipList[:-1]
    LoopIds = []
    for i, t in enumerate(Tuples):
        v0, v1 = mesh.vertices[t[0]], mesh.vertices[t[1]]
        LoopIds.append(v0.index)

        while True:
            CurrentID = LoopIds[-1]

            V_current = mesh.vertices[CurrentID]
            TargetVector = v1.co - V_current.co
            edges, verts = Linked_Edges_Verts(V_current, mesh)
            if verts:
                if v1 in verts:
                    LoopIds.append(v1.index)
                    break
                else:

                    v = min(
                        [
                            (abs(Angle(v.co - V_current.co, TargetVector)), v)
                            for v in verts
                        ]
                    )[1]
                    LoopIds.append(v.index)
                    print(v.index)
            else:
                break

    return LoopIds

def ShortestPath(obj, VidList, close=True):

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.mesh.select_all(action="DESELECT")

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()

    Ids = VidList
    zipList = list(zip(Ids, Ids[1:] + [Ids[0]]))
    Ids_Tuples = zipList
    if not close:
        Ids_Tuples = zipList[:-1]

    Path = []

    for i, Ids in enumerate(Ids_Tuples):
        Path.append(Ids[0])
        bpy.ops.mesh.select_all(action="DESELECT")
        for id in Ids:
            bm.verts[id].select_set(True)
        select = [v.index for v in bm.verts if v.select]
        if len(select) > 1:
            try:
                bpy.ops.mesh.vert_connect_path()
            except:
                bpy.ops.mesh.shortest_path_select()
            bm.verts.ensure_lookup_table()

            # bpy.ops.mesh.shortest_path_select()
        select = [v.index for v in bm.verts if v.select]
        Path.extend(select)
        print(f"loop ({i}/{len(Ids_Tuples)}) processed ...")

    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for id in Path:
        me.vertices[id].select = True
    CutLine = [v.index for v in me.vertices if v.select]
    print(f"selected verts : {len(CutLine)}")

    return CutLine

def ConnectPath(obj, Ids, close=True):
    CutLine = []
    Ids_Tuples = list(zip(Ids, Ids[1:] + [Ids[0]]))

    if not close:
        Ids_Tuples = Ids_Tuples[:-1]

    good_pairs = []
    for pair in Ids_Tuples:
        if pair[0] != pair[1]:
            good_pairs.append(pair)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.mesh.select_all(action="DESELECT")
    # bpy.ops.object.mode_set(mode="OBJECT")

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    # bm = bmesh.new()   # create an empty BMesh
    # bm.from_edit_mesh(me)

    for i, pair in enumerate(good_pairs):
        bm.verts.ensure_lookup_table()
        _verts = [bm.verts[i] for i in pair]
        result = bmesh.ops.connect_vert_pair(bm, verts=_verts)
        for e in result.get("edges"):
            e.select = True
    CutLine = [v.index for v in bm.verts if v.select]

    # for i in Ids : bm.verts[i].select = True

    # bmv_list = [bm.verts[i] for i in Ids]
    # print(f"bmv_list : {len(bmv_list)}")
    # bm.verts.ensure_lookup_table()
    # bme_dict = bmesh.ops.connect_vert_pair(bm, verts=bmv_list)
    # bm.verts.ensure_lookup_table()
    # print(bme_dict)
    # for e in bme_dict.get('edges') :
    #     e.select = True

    # bm.verts.ensure_lookup_table()
    # CutLine = [v.index for v in bm.verts if v.select]

    # if close :
    #     for v in bm.verts :  v.select = False
    #     bme_close_list = bmesh.ops.connect_verts(bm, verts=[bmv_list[-1] , bmv_list[0]])
    #     for e in bme_close_list :
    #         e.select = True
    #     bm.verts.ensure_lookup_table()
    #     CutLine.extend([v.index for v in bm.verts if v.select])
    # bm.verts.ensure_lookup_table()
    bmesh.update_edit_mesh(me)
    # bm.to_mesh(me)
    bm.free()
    bpy.ops.object.mode_set(mode="OBJECT")

    return CutLine

def get_odent_workspaces():
    success = 0
    override_data = {
        "ws_main_area_3d": None,
        "ws_main_area_outliner": None,
        "ws_slicer_area_3d": None,
        "ws_slicer_area_axial": None,
        "ws_slicer_area_coronal": None,
        "ws_slicer_area_sagittal": None,
        "ws_slicer_area_outliner": None,
    }
    try :
        ws_main = bpy.data.workspaces.get(OdentConstants.MAIN_WORKSPACE_NAME)
        ws_slicer = bpy.data.workspaces.get(OdentConstants.SLICER_WORKSPACE_NAME)
        
        if ws_main and ws_slicer:
            #get main 3d area :
            scr = ws_main.screens[0]
            for a in scr.areas:
                if a.type == "VIEW_3D":
                    space_data = a.spaces.active
                    region = [r for r in a.regions if r.type == "WINDOW"][0]
                    override = {
                        "workspace": ws_main,
                        "screen": scr,
                        "area": a,
                        "space_data": space_data,
                        "region": region,
                    }
                    override_data["ws_main_area_3d"] = override
                    break
            #ws main area outliner :
            for a in scr.areas:
                if a.type == "OUTLINER":
                    space_data = a.spaces.active
                    region = [r for r in a.regions if r.type == "WINDOW"][0]
                    override = {
                        "workspace": ws_slicer,
                        "screen": scr,
                        "area": a,
                        "space_data": space_data,
                        "region": region,
                    }
                    override_data["ws_main_area_outliner"] = override
                    break
                
            #get slicer areas :
            scr = ws_slicer.screens[0]
            #ws slicer area outliner :
            for a in scr.areas:
                if a.type == "OUTLINER":
                    space_data = a.spaces.active
                    region = [r for r in a.regions if r.type == "WINDOW"][0]
                    override = {
                        "workspace": ws_slicer,
                        "screen": scr,
                        "area": a,
                        "space_data": space_data,
                        "region": region,
                    }
                    override_data["ws_slicer_area_outliner"] = override
                    break

            areas = [a for a in scr.areas if a.type == "VIEW_3D"]
            sorted_areas_by_y_co = sorted(areas, key=lambda a: a.y)
            sorted_areas_by_x_co = sorted(areas, key=lambda a: a.x)

            #get ws_slicer 3d area :
            area_3d = sorted_areas_by_y_co[0]
            space_data = area_3d.spaces.active
            region = [r for r in area_3d.regions if r.type == "WINDOW"][0]
            override = {
                "workspace": ws_slicer,
                "screen": scr,            
                "area": area_3d,
                "space_data": space_data,
                "region": region,
            }
            override_data["ws_slicer_area_3d"] = override

            
            areas = sorted_areas_by_x_co.copy()
            areas.remove(area_3d)
            #get ws_slicer area axial:
            area_axial = areas[0]
            space_data = area_axial.spaces.active
            region = [r for r in area_axial.regions if r.type == "WINDOW"][0]
            override = {
                "workspace": ws_slicer,
                "screen": scr,            
                "area": area_axial,
                "space_data": space_data,
                "region": region,
            }
            override_data["ws_slicer_area_axial"] = override
            
            #get ws_slicer area coronal:
            area_coronal = areas[1]
            space_data = area_coronal.spaces.active
            region = [r for r in area_coronal.regions if r.type == "WINDOW"][0]
            override = {
                "workspace": ws_slicer,
                "screen": scr,            
                "area": area_coronal,
                "space_data": space_data,
                "region": region,
            }
            override_data["ws_slicer_area_coronal"] = override

            #get ws_slicer area sagittal:
            area_sagittal = areas[2]
            space_data = area_sagittal.spaces.active
            region = [r for r in area_sagittal.regions if r.type == "WINDOW"][0]
            override = {
                "workspace": ws_slicer,
                "screen": scr,            
                "area": area_sagittal,
                "space_data": space_data,
                "region": region,
            }
            override_data["ws_slicer_area_sagittal"] = override
            
            success = 1
    except Exception as e:
        odent_log(f"Error in get_odent_workspaces : {e}")
    
    return success, override_data
