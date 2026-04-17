"""
Created on May 31 2023
@author: ynomura
"""

import numpy as np
import os
import SimpleITK as sitk

def get_voxel_spacing_from_mhd(mhd_file_name):

    reader = sitk.ImageFileReader()
    reader.SetFileName(mhd_file_name)
    reader.ReadImageInformation()

    return reader.GetSpacing()

def get_dimension_from_mhd(mhd_file_name):

    reader = sitk.ImageFileReader()
    reader.SetFileName(mhd_file_name)
    reader.ReadImageInformation()

    return reader.GetSize()


def get_met_type_from_numpy_type(dtype):
    if dtype == np.int8:
        return 'MET_CHAR'
    elif dtype == np.uint8:
        return 'MET_UCHAR'
    elif dtype == np.int16:
        return 'MET_SHORT'
    elif dtype == np.uint16:
        return 'MET_USHORT'
    elif dtype == np.int32:
        return 'MET_INT'
    elif dtype == np.uint32:
        return 'MET_UINT'
    elif dtype == np.float32:
        return 'MET_FLOAT'
    elif dtype == np.float64:
        return 'MET_DOUBLE'
    return 'MET_OTHER'


def export_to_mhd_and_raw(mhd_file_name, volume, voxel_spacing, with_metaheader=True):

    # set file name of raw data
    base_dir_pair = os.path.split(mhd_file_name)
    raw_file_name = os.path.splitext(base_dir_pair[1])[0] + ".raw"

    size_str = f'DimSize = {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}'
    spacing_str = f'ElementSpacing = {voxel_spacing[0]} {voxel_spacing[1]} {voxel_spacing[2]}'
    element_type_str = "ElementType = " + get_met_type_from_numpy_type(volume.dtype)

    with open(mhd_file_name, mode="w", newline="\r\n") as fp: # add newline to make pluto read 
        fp.write("ObjectType = Image\n")
        fp.write(f"NDims = {volume.ndim}\n")
        fp.write(f"{size_str}\n")
        #fp.write("TransformMatrix = 1 0 0 0 1 0 0 0 -1\n")
        #fp.write("Offset = 0 0 0\n")
        #fp.write("CenterOfRotation = 0 0 0\n")
        #fp.write("AnatomicalOrientation = RAS\n")
        fp.write(f"{element_type_str}\n")
        fp.write(f"{spacing_str}\n")
        fp.write("ElementByteOrderMSB = False\n")
        fp.write(f"ElementDataFile = {raw_file_name}\n")

    # export raw data
    raw_file_name = os.path.join(base_dir_pair[0], raw_file_name)
    volume.tofile(raw_file_name)