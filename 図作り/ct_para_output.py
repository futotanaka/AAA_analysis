# 指定したフォルダ内のmhdファイルのパラメータ範囲を統計してコメントラインに出力
import os
import SimpleITK as sitk
import numpy as np

def print_mhd_info(file_path):
    try:
        image = sitk.ReadImage(file_path)
        spacing = image.GetSpacing()
        size = image.GetSize()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        print(f"File: {file_path}")
        print(f"  Spacing:   {spacing}")
        print(f"  Size:      {size}")
        print(f"  Origin:    {origin}")
        print(f"  Direction: {direction}")
        print("-" * 50)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def scan_mhd_and_collect(folder):
    spacings = []
    sizes = []
    origins = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.mhd'):
                mhd_path = os.path.join(root, file)
                try:
                    image = sitk.ReadImage(mhd_path)
                    spacings.append(image.GetSpacing())
                    sizes.append(image.GetSize())
                    origins.append(image.GetOrigin())
                    # print_mhd_info(mhd_path) 
                except Exception as e:
                    print(f"Error reading {mhd_path}: {e}")
    return spacings, sizes, origins

def print_param_range(param_list, name):
    arr = np.array(param_list)
    print(f"{name} min : {np.min(arr, axis=0)}")
    print(f"{name} max : {np.max(arr, axis=0)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calculation parameter ranges of all CT images')
    parser.add_argument('folder', help='input files')
    args = parser.parse_args()
    spacings, sizes, origins = scan_mhd_and_collect(args.folder)

    if spacings:
        print_param_range(spacings, "Spacing")
    else:
        print("No spacing info found.")

    if sizes:
        print_param_range(sizes, "Size")
    else:
        print("No size info found.")

    if origins:
        print_param_range(origins, "Origin")
    else:
        print("No origin info found.")
