from scipy import ndimage
from skimage.morphology import skeletonize
# python skeleton_test.py /home/tanaka/data/prediction_files /home/tanaka/data/analysis_res
# python 241106test.py /home/sunjiawei/data/evar_ct_images/241106test /home/sunjiawei/data/evar_ct_images/241106test
import os
import argparse
import SimpleITK as sitk
from scipy import ndimage
import cv2
from scipy.spatial.distance import cdist
import mhd_io
import numpy as np
import matplotlib.pyplot as plt
import arterial_analysis
import stent_analysis
import branch_segmentation
import sys
import pandas as pd

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for stream in self.streams:
            stream.write(message)
    def flush(self):
        for stream in self.streams:
            stream.flush()

log_file = open("output.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)

def _circle_structure(radius):
    
    width = radius * 2 + 1
    y,x = np.ogrid[-radius:width-radius, -radius:width-radius]
    structure = x * x + y * y <= radius * radius

    return structure.astype(np.uint8)  

def _spherical_structure(radius):
    
    width = radius * 2 + 1
    z, y, x = np.ogrid[-radius:width-radius,
                       -radius:width-radius,
                       -radius:width-radius]
    structure = x * x + y * y + z * z <= radius * radius

    return structure.astype(np.uint8)

def window_processing(image, level, width):
    lower = level - (width / 2)
    upper = level + (width / 2)
    windowed_image = np.clip(image, lower, upper)
    normalized_image = (windowed_image - lower) / (upper - lower)
    return normalized_image

def branch_point_search_new(result, spacing, array_stent):
    branch_point = 0
    num_labels = []
    search_start = 50.0 / spacing[2] # start from 5cm below start position
    
    # Calculate start_slice and end_slice
    z_indices = np.any(array_stent != 0, axis=(1, 2))
    start_slice = np.argmax(z_indices)  # First non-zero slice
    z_indices_res = np.any(result != 0, axis=(1, 2))
    end_slice = len(z_indices_res) - 1 - np.argmax(z_indices_res[::-1])  # Last non-zero slice
    print(f"start:{start_slice} end:{end_slice}")

    for i in range(start_slice, end_slice + 1):
        img = result[i,:,:]
        labeled, num = ndimage.label(img, structure=np.ones((3, 3)))  # Use ndimage's label function with 8-connectivity
        num_labels.append(num)

    candi = []
    lens = []

    i = 0
    while i < len(num_labels):
        if num_labels[i] > 1 and i >= search_start:
            j = 1
            count_len = 1
            while i + j < len(num_labels) and num_labels[i + j] > 1:
                count_len += 1
                j += 1

            if i == 0:
                i += j
                continue

            if count_len >= 3:
                candi.append(i)
                lens.append(count_len)
            i += j
        else:
            i += 1

    max_distances = []
    for i in range(len(candi)):
        start_idx = start_slice + candi[i]
        end_idx = start_slice + candi[i] + lens[i] - 1
        start_points = np.argwhere(result[start_idx] > 0)
        end_points = np.argwhere(result[end_idx]  > 0)
        if len(start_points) > 0 and len(end_points) > 0:
                # Compute pairwise distances between start_points and end_points
                distances = cdist(start_points, end_points)
                max_distance = np.max(distances) if distances.size > 0 else 0
        else:
            max_distance = 0
        
        # If there are at least two points in start_points, compute pairwise distances among them
        if start_points.shape[0] >= 2:
            # Calculate pairwise distances between the start_points
            pairwise_distances = cdist(start_points, start_points)
            # Extract the upper triangular portion of the distance matrix (to avoid duplicate comparisons)
            upper_indices = np.triu_indices_from(pairwise_distances, k=1)
            distances = pairwise_distances[upper_indices]
            # If all pairwise distances are greater than 20 pixels, skip the current candidate
            if np.all(distances > 10):
                max_distance = 0
        max_distances.append(max_distance)
        
    if max_distances:
        max_dist_index = np.argmax(max_distances)
        branch_point = start_slice + candi[max_dist_index]
        # print(f"Branch point index: {branch_point}")
    
    for i in range(len(candi)):
        ori_index = start_slice + candi[i]
        print(f"{ori_index}-{int(max_distances[i])}-{lens[i]}", end=" ")
    print("")

    return branch_point

def branch_point_search(result):
    branch_point = 0
    num_labels = []
    
    # Calculate start_slice and end_slice
    z_indices = np.any(result != 0, axis=(1, 2))
    start_slice = np.argmax(z_indices)  # First non-zero slice
    end_slice = len(z_indices) - 1 - np.argmax(z_indices[::-1])  # Last non-zero slice
    print(f"start:{start_slice} end:{end_slice}")

    for i in range(start_slice, end_slice + 1):
        img = result[i,:,:]
        labeled, num = ndimage.label(img, structure=np.ones((3, 3)))  # Use ndimage's label function with 8-connectivity
        num_labels.append(num)

    candi = []
    lens = []

    i = 0
    while i < len(num_labels):
        if num_labels[i] > 1:
            j = 1
            count_len = 1
            while i + j < len(num_labels) and num_labels[i + j] > 1:
                count_len += 1
                j += 1

            if i == 0:
                i += j
                continue
            
            if count_len >= 3:
                candi.append(i)
                lens.append(count_len)
            i += j
        else:
            i += 1

    if candi:
        max_len_index = np.argmax(lens)
        branch_point = start_slice + candi[max_len_index]
        # print(f"Branch point index: {branch_point}")
    
    for index in candi:
        ori_index = start_slice + index
        print(f"{ori_index}", end=" ")
    print("")

    return branch_point

def branch_point_search_old(result):
    candidates = []

    # Identify candidate points and check neighbor counts vectorized
    non_zero_indices = np.argwhere(result != 0)
    for k, j, i in non_zero_indices:
        sub_volume = result[max(0, k-4):k+5, max(0, j-4):j+5, max(0, i-4):i+5]

        sub_volume_copy = sub_volume.copy()
        center_k, center_j, center_i = sub_volume.shape[0] // 2, sub_volume.shape[1] // 2, sub_volume.shape[2] // 2
        sub_volume_copy[
            center_k-1:center_k+2,
            center_j-1:center_j+2,
            center_i-1:center_i+2
        ] = 0

        labeled, num_features = ndimage.label(sub_volume_copy, structure=np.ones((3, 3, 3)))
        if num_features >= 3:
            candidates.append((k, j, i))
            
    largest_min_size = 0
    best_candidate = None
    print(candidates)
    for p in candidates:
        k,j,i = p
        img = result[k, :, :]
        result[k, :, :] = np.zeros_like(result[k])
        labeled, num_labeled = ndimage.label(result, structure=np.ones((3, 3, 3)))
        result[k] = img

        # Calculate histogram of labeled regions
        hist = np.bincount(labeled.ravel(), minlength=num_labeled + 1)
        
        # Exclude the background (index 0) from the histogram
        component_sizes = hist[1:] if len(hist) > 1 else []
        if component_sizes.any():
            # Find the smallest connected component size for this candidate point
            min_size = np.min(component_sizes)
            print(p, component_sizes)
            if min_size > largest_min_size:
                largest_min_size = min_size
                best_candidate = p
    
    return best_candidate[0]
        
def short_branch_remove(skeleton, threshold):
    result = skeleton.copy()
    if result is None or not isinstance(result, np.ndarray):
        raise ValueError("Input 'result' must be a valid NumPy array.")
    candidates = []

    # Identify candidate points and check neighbor counts vectorized
    non_zero_indices = np.argwhere(result != 0)
    for k, j, i in non_zero_indices:
        sub_volume = result[max(0, k-4):k+5, max(0, j-4):j+5, max(0, i-4):i+5]

        sub_volume_copy = sub_volume.copy()
        center_k, center_j, center_i = sub_volume.shape[0] // 2, sub_volume.shape[1] // 2, sub_volume.shape[2] // 2
        sub_volume_copy[
            center_k-1:center_k+2,
            center_j-1:center_j+2,
            center_i-1:center_i+2
        ] = 0

        labeled, num_features = ndimage.label(sub_volume_copy, structure=np.ones((3, 3, 3)))
        if num_features >= 3:
            candidates.append((k, j, i))
    # for k, j, i in non_zero_indices:
    #     sub_volume = result[max(0, k-1):k+2, max(0, j-1):j+2, max(0, i-1):i+2]
    #     count = np.sum(sub_volume != 0)
    #     if count > 3:
    #         candidates.append((k, j, i))

    # Preallocate arrays for reuse
    space_size = threshold * 2
    cutted = np.zeros((space_size * 2, space_size * 2, space_size * 2), dtype=np.int16)
    labeled = np.zeros_like(cutted, dtype=np.int16)
    # print("len candi: ", len(candidates))

    for n in range(len(candidates)):
        cutted.fill(0)
        labeled.fill(0)

        k, j, i = candidates[n]
        result[k, j, i] = 0

        # Extract a sub-volume around the candidate
        for x in range(max(0, k - space_size), min(result.shape[0], k + space_size)):
            for y in range(max(0, j - space_size), min(result.shape[1], j + space_size)):
                for z in range(max(0, i - space_size), min(result.shape[2], i + space_size)):
                    if result[x, y, z] != 0:
                        cutted[x - k + space_size, y - j + space_size, z - i + space_size] = 1
        
        if np.all(cutted == 0):
            print("Cutted all 0!!")
            continue

        # Perform labeling (connected components)
        labeled, num_labeled = ndimage.label(cutted, structure=np.ones((3, 3, 3)))
        result[k, j, i] = 1

        # Calculate histogram of labeled regions
        hist = np.bincount(labeled.ravel(), minlength=num_labeled + 1)

        # Remove small labeled regions
        for label_id in range(1, len(hist)):
            if hist[label_id] < threshold:
                for x in range(max(0, k - space_size), min(result.shape[0], k + space_size)):
                    for y in range(max(0, j - space_size), min(result.shape[1], j + space_size)):
                        for z in range(max(0, i - space_size), min(result.shape[2], i + space_size)):
                            if (labeled[x - k + space_size, y - j + space_size, z - i + space_size] == label_id):
                                result[x, y, z] = 0
    
    return result

def skeleton_analysis(array_aorta, array_stent, spacing, dimension, output_base_dir, output_name, zoom_factor,
                      array_aorta_ori, array_stent_ori, spacing_ori, dim_ori):
    
    print("Skeleton processing...")
    
    # Stent part
    num_slices = array_stent.shape[0]
    nonzero_slices = np.where(np.any(array_stent != 0, axis=(1, 2)))[0]
    first_labeled_slice = nonzero_slices[0]
    last_labeled_slice = nonzero_slices[-1]
    first_labeled_slice = max(0, first_labeled_slice - 10)
    last_labeled_slice = min(num_slices, last_labeled_slice + 10)
    
    seperated_stent = branch_segmentation.do_segment(array_stent, array_aorta,spacing,output_base_dir,output_name)
    
    skeleton_array_stent = np.zeros_like(seperated_stent, dtype=np.uint8)
    sub_array_stent = seperated_stent[first_labeled_slice:last_labeled_slice]
    # skeletonize by branches
    sub_skeleton_stent_fin = np.zeros_like(sub_array_stent)
    for label in [1, 2, 3]:
        sub_branch_stent = (sub_array_stent == label).astype(np.uint8)
        skeleton = skeletonize(sub_branch_stent)
        sub_skeleton_stent_fin[skeleton > 0] = label
    skeleton_array_stent[first_labeled_slice:last_labeled_slice] = sub_skeleton_stent_fin
    post_skeleton_stent = skeleton_array_stent
    
    # arterial part
    num_slices = array_aorta.shape[0]
    # Find the first slice with labels
    first_labeled_slice = next((i for i in range(num_slices) if np.any(array_aorta[i] != 0) or np.any(array_aorta[i] != 0)), 0)
    # Find the last slice with labels
    last_labeled_slice = next((i for i in range(first_labeled_slice, num_slices) if not np.any(array_aorta[i] != 0)), num_slices)
    first_labeled_slice = max(0, first_labeled_slice - 10)
    last_labeled_slice = min(num_slices, last_labeled_slice + 10)
    
    skeleton_array = np.zeros_like(array_aorta, dtype=np.uint8)
    sub_array_aorta = array_aorta[first_labeled_slice:last_labeled_slice]
    # distance = ndimage.distance_transform_edt(sub_array_aorta)
    # skeleton = skeletonize((distance > 5).astype(np.uint8))  
    structure = _circle_structure(5)
    structure_3d_0 = _spherical_structure(3)
    structure_element = np.zeros((3, 1, 1))
    structure_element[:, 0, 0] = 1
    structure_3d = _spherical_structure(7)
    sub_array_aorta = ndimage.binary_opening(sub_array_aorta, structure=structure_3d_0)
    sub_array_aorta = ndimage.binary_opening(sub_array_aorta, structure=structure_element)
    sub_array_aorta_eroded = np.zeros_like(sub_array_aorta, dtype=sub_array_aorta.dtype)
    for z in range(sub_array_aorta.shape[0]):
        sub_array_aorta_eroded[z] = ndimage.binary_erosion(sub_array_aorta[z], structure=structure)
    sub_array_aorta_eroded = ndimage.binary_closing(sub_array_aorta_eroded, structure=structure_3d)
    skeleton = skeletonize(sub_array_aorta_eroded)  
    skeleton = np.where(skeleton, 1, 0).astype(np.uint8)
    skeleton_array[first_labeled_slice:last_labeled_slice] = skeleton
    # Post-processing of skeleton
    # skeleton_array = ndimage.binary_closing(skeleton_array,structure=structure_3d).astype(np.uint8)
    # post_skeleton = short_branch_remove(skeleton_array, 15)
    post_skeleton_arterial = skeleton_array
    
    print("---------------------Analysis")
    # Aortic bifurcation point decision
    branch_point = branch_point_search_new(post_skeleton_arterial, spacing, array_stent)
    branch_point_ori = int(branch_point / zoom_factor)
    print(f"branch point: {branch_point}----ori: {branch_point_ori}")
    
    # Stent analysis
    print("...")
    stent_analysis.postProcessingForStent(post_skeleton_stent, array_stent, spacing, branch_point)
    # Arterial analysis
    aaa_range = arterial_analysis.post_processing_for_arterial(array_aorta,spacing,branch_point,array_stent,zoom_factor)

    # --- Bounding Box 計算 ---
    def get_bbox_metrics(array, spc, z_limits=None):
        target = array
        if z_limits is not None:
            temp = np.zeros_like(array)
            temp[int(z_limits[0]):int(z_limits[1])+1, :, :] = array[int(z_limits[0]):int(z_limits[1])+1, :, :]
            target = temp
        
        coords = np.argwhere(target > 0)
        if coords.size == 0: return 0, 0, 0
        dz = (coords[:, 0].max() - coords[:, 0].min() + 1) * spc[2]
        dy = (coords[:, 1].max() - coords[:, 1].min() + 1) * spc[1]
        dx = (coords[:, 2].max() - coords[:, 2].min() + 1) * spc[0]
        #print(coords[:, 2].min(), coords[:, 2].max(),:: coords[:, 1].min(), coords[:, 1].max(),:: coords[:, 0].min(), coords[:, 0].max())
        return round(dx, 2), round(dy, 2), round(dz, 2)

    a_x, a_y, a_z = get_bbox_metrics(array_aorta, spacing)
    s_x, s_y, s_z = get_bbox_metrics(array_stent, spacing)
    aaa_x, aaa_y, aaa_z = get_bbox_metrics(array_aorta, spacing, z_limits=aaa_range)
    
    total_len, mask = stent_analysis.AAA_part_stent_analysis(array_stent, post_skeleton_stent, aaa_range, spacing, array_stent_ori, spacing_ori, zoom_factor)
    # print(array_aorta.shape, skeleton.shape)

    # raw+mhdファイルを出力するには以下の3行を有効にする
    mhd_io.export_to_mhd_and_raw(f"{output_base_dir}/{output_name}_skeleton_arterial.mhd", post_skeleton_arterial, spacing, dimension)
    mhd_io.export_to_mhd_and_raw(f"{output_base_dir}/{output_name}_skeleton_stent.mhd", mask, spacing, dimension)
    mhd_io.export_to_mhd_and_raw(f"{output_base_dir}/{output_name}_arterial.mhd", array_aorta, spacing, dimension)

    # 今回の計算結果を辞書で返す
    return {
        "Case_ID": output_name,
        "Aorta_X_mm": a_x, "Aorta_Y_mm": a_y, "Aorta_Z_mm": a_z,
        "Stent_X_mm": s_x, "Stent_Y_mm": s_y, "Stent_Z_mm": s_z,
        "AAA_X_mm": aaa_x, "AAA_Y_mm": aaa_y, "AAA_Z_mm": aaa_z
    }

def process_directory(input_dir, output_base_dir):
    if not os.path.exists(input_dir):
        print("Directory not exists.")
        return  # Exit if the input directory does not exist

    all_results = []
    
    # Traverse directories and process .mhd files
    for subdir, dirs, files in os.walk(input_dir):
        original_file_path = None
        stent_file_path = None
        aorta_file_path = None
        prediction_file_pathes = []
        # Check for the required files in the directory
        for filename in sorted(files):
            if filename.endswith('original.mhd'):
                original_file_path = os.path.join(subdir, filename)
            elif filename.endswith('stent_mask.mhd'):
                stent_file_path = os.path.join(subdir, filename)
            elif filename.endswith('vol000-label.mhd'):
                aorta_file_path = os.path.join(subdir, filename)
            elif filename.endswith("_prediction.mhd"):
                prediction_file_pathes.append(os.path.join(subdir, filename))

        if original_file_path and stent_file_path and aorta_file_path or len(prediction_file_pathes) > 0:
            try:
                if original_file_path and stent_file_path and aorta_file_path:
                    itkimage_stent = sitk.ReadImage(stent_file_path)
                    itkimage_aorta = sitk.ReadImage(aorta_file_path)
    
                    array_stent = sitk.GetArrayFromImage(itkimage_stent) 
                    array_aorta = sitk.GetArrayFromImage(itkimage_aorta)       
                    dimension = mhd_io.get_dimension_from_mhd(aorta_file_path)
                    spacing = itkimage_aorta.GetSpacing()
                    
                    zoom_factor = spacing[2] / spacing[0]
                    # print("Interpolation...zoom factor: ", zoom_factor)
                    print("--------------------------------------------------------------------")
                    array_aorta_ori = array_aorta.copy()
                    array_stent_ori = array_stent.copy()
                    array_aorta = ndimage.zoom(array_aorta, (zoom_factor, 1, 1), order=0).astype(np.uint8)
                    array_stent = ndimage.zoom(array_stent, (zoom_factor, 1, 1), order=0).astype(np.uint8)
                    spacing_ori = spacing
                    spacing = (spacing[0], spacing[1], spacing[0])
                    dim_ori = dimension
                    dimension = (dimension[0], dimension[1], array_aorta.shape[0])
                    
                    # Extract the parent folder name to use in the filename
                    parent_folder_name = os.path.basename(os.path.dirname(original_file_path))
                    additional_folder_name = os.path.basename(os.path.dirname(os.path.dirname(original_file_path)))
                    output_name = f"{additional_folder_name}_{parent_folder_name}"
                    print(output_name, spacing, dimension)
                    skeleton_analysis(array_aorta, array_stent, spacing, dimension, output_base_dir, output_name, zoom_factor,
                                      array_aorta_ori, array_stent_ori, spacing_ori, dim_ori)
                    
                elif len(prediction_file_pathes) > 0:
                    for file_path in prediction_file_pathes:
                        itkimage_mix = sitk.ReadImage(file_path)
                        array_mix = sitk.GetArrayFromImage(itkimage_mix)
                        dimension = mhd_io.get_dimension_from_mhd(file_path)
                        spacing = itkimage_mix.GetSpacing()
                        
                        print("--------------------------------------------------------------------")
                        zoom_factor = spacing[2] / spacing[0]
                        array_aorta = (array_mix > 0).astype(np.uint8)
                        array_stent = (array_mix == 1).astype(np.uint8)
                        array_aorta_ori = array_aorta.copy()
                        array_stent_ori = array_stent.copy()
                        array_aorta = ndimage.zoom(array_aorta, (zoom_factor, 1, 1), order=0).astype(np.uint8)
                        array_stent = ndimage.zoom(array_stent, (zoom_factor, 1, 1), order=0).astype(np.uint8)
                        spacing_ori = spacing
                        spacing = (spacing[0], spacing[1], spacing[0])
                        dim_ori = dimension
                        dimension = (dimension[0], dimension[1], array_aorta.shape[0])
                        
                        # # full stent bounding box
                        # mhd_name = file_path.split("/")[-1]
                        # words = mhd_name.split("_")
                        # output_name = f"{words[0]}_{words[1]}"
                        # print(output_name, spacing, dimension)
                        # stent_mask = (array_mix == 1)
                        # z_slices = stent_mask.any(axis=(1,2))
                        # if z_slices.any():
                        #     first_z = z_slices.argmax()
                        #     last_z = len(z_slices) - 1 - z_slices[::-1].argmax()
                        #     bbox = stent_analysis.boundingBoxCal(array_mix, spacing, [first_z, last_z])
                        # else:
                        #     print("No label=1 found in the mask.")
                        
                        mhd_name = file_path.split("/")[-1]
                        words = mhd_name.split("_")
                        output_name = f"{words[0]}_{words[1]}"
                        print(output_name, spacing, dimension)
                        print("Interpolation...zoom factor: ", zoom_factor)
                        res = skeleton_analysis(array_aorta, array_stent, spacing, dimension, output_base_dir, output_name, zoom_factor,
                                          array_aorta_ori, array_stent_ori, spacing_ori, dim_ori)
                        all_results.append(res) # リストに追加
                        print(f"Case {output_name} added to summary.")

            except Exception as e:
                print(f"Processing error: {e}")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        csv_output_path = os.path.join(output_base_dir, "all_cases_bbox_summary.csv")
        summary_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        print(f"\n[Success] Final summary saved to: {csv_output_path}")
    else:
        print("No results to save.")
                
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='AAA data extraction.',
                                     add_help=True)
    parser.add_argument('input_dir', help='Path of original data')
    parser.add_argument('output_dir', help='Path of output')
    args = parser.parse_args()
    
    # Call the function
    process_directory(args.input_dir, args.output_dir)
    print("Finished.")