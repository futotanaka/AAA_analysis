import os
import argparse
import SimpleITK as sitk
from scipy import ndimage
import cv2
import numpy as np

def window_processing(image, level, width):
    lower = level - (width / 2)
    upper = level + (width / 2)
    windowed_image = np.clip(image, lower, upper)
    normalized_image = (windowed_image - lower) / (upper - lower)
    return normalized_image

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

def _extract_body_trunc_in_axial_slice(img, threshold):
    
    #binary_img = (img >= threshold).astype(np.uint8)
    binary_img = ((threshold <= img) & (img <= 1000)).astype(np.uint8)

    # remove bed etc.
    structure = _circle_structure(2)
    binary_img = ndimage.binary_opening(binary_img, structure)
    #binary_img = ndimage.morphology.binary_closing(binary_img, structure)    
    binary_img = ndimage.binary_fill_holes(binary_img)
    
    binarized_cnt = float(np.sum(binary_img))
    #print(binarized_cnt)
    
    structure = np.ones([3,3])
    labeled, num_labels = ndimage.label(binary_img, structure)
    
    # Extract body region
    histogram = ndimage.histogram(labeled, 1, num_labels, num_labels)
    
    sorted_idx = np.argsort(-histogram)
    extracted_label_cnt = 0
    ratio_of_extracted_ratio = 0.0
    
    for n in range(num_labels):
        
        if ratio_of_extracted_ratio > 0.9:
            labeled[labeled == sorted_idx[n] + 1] = 0
            
        else:     
            extracted_label_cnt += histogram[sorted_idx[n]]
            ratio_of_extracted_ratio = float(extracted_label_cnt) / binarized_cnt

    binary_img = (labeled > 0).astype(np.uint8)
 
    return binary_img.astype(np.uint8)

def pre_processing(volume):
    body_mask = np.zeros(volume.shape, dtype=np.uint8)
 
    threshold = -150 # [HU]
     
    for k in range(volume.shape[0]):
        img = volume[k,:,:]
        binary_img = _extract_body_trunc_in_axial_slice(img, threshold)
        body_mask[k,:,:] = binary_img
    
    volume[(body_mask != 1)] = -2048
    
    return volume

def process_directory(input_dir, output_base_dir, ch3_in=False):
    if not os.path.exists(input_dir):
        print("Directory not exists.")
        return  # Exit if the input directory does not exist
    
    # Create top-level directories for outputs
    output_dir_original = os.path.join(output_base_dir, "original")
    output_dir_stent = os.path.join(output_base_dir, "masks")
    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_stent, exist_ok=True)

    # Traverse directories and process .mhd files
    for subdir, dirs, files in os.walk(input_dir):
        original_file_path = None
        stent_file_path = None
        aorta_file_path = None
        
        # Check for the required files in the directory
        for filename in files:
            if filename.endswith('original.mhd'):
                original_file_path = os.path.join(subdir, filename)
            elif filename.endswith('stent_mask.mhd'):
                stent_file_path = os.path.join(subdir, filename)
            elif filename.endswith('vol000-label.mhd'):
                aorta_file_path = os.path.join(subdir, filename)

        if original_file_path and stent_file_path and aorta_file_path:
            try:
                itkimage_original = sitk.ReadImage(original_file_path)
                itkimage_stent = sitk.ReadImage(stent_file_path)
                itkimage_aorta = sitk.ReadImage(aorta_file_path)

                array_original = sitk.GetArrayFromImage(itkimage_original)  
                array_stent = sitk.GetArrayFromImage(itkimage_stent)    
                array_aorta = sitk.GetArrayFromImage(itkimage_aorta)       

                # Apply window_processing and pre-processing
                array_original = pre_processing(array_original)
                array_original = window_processing(array_original, 100, 400) # 100 400
                
                if ch3_in:
                    z_spacing = itkimage_original.GetSpacing()[2]
                    zoom_factor = z_spacing / 2.5
                    array_original = ndimage.zoom(array_original, (zoom_factor, 1, 1), order=1)
                    array_stent = ndimage.zoom(array_stent, (zoom_factor, 1, 1), order=0)
                    array_aorta = ndimage.zoom(array_aorta, (zoom_factor, 1, 1), order=0)
                
                # Extract the parent folder name to use in the filename
                parent_folder_name = os.path.basename(os.path.dirname(original_file_path))
                additional_folder_name = os.path.basename(os.path.dirname(os.path.dirname(original_file_path)))

                # Process each slice
                for i in range(len(array_original)):
                    if np.any(array_stent[i] != 0) or np.any(array_aorta[i] != 0) or input_dir[-4:] == "testnot":  # Check if the stent slice contains non-zero labels
                        if ch3_in and np.any(array_aorta[i-1] != 0) and np.any(array_aorta[i+1] != 0):
                            # Combine three slices (i-1, i, i+1)
                            slices_original = []
                            # slices_mask = []
                            for offset in [-1, 0, 1]:
                                idx = i + offset  # Index
                                slices_original.append(array_original[idx])
                                # slice_mask = np.zeros_like(array_aorta[idx])
                                # slice_mask[array_stent[idx] == 1] = 1
                                # slice_mask[array_aorta[idx] == 1] = 2
                                # slice_mask[(array_stent[idx] == 1) & (array_aorta[idx] == 1)] = 1
                                # slices_mask.append(slice_mask)
                            # Stack slices into a 3D array
                            combined_original = np.stack(slices_original, axis=0)
                            slice_mask = np.zeros_like(array_aorta[i])
                            slice_mask[array_stent[i] == 1] = 1
                            slice_mask[array_aorta[i] == 1] = 2
                            slice_mask[(array_stent[i] == 1) & (array_aorta[i] == 1)] = 1
                            combined_mask = slice_mask
                            # combined_mask = np.stack(slices_mask, axis=0)
                        elif not ch3_in:
                            # Single slice processing
                            combined_original = array_original[i]
                            slice_mask = np.zeros_like(array_aorta[i])
                            slice_mask[array_stent[i] == 1] = 1
                            slice_mask[array_aorta[i] == 1] = 2
                            slice_mask[(array_stent[i] == 1) & (array_aorta[i] == 1)] = 1
                            combined_mask = slice_mask
                        else:
                            continue

                        # Save to .npy files
                        output_filename_original = f"{additional_folder_name}_{parent_folder_name}_original_slice_{i}.npy"
                        output_filename_mask = f"{additional_folder_name}_{parent_folder_name}_mask_slice_{i}.npy"

                        np.save(os.path.join(output_dir_original, output_filename_original), combined_original)
                        np.save(os.path.join(output_dir_stent, output_filename_mask), combined_mask)

            except Exception as e:
                print(f"Processing error: {e}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='AAA data extraction.',
                                     add_help=True)
    parser.add_argument('input_dir', help='Path of original data')
    parser.add_argument('output_dir', help='Path of output')
    parser.add_argument('--ch3_in',
                        help='Export output mask as png file',
                        action='store_true')
    # Specify the input and output directories
    # input_dir = '/home/sunjiawei/data/evar_ct_images'
    # output_dir = '/home/sunjiawei/data/evar_ct_outputs'
    
    args = parser.parse_args()
    # Call the function
    process_directory(args.input_dir, args.output_dir, args.ch3_in)
    print("Finished.")