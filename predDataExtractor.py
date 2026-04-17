import os
import argparse
import SimpleITK as sitk
from scipy import ndimage
import mhd_io
import preprocessing
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

def process_directory(input_dir, output_base_dir):
    if not os.path.exists(input_dir):
        print("Directory not exists.")
        return  # Exit if the input directory does not exist
    
    # Create top-level directories for outputs
    output_dir_original = os.path.join(output_base_dir, "original")
    os.makedirs(output_dir_original, exist_ok=True)

    # Traverse directories and process .mhd files
    for subdir, dirs, files in os.walk(input_dir):
        original_file_path = None
        
        # Check for the required files in the directory
        for filename in files:
            if filename.endswith('original.mhd'):
                original_file_path = os.path.join(subdir, filename)

        if original_file_path:
            try:
                itkimage_original = sitk.ReadImage(original_file_path)

                array_original = sitk.GetArrayFromImage(itkimage_original)    
                voxel_spacing = mhd_io.get_voxel_spacing_from_mhd(original_file_path)

                body_mask = preprocessing.body_trunk_extraction(array_original).astype(np.uint8)

                bone_mask = preprocessing.bone_extraction(array_original, body_mask, voxel_spacing)

                landmarks = preprocessing.chest_and_iliac_slice_estimation(array_original,
                                                                        body_mask,
                                                                        bone_mask,
                                                                        voxel_spacing, False)
                # Apply window_processing and pre-processing
                array_original = pre_processing(array_original)
                array_original = window_processing(array_original, 100, 400)
                
                # Extract the parent folder name to use in the filename
                # parent_folder_name = os.path.basename(os.path.dirname(original_file_path)) # no dir 'plane_0.5' version
                # additional_folder_name = os.path.basename(os.path.dirname(os.path.dirname(original_file_path))) # no dir 'plane_0.5' version
                parent_folder_name = os.path.basename(os.path.dirname(original_file_path)) # dir 'plane_0.5' version
                additional_folder_name = os.path.basename(os.path.dirname(os.path.dirname(original_file_path))) # dir 'plane_0.5' version
                print(additional_folder_name, parent_folder_name, landmarks)
                # Process each slice
                ch3_in = False
                for i, slice_original in enumerate(array_original):
                    if landmarks[1] <= i <= landmarks[2] :  # landmarks[1] <= i <= landmarks[2]
                        if ch3_in:
                            # Combine three slices (i-1, i, i+1)
                            slices_original = []
                            for offset in [-1, 0, 1]:
                                idx = i + offset  # Index
                                slices_original.append(array_original[idx])
                            # Stack slices into a 3D array
                            slice_original = np.stack(slices_original, axis=0)

                        # Save to .npy files
                        output_filename_original = f"{additional_folder_name}_{parent_folder_name}_{voxel_spacing[2]}_original_slice_{i}.npy"

                        # Output gray image
                        # output_original = f"{parent_folder_name}_slice_{i}.png"
                        # norm_image = (slice_mask * 125).astype(np.uint8)
                        # cv2.imwrite(os.path.join(output_dir_original, output_original), norm_image)

                        np.save(os.path.join(output_dir_original, output_filename_original), slice_original)

            except Exception as e:
                print(f"Processing error: {e}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='AAA data extraction.',
                                     add_help=True)
    parser.add_argument('input_dir', help='Path of original data')
    parser.add_argument('output_dir', help='Path of output')
    
    args = parser.parse_args()
    # Call the function
    process_directory(args.input_dir, args.output_dir)
    print("Finished.")