# 画像のR(z)を計算し、図にする
import os
import argparse
import SimpleITK as sitk
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib

def branch_point_search(result, labeled_volume, start_slice, end_slice):
    branch_point = 0
    num_labels = []

    for i in range(start_slice, end_slice + 1):
        img = np.zeros((result.shape[1], result.shape[2]), dtype=np.int16)
        for j in range(result.shape[2]):
            for k in range(result.shape[1]):
                if result[k, j, i] != 0:
                    img[k, j] = 1

        labeled, num = ndimage.label(img, structure=np.ones((3, 3)))  # Use ndimage's label function with 8-connectivity
        num_labels.append(num)

        for j in range(labeled.shape[1]):
            for k in range(labeled.shape[0]):
                if labeled[k, j] != 0:
                    labeled_volume[k, j, i] = labeled[k, j]

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

            candi.append(i)
            lens.append(count_len)
            i += j
        else:
            i += 1

    if candi:
        max_len_index = np.argmax(lens)
        branch_point = start_slice + candi[max_len_index]
        print(f"Branch point index: {branch_point}")

    return branch_point

def process_directory(input_dir, output_base_dir, branch_point=9999):
    branch_point = 82 # 2012GO19 2012910
    
    if not os.path.exists(input_dir):
        print("Directory not exists.")
        return  # Exit if the input directory does not exist

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
                itkimage_stent = sitk.ReadImage(stent_file_path)
                itkimage_aorta = sitk.ReadImage(aorta_file_path)

                array_stent = sitk.GetArrayFromImage(itkimage_stent)    
                array_aorta = sitk.GetArrayFromImage(itkimage_aorta)       
                
                # Extract the parent folder name to use in the filename
                parent_folder_name = os.path.basename(os.path.dirname(original_file_path))
                additional_folder_name = os.path.basename(os.path.dirname(os.path.dirname(original_file_path)))
                output_name = f"{additional_folder_name}_{parent_folder_name}_ratio_img.png"

                stent_areas = []
                arterial_areas = []
                area_rates = []
                area_rates_index = []
                spacing = itkimage_stent.GetSpacing()
                # Process each slice
                for i, (slice_stent, slice_aorta) in enumerate(zip(array_stent, array_aorta)):
                    if np.any(slice_stent != 0):  # Check if the slice contains non-zero labels
                        # Calculate the ratio of stent area and aorta area.
                        stent_count = np.count_nonzero(slice_stent)
                        arterial_count = np.count_nonzero(slice_aorta)

                        if stent_count != 0 and arterial_count != 0 and i <= branch_point:
                            stent_area = stent_count * spacing[0] * spacing[1]
                            arterial_area = arterial_count * spacing[0] * spacing[1]
                            stent_areas.append(stent_area)
                            arterial_areas.append(arterial_area)
                            area_rates.append(stent_area / arterial_area)
                            area_rates_index.append(i)
                            print(f"{i} rate: {stent_area} {arterial_area} | {stent_count / arterial_count}")
                # Find index of maximum arterial area
                index_max = np.argmax(arterial_areas)
                index_i = area_rates_index[index_max]
                start_threshold, end_threshold = 0.7, 0.7

                # Determine AAAstart and AAAend
                def find_boundary(pointer, direction, threshold):
                    while threshold >= 0.4:
                        pointer += direction
                        if pointer < 0 or pointer >= len(area_rates):
                            pointer = index_max + direction
                            threshold -= 0.05
                        elif area_rates[pointer] > threshold:
                            return area_rates_index[pointer]
                    return area_rates_index[0] if direction == -1 else area_rates_index[-1]

                AAAstart = find_boundary(index_max, -1, start_threshold)
                AAAend = find_boundary(index_max, 1, end_threshold)
                
                # Print calculated AAAstart and AAAend
                print(f"AAAstart: {AAAstart}, AAAend: {AAAend}")

                # # Create line plot of area_rates vs area_rates_index
                # plt.figure()
                # plt.plot(area_rates_index, area_rates, label='R(z)',marker='.')
                # plt.ylim(0, 1)
                # plt.xlabel('z')
                # plt.ylabel('R(z)')
                # plt.title(f'{additional_folder_name}-{parent_folder_name}')
                # plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

                # 2 Y AXIS VERSION
                plt.figure()
                fig, ax1 = plt.subplots(figsize=(7,5))

                # R(z)
                color = 'tab:blue'
                ax1.plot(area_rates_index, area_rates, label='R(z)', marker='.', color=color, linewidth=2.5)
                ax1.set_xlabel('スライス番号',fontsize=20)
                ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax1.set_ylabel('R(z)',labelpad=0,fontsize=20)
                ax1.set_ylim(0, 1)
                ax1.tick_params(axis='y')
                # ax1.set_title(f'{additional_folder_name}-{parent_folder_name}')
                ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)

                # arterial area
                ax2 = ax1.twinx()
                color2 = 'tab:red'
                arterial_areas_for_plot = [arterial_areas[area_rates_index.index(z)] if z in area_rates_index else 0 for z in area_rates_index]
                ax2.plot(area_rates_index, arterial_areas_for_plot, label='血管領域面積 ($mm^2$)', marker='.', linestyle='--', color=color2)
                ax2.set_ylabel('血管領域面積 ($mm^2$)',labelpad=0,fontsize=20)
                ax2.tick_params(axis='y')

                # legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center')

                # Add vertical lines for AAAstart and AAAend
                # plt.axvline(x=AAAstart, color='red', linestyle='--', label='AAAstart')
                # plt.text(AAAstart+0.05, 0.02, f'start-{AAAstart}', color='black', verticalalignment='bottom')
                # plt.axvline(x=AAAend, color='red', linestyle='--', label='AAAend')
                # plt.text(AAAend+0.05, 0.02, f'end-{AAAend}', color='black', verticalalignment='bottom')
                
                # Mark data points on the plot
                # for x, y in zip(area_rates_index, area_rates):
                #     plt.text(x, y, f'({x:.0f}, {y:.2f})', fontsize=5, ha='right')
                # plt.legend()

                # Save the plot to the specified output directory
                output_path = os.path.join(output_base_dir, output_name)
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close(fig)
                print(f"Plot saved to {output_path}")
                
                # ---------------------- New: CT and mask Maximum Intensity Projection (MIP) ----------------------
                zoom_factor = spacing[2] / spacing[0]
                array_aorta = ndimage.zoom(array_aorta, (zoom_factor, 1, 1), order=0).astype(np.uint8)
                array_stent = ndimage.zoom(array_stent, (zoom_factor, 1, 1), order=0).astype(np.uint8)
                spacing = (spacing[0], spacing[1], spacing[0])
                # Combine two masks: assign 2 where stent is nonzero, 1 where aorta is nonzero (stent takes priority if overlapping)
                combined_mask = np.where(array_stent != 0, 2, np.where(array_aorta != 0, 1, 0))
                # Maximum intensity projection
                mip_mask = np.max(combined_mask, axis=1)  # Get the 2D mask image

                # Generate pseudo-color image (e.g., aorta in green, stent in red)
                mask_color = np.zeros((mip_mask.shape[0], mip_mask.shape[1], 3), dtype=np.uint8)
                mask_color[mip_mask == 1] = [255, 229, 0]
                mask_color[mip_mask == 2] = [0, 255, 255]

                # Save the MIP mask image and the overlay image
                mip_mask_output_path = os.path.join(output_base_dir, f"{additional_folder_name}_{parent_folder_name}_mip_mask.png")
                overlay_output_path = os.path.join(output_base_dir, f"{additional_folder_name}_{parent_folder_name}_mip_color.png")
                # cv2.imwrite(mip_mask_output_path, mip_mask*120)
                cv2.imwrite(overlay_output_path, mask_color)
                # print(f"MIP mask saved to {mip_mask_output_path}")
                print(f"Color MIP image saved to {overlay_output_path}")
                # ---------------------------------------------------------------------------

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