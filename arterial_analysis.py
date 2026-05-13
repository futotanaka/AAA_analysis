import numpy as np
from scipy.ndimage import label
from scipy.ndimage import zoom
import math
import cv2
import pandas as pd

# Convert "postProcessingForArterial" to Python
def post_processing_for_arterial(arterial_mask, spacing, branch_point, stent_mask, zoom_factor):
    start_slice, end_slice = 0, 0
    flag = False

    # Calculate start_slice and end_slice
    z_indices = np.any(arterial_mask != 0, axis=(1, 2))
    start_slice = np.argmax(z_indices)  # First non-zero slice
    end_slice = len(z_indices) - 1 - np.argmax(z_indices[::-1])  # Last non-zero slice

    print(f"Arterial start slice: {start_slice} | End slice: {end_slice}")

    # Bounding box calculation
    # bounding_box_main = bounding_box_cal(arterial_mask, spacing, branch_point)
    # bounding_box_whole = bounding_box_cal(arterial_mask, spacing, end_slice)

    # Slice thickness
    slice_thickness = 5.0 / spacing[2]

    stent_areas, arterial_areas, area_rates, area_rates_index = [], [], [], []
    max_arterial, index_max = 0, 0

    # Calculate areas and area rates
    for i in range(start_slice, end_slice + 1):
        stent_count = np.count_nonzero(stent_mask[i])
        arterial_count = np.count_nonzero(arterial_mask[i])

        if stent_count > 0 and arterial_count > 0:
            stent_area = stent_count * spacing[0] * spacing[1]
            arterial_area = arterial_count * spacing[0] * spacing[1]
            
            if arterial_area > max_arterial:
                max_arterial = arterial_area
                index_max = i

            stent_areas.append(stent_area)
            arterial_areas.append(arterial_area)
            area_rates.append(stent_area / arterial_area)
            area_rates_index.append(i)
            
            # print(f"{i} rate: {stent_area} {arterial_area} | {stent_count / arterial_count}")

    # Analyze axial areas
    analyze_axial_areas(arterial_mask, start_slice, end_slice, slice_thickness, spacing)

    # Calculate AAA range and volume
    aaa_start, aaa_end, volume = calculate_aaa_range(area_rates, area_rates_index, arterial_mask, spacing, start_slice, index_max, zoom_factor, branch_point)

    # Calculate maximum diameter and short axis
    max_diameter, max_short_diameter = calculate_diameters(arterial_mask, spacing, aaa_start, aaa_end)

    # Save part of the data
    data = [[volume, max_diameter, max_short_diameter]]
    df = pd.DataFrame(data, columns=['AAA_volume', 'max_diameter', 'max_short_diameter'])
    save_path = '/home/tanaka/data/analysis_res3/part_of_the_results.csv'
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print("~~~~~~~~~~ Save part of the data ~~~~~~~~~~~~~~~")
    
    return [aaa_start, aaa_end]

# Bounding box calculation
def bounding_box_cal(mask, spacing, reference_slice):
    bounding_box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]

    for i in range(reference_slice + 1):
        non_zero_coords = np.argwhere(mask[i] > 0)
        if non_zero_coords.size > 0:
            bounding_box[:3] = np.minimum(bounding_box[:3], np.min(non_zero_coords, axis=0))
            bounding_box[3:] = np.maximum(bounding_box[3:], np.max(non_zero_coords, axis=0))

    print(f"Bounding box: {bounding_box}")
    print(f"Volume (mm^3): {(bounding_box[3]-bounding_box[0])*spacing[0] * (bounding_box[4]-bounding_box[1])*spacing[1] * (bounding_box[5]-bounding_box[2])*spacing[2]}")
    return bounding_box

# Analyze axial areas
def analyze_axial_areas(arterial_mask, start_slice, end_slice, slice_thickness, spacing):
    z_areas = []

    for i in range(int(start_slice + slice_thickness), int(end_slice - slice_thickness), int(slice_thickness)):
        z_area = np.count_nonzero(arterial_mask[i]) * spacing[0] * spacing[1]
        z_areas.append(z_area)

    if z_areas:
        median = np.median(z_areas)
        max_area = max(z_areas)
        vcs_ratio = median / max_area
        print(f"Median area: {median}, Max Area: {max_area}, VCS Ratio: {vcs_ratio}")

# Calculate AAA range and volume
def calculate_aaa_range(area_rates, area_rates_index, arterial_mask, spacing, start_slice, index_max, zoom_factor, branch_point):
    aaa_start, aaa_end = 0, 0
    apointer, bpointer = area_rates_index.index(index_max), area_rates_index.index(index_max)

    start_threshold, end_threshold = 0.7, 0.7
    while aaa_start == 0 or aaa_end == 0:
        if aaa_start == 0:
            start_found = False
            while start_threshold >= 0.4 and not start_found:
                apointer -= 1
                if apointer < 0:
                    apointer = 0
                    apointer = area_rates_index.index(index_max) - 1
                    start_threshold -= 0.05

                if area_rates[apointer] > start_threshold:
                    aaa_start = area_rates_index[apointer]
                    start_found = True
                    print(f"Start found at threshold: {start_threshold}")

            if start_threshold < 0.4 and aaa_start == 0:
                apointer = 0
                aaa_start = area_rates_index[apointer]
                print("Start not found. Defaulting to first slice.")

        if aaa_end == 0:
            end_found = False
            while end_threshold >= 0.4 and not end_found:
                bpointer += 1
                if bpointer >= len(area_rates):
                    bpointer = area_rates_index.index(index_max) + 1
                    end_threshold -= 0.05

                if area_rates[bpointer] > end_threshold:
                    aaa_end = area_rates_index[bpointer]
                    end_found = True
                    print(f"End found at threshold: {end_threshold}")

            if end_threshold < 0.4 and aaa_end == 0:
                bpointer = len(area_rates_index) - 1
                aaa_end = area_rates_index[bpointer]
                print("End not found. Defaulting to last slice.")
                
    print(f"AAA range: [{aaa_start}, {aaa_end}] ---- interpolation: [{round(aaa_start / zoom_factor)}, {round(aaa_end / zoom_factor)}]")
    if aaa_end > branch_point:
        aaa_end = branch_point
        print(f"AAA end greater than branch point.\nFinal AAA range: [{aaa_start}, {aaa_end}] ---- interpolation: [{round(aaa_start / zoom_factor)}, {round(aaa_end / zoom_factor)}]")
    # volume = np.sum([np.count_nonzero(arterial_mask[i]) * spacing[0] * spacing[1] * spacing[2] for i in range(aaa_start, aaa_end + 1)])
    volume = np.count_nonzero(arterial_mask[aaa_start:aaa_end+1]) * spacing[0] * spacing[1] * spacing[2]
    print(f"AAA Volume: {volume}")
    return aaa_start, aaa_end, volume

# Calculate maximum diameter and short axis
def calculate_diameters(arterial_mask, spacing, start_slice, end_slice):
    max_diameter = 0
    max_short_diameter = 0
    max_dindex = 0
    max_shortidx = 0
    slice_thickness = int(5.0 / spacing[2])

    for i in range(start_slice, end_slice + 1, slice_thickness):
        img = arterial_mask[i]
        
        # # print("Ellipse fit not applied.")
        # max_p1, max_p2, short_p1, short_p2 = None, None, None, None
        # diameter, max_p1, max_p2 = max_diameter_calculation(arterial_mask[i])
        # diameter *= spacing[0]
        # # print(f"Slice {i}: Diameter = {diameter}")

        # draw_line_on_mask(arterial_mask, i, max_p1, max_p2, value=3)

        # short_axis, short_p1, short_p2 = calculate_max_short_axis(arterial_mask[i], max_p1, max_p2)
        # short_axis *= spacing[0]
        # # print(f"Slice {i}: Short Axis = {short_axis}")

        # draw_line_on_mask(arterial_mask, i, short_p1, short_p2, value=4)
        
        # print("Ellipse fit applied.")
        contours,hierarchy =  cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for _, cnt in enumerate(contours):
            if len(cnt) < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)

            diameter = max(ellipse[1]) * spacing[0]
            short_axis = min(ellipse[1]) * spacing[0]
            # print(f"Slice {i}: Short Axis = {short_axis}")
            # Draw ellipse
            arterial_mask[i] = cv2.ellipse(arterial_mask[i],ellipse,5,2)
            max_p1, max_p2, short_p1, short_p2 = points_of_ellipse(ellipse)
            draw_line_on_mask(arterial_mask, i, max_p1, max_p2, value=3, thickness=1)
            draw_line_on_mask(arterial_mask, i, short_p1, short_p2, value=4, thickness=1)

        if diameter > max_diameter:
            max_diameter = diameter
            max_dindex = i
        if short_axis > max_short_diameter:
            max_short_diameter = short_axis
            max_shortidx = i
            

    print(f"Max Diameter: {max_diameter} at slice {max_dindex}")
    print(f"Max Short Diameter: {max_short_diameter} at slice {max_shortidx}")
    return max_diameter, max_short_diameter

def points_of_ellipse(ellipse):
    (cy, cx), (w, h), angle = ellipse

    if w >= h:
        major = w / 2.0
        minor = h / 2.0
        theta = math.radians(angle)
    else:
        major = h / 2.0
        minor = w / 2.0
        theta = math.radians(angle + 90)

    # Minor axis intersection points
    pt_minor1 = (cx + minor * (-math.cos(theta)), cy + minor * math.sin(theta))
    pt_minor2 = (cx - minor * (-math.cos(theta)), cy - minor * math.sin(theta))

    # Major axis intersection points (direction orthogonal to the minor axis)
    pt_major1 = (cx + major * math.sin(theta), cy + major * math.cos(theta))
    pt_major2 = (cx - major * math.sin(theta), cy - major * math.cos(theta))
    
    return pt_major1, pt_major2, pt_minor1, pt_minor2
# Draw a line on the mask
def draw_line_on_mask(mask, slice_idx, p1, p2, value, thickness=1):
    x1, y1 = p1
    x2, y2 = p2

    for t in np.linspace(0, 1, 100):
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))

        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                xi = x + dx
                yi = y + dy
                if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[2]:
                    mask[slice_idx, xi, yi] = value

# Calculate maximum diameter
def max_diameter_calculation(slice_data):
    points = np.argwhere(slice_data > 0)
    if points.size == 0:
        return 0, (0, 0), (0, 0)

    points = [tuple(point) for point in points]
    hull = cv2.convexHull(np.array(points))

    max_diameter = 0
    max_p1, max_p2 = (0, 0), (0, 0)

    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            dist = np.linalg.norm(hull[i][0] - hull[j][0])
            if dist > max_diameter:
                max_diameter = dist
                max_p1, max_p2 = tuple(hull[i][0]), tuple(hull[j][0])

    return max_diameter, max_p1, max_p2

# Calculate maximum short axis
def calculate_max_short_axis(slice_data, p1, p2):
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    perp_dx, perp_dy = -dy, dx
    magnitude = math.sqrt(perp_dx**2 + perp_dy**2)
    perp_dx /= magnitude
    perp_dy /= magnitude

    neg_dir_end, pos_dir_end = (mid_x, mid_y), (mid_x, mid_y)

    for direction in [-1, 1]:
        length = 0
        current_x, current_y = mid_x, mid_y

        while True:
            rounded_x = int(round(current_x))
            rounded_y = int(round(current_y))

            if not (0 <= rounded_x < slice_data.shape[0] and 0 <= rounded_y < slice_data.shape[1]):
                break

            if slice_data[rounded_x, rounded_y] == 0:
                break

            length += 1
            current_x += direction * perp_dx
            current_y += direction * perp_dy

            if direction == -1:
                neg_dir_end = (current_x, current_y)
            else:
                pos_dir_end = (current_x, current_y)

    max_short_axis = math.sqrt((pos_dir_end[0] - neg_dir_end[0])**2 + (pos_dir_end[1] - neg_dir_end[1])**2)
    return max_short_axis, neg_dir_end, pos_dir_end
