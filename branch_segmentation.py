import numpy as np
import os
import cv2
import skimage.io
from scipy import ndimage
import matplotlib.pyplot as plt
import mhd_io
import preprocessing
import argparse
from arterial_analysis import points_of_ellipse
from arterial_analysis import draw_line_on_mask

def euclidean_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.sqrt(np.sum((a - b) ** 2))

def separation(image):
    sure_bg = cv2.dilate(image, preprocessing._circle_structure(3), iterations=3)
    dist = cv2.distanceTransform(image, cv2.DIST_L2,5)
    
    # search the threshold value to get two branches
    threshold_rate = 0.6
    _, sure_fg = cv2.threshold(dist, threshold_rate * dist.max(), 1, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(sure_fg)
    while num_labels == 2 and threshold_rate <= 0.95:
        threshold_rate = threshold_rate + 0.02
        _, sure_fg = cv2.threshold(dist, threshold_rate * dist.max(), 1, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(sure_fg)
    
    unknown = ((sure_bg == 1) & (sure_fg == 0)).astype(np.uint8) 
    
    ret, markers = cv2.connectedComponents(sure_fg)
    # print("number labels: ", ret)
    markers = markers + 1
    markers[unknown == 1] = 0
    three_channel_img = cv2.cvtColor(image*255, cv2.COLOR_GRAY2BGR)
    water_mark = cv2.watershed(three_channel_img, markers)
    image[(water_mark > 1)] = water_mark[water_mark > 1] - 1
    
    return image
    
def watershed_processing(stent_mask):
    num_l_list = []
    no_sep_slice = []
    branch_point = 0
    connected_stent_end = 0
    slice_count = 0
    separated_mask = np.zeros(stent_mask.shape, dtype=np.uint8)
    max_threshold = 0.8 # 0.7?
    higher_threshold = 0.98
    # stent_mask = ndimage.binary_closing(stent_mask, preprocessing._spherical_structure(5))
    for z in range(stent_mask.shape[0]):
        image = stent_mask[z].astype(np.uint8) 
        # image = ndimage.binary_closing(image, preprocessing._circle_structure(3)).astype(np.uint8)
        image = ndimage.binary_fill_holes(image).astype(np.uint8)
        
        sure_bg = cv2.dilate(image, preprocessing._circle_structure(3), iterations=3)
        dist = cv2.distanceTransform(image, cv2.DIST_L2,5)
        
        # search the threshold value to get two branches
        threshold_rate = 0.4
        _, sure_fg = cv2.threshold(dist, threshold_rate * dist.max(), 1, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(sure_fg)
        while num_labels == 2 and threshold_rate <= max_threshold:
            threshold_rate = threshold_rate + 0.02
            _, sure_fg = cv2.threshold(dist, threshold_rate * dist.max(), 1, cv2.THRESH_BINARY)
            sure_fg = sure_fg.astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(sure_fg)
            if num_labels != 2:
                max_threshold = higher_threshold
        
        unknown = ((sure_bg == 1) & (sure_fg == 0)).astype(np.uint8) 
        
        ret, markers = cv2.connectedComponents(sure_fg)
        num_l_list.append(ret)
        # print("number labels: ", ret)
        markers = markers + 1
        markers[unknown == 1] = 0
        out_marker = markers.copy()
        three_channel_img = cv2.cvtColor(image*255, cv2.COLOR_GRAY2BGR)
        water_mark = cv2.watershed(three_channel_img, markers)
        out_marker2 = markers.copy()
        # edges = (water_mark == -1).astype(np.uint8)
        # edges = cv2.dilate(edges, preprocessing._circle_structure(1.8))
        # image[edges == 1] = 2  
        separated_mask[z][(water_mark > 1)] = water_mark[water_mark > 1] - 1
        out_marker3 = separated_mask[z].copy()
        
        # find the two largest area
        labels, num_features = ndimage.label(separated_mask[z])

        areas = ndimage.sum(separated_mask[z], labels, range(1, num_features + 1))

        largest_areas = np.argsort(areas)[-2:]
        largest_labels = largest_areas + 1  # adjust index

        separated_mask[z] = np.zeros_like(separated_mask[z])

        for new_label, label in enumerate(largest_labels, start=1):
            separated_mask[z][labels == label] = new_label
        
       
        if num_l_list[z] > 1: slice_count += 1
        if branch_point == 0 and slice_count > 5 and num_l_list[z] > 2 and num_l_list[z-1] <= 2:
            branch_point = z
        
        if branch_point != 0 and z >= branch_point + 1 and connected_stent_end == 0:
            _, num_labels = ndimage.label(stent_mask[z])
            if num_labels > 1:
                connected_stent_end = z
        
        if branch_point != 0 and z >= branch_point + 1:
            # separete optimize
            if connected_stent_end == 0:
                num_labels = np.unique(separated_mask[z])
                num_labels = len(num_labels[num_labels != 0])
                _, test = ndimage.label(separated_mask[z])
                if num_labels == 1:
                    no_sep_slice.append(z)

        # if z == 90:
        #     # print("number labels: ", ret)
        #     cv2.imwrite("0.jpg",ndimage.binary_fill_holes(stent_mask[z]) * 255)
        #     plt.pcolor(dist)
        #     plt.savefig("dist.jpg")
        #     cv2.imwrite("1.jpg",((sure_bg == 0)) * 255)
        #     cv2.imwrite("2.jpg",sure_fg * 255)
        #     cv2.imwrite("3.jpg",unknown * 255)
        #     plt.pcolor(out_marker, vmin=0, vmax = ret)
        #     plt.colorbar()
        #     plt.savefig("4.jpg")
        #     plt.pcolor(out_marker2, vmin=0, vmax = ret)
        #     plt.savefig("5.jpg")
        #     plt.pcolor(out_marker3, vmin=0, vmax = ret)
        #     plt.savefig("6.jpg")
        #     # cv2.imwrite("6.jpg",image * 255) 

    for i in range(len(no_sep_slice)):
        # separete optimize
        z = no_sep_slice[i]
        # print(z)
        # 見つからない場合、楕円近似して短軸を分割線とする。
        img = separated_mask[z]
        contours,hierarchy =  cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for _, cnt in enumerate(contours):
            if len(cnt) < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)

            diameter = max(ellipse[1])
            short_axis = min(ellipse[1])
            # print(f"Slice {i}: Short Axis = {short_axis}")
            # Draw ellipse
            max_p1, max_p2, short_p1, short_p2 = points_of_ellipse(ellipse)
            pt1 = (int(short_p1[0]), int(short_p1[1]))
            pt2 = (int(short_p2[0]), int(short_p2[1]))
            
            draw_line_on_mask(separated_mask, z, pt1, pt2,value=0)
            _, num_labels = ndimage.label(separated_mask[z])
            
            
        # 前のスライスの分割情報を用いて適用する方法
        # pre_z = z - 1
        # aft_z = z + 1
        # while pre_z in no_sep_slice:
        #     pre_z = pre_z - 1
        # while aft_z in no_sep_slice:
        #     aft_z = aft_z + 1
        # ref_slice = pre_z if abs(pre_z-z) < abs(aft_z-z) else aft_z
        # # print(ref_slice, pre_z, aft_z)
        # cur_img = (separated_mask[z] > 0).astype(np.uint8)
        # pre_l1 = ((separated_mask[ref_slice] == 1) & (cur_img > 0)).astype(np.uint8)
        # pre_l2 = ((separated_mask[ref_slice] == 2) & (cur_img > 0)).astype(np.uint8)
        # structure = np.ones((3, 3), dtype=np.uint8)
        # while not np.all((pre_l1 | pre_l2) >= cur_img):
        #     new_pre_l1 = ndimage.binary_dilation(pre_l1, structure=structure)
        #     new_pre_l2 = ndimage.binary_dilation(pre_l2, structure=structure)
        #     pre_l1 = new_pre_l1 & ~(pre_l2)
        #     pre_l2 = new_pre_l2 & ~(pre_l1)
        # final_mask = (pre_l1 * 1 + pre_l2 * 2) * (cur_img > 0).astype(np.uint8)
        # exchange? in-progressing
        # if not abs(pre_z-z) < abs(aft_z-z):
        #     before_1 = (separated_mask[z-1]==1).astype(np.uint8)
        #     before_2 = (separated_mask[z-1]==2).astype(np.uint8)
        #     cover_1 = ((pre_l1 > 0) & (before_1>0)).astype(np.uint8)
        #     cover_2 = ((pre_l2 > 0) & (before_2>0)).astype(np.uint8)
        #     rate_1 = np.sum(cover_1)/np.sum(before_1)
        #     rate_2 = np.sum(cover_2)/np.sum(before_2)
        #     # print(rate_1,rate_2)
        #     if (rate_1 < 1/3 or rate_2 < 1/3) and rate_1 != 0 and rate_2 != 0:
        #         temp = pre_l1
        #         pre_l1 = pre_l2
        #         pre_l2 = temp
        
        
        # elif abs(pre_z-z) == abs(aft_z-z) and abs(pre_z-z) > 1 and abs(aft_z-z) > 1:
        #     print(z)
    
    # 3 branches have different labels or we only have 1 and 2.
    for z in range(stent_mask.shape[0]):
        if branch_point != 0 and z >= branch_point:
            separated_mask[z][(separated_mask[z] > 0)] = separated_mask[z][separated_mask[z] > 0] + 1
    
    # define the label 2 is the right branch
    z = branch_point
    coords_2 = np.where(separated_mask[z] == 2)
    coords_3 = np.where(separated_mask[z] == 3)
    rightmost_2 = np.max(coords_2[1])  
    rightmost_3 = np.max(coords_3[1])  
    if rightmost_2 > rightmost_3:
        separated_mask[z][separated_mask[z] == 2] = 4
        separated_mask[z][separated_mask[z] == 3] = 2
        separated_mask[z][separated_mask[z] == 4] = 3
        
    # 枝の連続性を維持　ー　重なる面積
    # for z in range(branch_point + 1, stent_mask.shape[0]):
    #     contiguity_img = separated_mask[z].copy()
    #     # count_list = []
    #     for label in np.unique(separated_mask[z]):
    #         if(label == 0):
    #             continue
    #         mask = (separated_mask[z] == label)
    #         overlap = separated_mask[z-1][mask]
    #         overlap = overlap[overlap > 0]
    #         if overlap.size > 0:
    #             counts = np.bincount(overlap)
    #             # print(z, counts)
    #             correct_label = np.argmax(counts)
    #             contiguity_img[mask] = correct_label
    #      separated_mask[z] = contiguity_img

    # 枝の連続性を維持　ー　重心の距離
    for z in range(branch_point + 1, stent_mask.shape[0]):
        pre_img = separated_mask[z-1].copy()
        contiguity_img = separated_mask[z].copy()
        pre_centers = ndimage.center_of_mass(pre_img, labels=pre_img, index=[2, 3])
        unique_labels = np.unique(contiguity_img)
        unique_labels = unique_labels[unique_labels != 0]
        # print(z, unique_labels)
        
        if len(unique_labels) == 1:
            binary_mask = contiguity_img > 0
            labeled_components, num_components = ndimage.label(binary_mask)
            # print(f"Slice {z}: Only one label ({unique_labels[0]}). Found {num_components} connected components.")
        
            for comp in range(1, num_components + 1):
                comp_center = ndimage.center_of_mass(binary_mask, labels=labeled_components, index=comp)
                d2 = euclidean_distance(comp_center, pre_centers[0])
                d3 = euclidean_distance(comp_center, pre_centers[1])
                d2 = np.nan_to_num(d2, nan=np.inf)
                d3 = np.nan_to_num(d3, nan=np.inf)
                new_label = 2 if d2 < d3 else 3
                contiguity_img[labeled_components == comp] = new_label
        else:
            cur_centers = ndimage.center_of_mass(contiguity_img, labels=contiguity_img, index=[2, 3])
            dist_pre2_cur2 = euclidean_distance(cur_centers[0], pre_centers[0])
            dist_pre2_cur3 = euclidean_distance(cur_centers[0], pre_centers[1])
            dist_pre3_cur2 = euclidean_distance(cur_centers[1], pre_centers[0])
            dist_pre3_cur3 = euclidean_distance(cur_centers[1], pre_centers[1])
            
            dist_pre2_cur2 = np.nan_to_num(dist_pre2_cur2, nan=np.inf)
            dist_pre2_cur3 = np.nan_to_num(dist_pre2_cur3, nan=np.inf)
            dist_pre3_cur2 = np.nan_to_num(dist_pre3_cur2, nan=np.inf)
            dist_pre3_cur3 = np.nan_to_num(dist_pre3_cur3, nan=np.inf)
            # print(f"Slice {z}: dist(cur_label2, pre_label2) = {dist_pre2_cur2:.3f}, dist(cur_label2, pre_label3) = {dist_pre2_cur3:.3f}")
            
            temp = contiguity_img.copy()
            if dist_pre2_cur3 < dist_pre2_cur2:
                # print(f"Slice {z}: swapping labels 2 and 3")
                contiguity_img[temp == 2] = 3
                # temp_val = 99
                # contiguity_img[contiguity_img == 2] = temp_val
                # contiguity_img[contiguity_img == 3] = 2
                # contiguity_img[contiguity_img == temp_val] = 3
            if dist_pre3_cur2 < dist_pre3_cur3:
                contiguity_img[temp == 3] = 2

        separated_mask[z] = contiguity_img.copy()
    
    # fill the 1-pixel line between parts
    for z in range(stent_mask.shape[0]):
        # print(z, end=" ")
        ori = stent_mask[z].astype(np.uint8) 
        num_labels = np.unique(separated_mask[z])
        num_labels = len(num_labels[num_labels != 0])
        if num_labels == 1:
            # separated_mask[z] = ori
            l1 = (separated_mask[z] > 0).astype(np.uint8)
            structure = np.ones((3, 3), dtype=np.uint8)
            while not np.all(l1 >= ori):
                l1 = ndimage.binary_dilation(l1, structure=structure)
            
            unique_labels = np.unique(separated_mask[z][separated_mask[z] > 0])
            final_mask = (l1 * unique_labels[0]) * (ori > 0).astype(np.uint8)
            separated_mask[z] = final_mask
        elif num_labels == 2:
            l1 = (separated_mask[z] == 2).astype(np.uint8)
            l2 = (separated_mask[z] == 3).astype(np.uint8)
            structure = np.ones((3, 3), dtype=np.uint8)
            while not np.all((l1 | l2) >= ori):
                new_l1 = ndimage.binary_dilation(l1, structure=structure)
                new_l2 = ndimage.binary_dilation(l2, structure=structure)
                l1 = new_l1 & ~(l2)
                l2 = new_l2 & ~(l1)
            final_mask = (l1 * 2 + l2 * 3) * (ori > 0).astype(np.uint8)
            separated_mask[z] = final_mask
        
    for z in range(branch_point + 1, stent_mask.shape[0]):
        pre_img = separated_mask[z-1].copy()
        contiguity_img = separated_mask[z].copy()
        pre_centers = ndimage.center_of_mass(pre_img, labels=pre_img, index=[2, 3])
        unique_labels = np.unique(contiguity_img)
        unique_labels = unique_labels[unique_labels != 0]
        
        binary_mask = contiguity_img > 0
        labeled_components, num_components = ndimage.label(binary_mask)
        # print(f"Slice {z}: label ({unique_labels}). Found {num_components} connected components.")
        if len(unique_labels) == 1 and num_components > 1:
            for comp in range(1, num_components + 1):
                comp_center = ndimage.center_of_mass(binary_mask, labels=labeled_components, index=comp)
                d2 = euclidean_distance(comp_center, pre_centers[0])
                d3 = euclidean_distance(comp_center, pre_centers[1])
                d2 = np.nan_to_num(d2, nan=np.inf)
                d3 = np.nan_to_num(d3, nan=np.inf)
                new_label = 2 if d2 < d3 else 3
                contiguity_img[labeled_components == comp] = new_label
            separated_mask[z] = contiguity_img.copy()
    
    return separated_mask

def do_segment(stent_mask, arterial_mask, voxel_spacing, base_path, outname):
    separated_mask = watershed_processing(stent_mask)
    
    # remove the extra voxels based on count (mainly caused by calcification)
    flag = False
    # count_2 = np.sum(separated_mask == 2)
    # count_3 = np.sum(separated_mask == 3)
    # if count_2 < count_3 / 3:
    #     separated_mask[separated_mask == 2] = 0  
    #     count_2 = np.sum(separated_mask == 2)
    #     if count_2 != 0:
    #         flag = True
    # elif count_3 < count_2 / 3:
    #     separated_mask[separated_mask == 3] = 0  
    #     count_3 = np.sum(separated_mask == 3)
    #     if count_3 != 0:
    #         flag = True
    
    while False:
        for z in range(stent_mask.shape[0]):
            separated_mask[z][separated_mask[z] > 0] = 1
            separated_mask[z] = ndimage.binary_closing(separated_mask[z]).astype(np.uint8)
            # separated_mask[z] = ndimage.binary_dilation(separated_mask[z]).astype(np.uint8)
        
        separated_mask = watershed_processing(separated_mask)
        
        count_2 = np.sum(separated_mask == 2)
        count_3 = np.sum(separated_mask == 3)
        if count_2 < count_3 / 2:
            separated_mask[separated_mask == 2] = 0  
            flag = True
        elif count_3 < count_2 / 2:
            separated_mask[separated_mask == 3] = 0  
            flag = True
        else:
            flag = False
    
    # largest element
    labeled, _ = ndimage.label(separated_mask)
    component_sizes = np.bincount(labeled.ravel())
    largest_component_index = np.argmax(component_sizes[1:]) + 1
    separated_mask = np.where(labeled == largest_component_index, separated_mask, 0)
    
    # largest element by label
    # result_mask = np.zeros_like(separated_mask)
    # for label_val in [1, 2, 3]:
    #     binary_mask = (separated_mask == label_val)
    #     labeled_array, num_features = ndimage.label(binary_mask)
    #     if num_features == 0:
    #         continue 
        
    #     component_sizes = ndimage.sum(binary_mask, labeled_array, index=np.arange(1, num_features+1))
    #     largest_component_label = np.argmax(component_sizes) + 1
    #     largest_component_mask = (labeled_array == largest_component_label)
    #     result_mask[largest_component_mask] = label_val
    
    # add arterial region
    separated_mask[(arterial_mask > 0) & (separated_mask == 0)] = 10
    
    out_mhd_file_name = os.path.join(base_path, f"{outname}_segment_mask.mhd")
    mhd_io.export_to_mhd_and_raw(out_mhd_file_name, separated_mask, voxel_spacing)
    
    return separated_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Test segmentation',
        add_help=True)
    parser.add_argument('test_data_path', help='Path of test data')
    args = parser.parse_args()
    
    predict = True
    # date_list = ["2012EN17","2012EN18","2012EN22","2012GO19","2012PO12"]
    # date_list = ["2012EN22"]
    # date_details = [["20121010", "20130312","20130909","20140905","20150907","20160928","20170322","20170920","20180830","20190822","20201216"]]
    # date_details = [["20150907"]]
    date_list = ["2012EN17"]
    date_details = [["20170629"]]
    # date_list = ["2012EN2","2012EN16","2012GO20"]
    # date_details = [["20120215", "20130110","20140130"],["20120724","20130110","20130628","20140704","20150713"],
    #                 ["20120928","20130208","20130830","20140822","20150817"]]
    # date_list = ["2012PO14","2012GO10","2012PO13"]
    # date_details = [["20120712","20170531"],["20120622","20140425"],["20121226","20160530"]]
    # date_list = ["2014EGO3","2015EEN5"]
    # date_details = [["20141022", "20150831","20160914", "20170921"],["20150930","20160210","20160810","20170809","20180809","20200629"]]

    if not predict:
        for ct_date in date_list:
            print(ct_date)
            base_path = args.test_data_path
            
            in_mhd_file_name = os.path.join(base_path, f"{ct_date}_prediction.mhd")
            out_mhd_file_name = os.path.join(base_path, f"{ct_date}_segment_mask.mhd")

            do_segment(in_mhd_file_name, base_path, ct_date)
    elif predict:
        for ct_date in date_list:
            index = date_list.index(ct_date)
            for date in date_details[index]:
                base_path = args.test_data_path
                print(ct_date, date)
                outname = f"{ct_date}_{date}"
                in_mhd_file_name = os.path.join(base_path, f"{outname}_prediction.mhd")
                out_mhd_file_name = os.path.join(base_path, f"{outname}_segment_mask.mhd")
                
                ori_stent_mask = skimage.io.imread(in_mhd_file_name, plugin='simpleitk')
                stent_mask = ori_stent_mask.copy()
                arterial_mask = (stent_mask == 10).astype(np.uint8)
                stent_mask[arterial_mask > 0] = 0
                voxel_spacing = mhd_io.get_voxel_spacing_from_mhd(in_mhd_file_name)
                print(voxel_spacing) 
                
                do_segment(stent_mask, arterial_mask, voxel_spacing, base_path, outname)