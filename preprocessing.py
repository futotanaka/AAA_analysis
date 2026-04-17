"""
Created on Thr Dec 9 2021
@author: ynomura
"""

import numpy as np
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage.io

test_idx = 40

def window_normalization(src_volume, src_range, dst_range):
    dst_volume = np.copy(src_volume)

    slope = (float(dst_range[1]) - float(dst_range[0])) \
            / (float(src_range[1]) - float(src_range[0]))

    dst_volume = (dst_volume.astype('float') - float(src_range[0])) * slope \
                 + float(dst_range[0])
    dst_volume[dst_volume < dst_range[0]] = dst_range[0]
    dst_volume[dst_volume > dst_range[1]] = dst_range[1]

    return dst_volume

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


def chest_and_iliac_slice_estimation(org_volume, body_mask, bone_mask,
                                     voxel_spacing, verbose=True):

    moment2 = np.zeros(org_volume.shape[0])
    r_air = np.zeros(org_volume.shape[0])
    thorax_likelihoods = np.zeros(org_volume.shape[0])

    img_raw_idx = np.arange(0, org_volume.shape[1], 1)
    img_col_idx = np.arange(0, org_volume.shape[2], 1)
    xx, yy = np.meshgrid(img_col_idx, img_raw_idx)

    for k in range(org_volume.shape[0]):

        body_cnt = np.sum(body_mask[k])
        bone_img = bone_mask[k]
        bone_cnt = np.sum(bone_img)
        air_cnt = np.sum(((org_volume[k] < -500) & (body_mask[k] == 1)).astype(np.uint8))

        if bone_cnt > 0:
            bone_pos = np.where(bone_img)
            gx = np.mean(np.array(bone_pos[1]))
            gy = np.mean(np.array(bone_pos[0]))

            bone_img_mask = ((bone_img == 1) & (org_volume[k] >= 200)).astype(np.uint8)
            cnt = np.sum(bone_img_mask)
            tmp_dist_matrix = np.sqrt((gx - xx) ** 2 + (gy - yy) ** 2) * bone_img_mask
            dist_sum = np.sum(tmp_dist_matrix)
            moment2[k] = (np.sum(tmp_dist_matrix ** 2) - dist_sum) / bone_cnt
            #print(k, moment2[k])

        if body_cnt > 0.0:
            r_air[k] = air_cnt / body_cnt

        #print(k, gx, gy, moment2[k], body_cnt, air_cnt, r_air[k])

    moment2_mean = np.average(moment2)
    thorax_likelihood_max = 0.0
    thorax_likelihood_max_idx = 0

    for k in range(org_volume.shape[0]):
        thorax_likelihoods[k] = moment2[k] / moment2_mean * r_air[k]

        if thorax_likelihoods[k] > thorax_likelihood_max:
            thorax_likelihood_max = thorax_likelihoods[k]
            thorax_likelihood_max_idx = k

    thorax_likelihood_mean = np.average(thorax_likelihoods)
    ret = np.zeros(3, dtype=int) # range of thorax and slice of iliac crest

    for k in reversed(range(0, thorax_likelihood_max_idx + 1)):
        if thorax_likelihoods[k] < thorax_likelihood_mean:
            break
        ret[0] = k

    for k in range(thorax_likelihood_max_idx, org_volume.shape[0]):
        if thorax_likelihoods[k] < thorax_likelihood_mean:
            break
        ret[1] = k

    moment2_max = 0
    moment2_max_idx = 0

    for k in range(ret[0], ret[1]):
        if moment2[k] > moment2_max:
            moment2_max = moment2[k]
            moment2_max_idx = k

    start_zpos = ret[1] + int(100.0 / voxel_spacing[2] + 1.0) # 胸腔より10cm下から
    print("10cm below chest: ", start_zpos)
    for k in range(start_zpos, org_volume.shape[0]):
        ret[2] = k + int(100.0 / voxel_spacing[2] + 1.0)
        if moment2[k] > moment2_max / 4.0:
            break
        

    if verbose:
        # plt.plot(np.arange(0, moment2.shape[0], 1), moment2, color="blue", label="Second moment")
        # plt.show()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        line1, = ax1.plot(np.arange(0, moment2.shape[0], 1), moment2, label="Second moment", color="turquoise")
        line2, =ax2.plot(np.arange(0, r_air.shape[0], 1), r_air, label="Air rate", color="tomato")
        '''
        胸郭範囲と腸骨稜の部分に点線を引く
        '''
        # plt.vlines(ret[0], 0, 20000, color='#EC008C', linestyles='solid', linewidth=2)
        # plt.vlines(ret[1], 0, 20000, color='#EC008C', linestyles='solid', linewidth=2)
        # plt.vlines(ret[2], 0, 20000, color='#C00000', linestyles='solid', linewidth=2)
        #plt.vlines(129, 0, 20000, color='k', linestyles='solid', linewidth=1)
        ax1.set_xlabel('Number of slice')
        ax1.set_ylabel('Second moment')
        ax2.set_ylabel('Air rate')
        plt.legend([line1,line2],["Second moment","Air rate"])
        plt.savefig("graph.png")
    
    #print(thorax_range)
    return ret


def bone_extraction(volume, body_mask, voxel_spacing, extracted_ratio_th=0.8):

     bone_mask = np.zeros(volume.shape, dtype=np.uint8)
     bone_mask[(volume >= 200) & (body_mask == 1)] = 1
     
     # first round (rough segmentation avoiding oversegmentation)
     for k in range(volume.shape[0]):
         binary_img = bone_mask[k,:,:]
         binarized_cnt = float(np.sum(binary_img))

         structure2d = np.ones([3, 3])
         labeled, num_labels = ndimage.label(binary_img, structure2d)

         histogram = ndimage.histogram(labeled, 1, num_labels, num_labels)

         sorted_idx = np.argsort(-histogram)
         extracted_label_cnt = 0
         sum_of_extracted_ratio = 0.0

         for n in range(num_labels):

             if sum_of_extracted_ratio > extracted_ratio_th:
                 labeled[labeled == sorted_idx[n] + 1] = 0

             else:
                 extracted_label_cnt += histogram[sorted_idx[n]]
                 sum_of_extracted_ratio = float(extracted_label_cnt) / binarized_cnt

         binary_img = (labeled > 0).astype(np.uint8)

         structure2d = _circle_structure(3)
         binary_img = ndimage.morphology.binary_closing(binary_img, structure2d)
         binary_img = ndimage.morphology.binary_fill_holes(binary_img)
         
         #structure2d = _circle_structure(2)
         #binary_img = ndimage.morphology.binary_opening(binary_img, structure2d)
         bone_mask[k,:,:] = binary_img

     img2save = np.zeros(bone_mask[test_idx,:,:].shape, dtype=np.uint8)
     img2save[bone_mask[test_idx,:,:]==1] = 255
    #  cv2.imwrite("2.jpg",img2save)
     
     landmarks = chest_and_iliac_slice_estimation(volume,
                                                  body_mask,
                                                  bone_mask,
                                                  voxel_spacing)
     print("landmarks: ",landmarks)

     # second round (additive extraction in ribs)
     for k in range(landmarks[1]):
         tmp_img = ((volume[k] >= 200) & (body_mask[k] == 1)).astype(np.uint8)
         structure2d = _circle_structure(3)
         binary_img = ndimage.morphology.binary_closing(tmp_img, structure2d)
         binary_img = ndimage.morphology.binary_fill_holes(binary_img)
         
         if k == test_idx:
             img2save = np.zeros(binary_img.shape, dtype=np.uint8)
             img2save[binary_img==1] = 255
            #  cv2.imwrite("3.jpg",img2save)
         
         bone_mask[k, :, :] = ((binary_img == 1) | (bone_mask[k] == 1)).astype(np.uint8)
             

     # structure1 = _spherical_structure(2.5)
     # bone_mask = ndimage.morphology.binary_closing(bone_mask, structure1)

     img2save = np.zeros(bone_mask[test_idx, :, :].shape, dtype=np.uint8)
     img2save[bone_mask[test_idx, :, :]==1] = 255
    #  cv2.imwrite("4.jpg",img2save)
    
     # extract the largest region
     structure_26 = np.ones([3, 3, 3])
     labeled, num_labels = ndimage.label(bone_mask, structure_26)
     histogram = ndimage.histogram(labeled, 1, num_labels, num_labels)
     largest_idx = np.argmax(histogram)
     bone_mask = (labeled == largest_idx + 1).astype(np.uint8)

     bone_mask = ndimage.morphology.binary_fill_holes(bone_mask)

     #bone_mask = ndimage.morphology.binary_erosion(bone_mask, structure_26)
     
     img2save = np.zeros(bone_mask[test_idx, :, :].shape, dtype=np.uint8)
     img2save[bone_mask[test_idx, :, :]==1] = 255
    #  cv2.imwrite("5.jpg",img2save)
     
     return bone_mask.astype(np.uint8)


def _extract_body_trunc_in_axial_slice(img, threshold):
    
    #binary_img = (img >= threshold).astype(np.uint8)
    binary_img = ((threshold <= img) & (img <= 1000)).astype(np.uint8)

    # remove bed etc.
    structure = _circle_structure(2)
    binary_img = ndimage.morphology.binary_opening(binary_img, structure)
    #binary_img = ndimage.morphology.binary_closing(binary_img, structure)    
    binary_img = ndimage.morphology.binary_fill_holes(binary_img)
    
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


def body_trunk_extraction(volume):
    
    body_mask = np.zeros(volume.shape, dtype=np.uint8)
 
    threshold = -150 # [HU]
     
    for k in range(volume.shape[0]):
        img = volume[k,:,:]
        binary_img = _extract_body_trunc_in_axial_slice(img, threshold)
        body_mask[k,:,:] = binary_img
    
    # extract the largest region
    structure = np.ones([3,3,3])
    labeled, num_labels = ndimage.label(body_mask, structure)    
        
    histogram = ndimage.histogram(labeled, 1, num_labels, num_labels)
    largest_idx = np.argmax(histogram)
    body_mask = (labeled == largest_idx + 1).astype(np.uint8)

    # refinement 
    radius = 3        
    structure = _spherical_structure(radius)
    body_mask = np.pad(body_mask, radius+1, 'constant')
    body_mask = ndimage.morphology.binary_closing(body_mask, structure)    
    body_mask = ndimage.morphology.binary_fill_holes(body_mask)

    body_mask = body_mask[radius+1:radius+1+volume.shape[0],\
                          radius+1:radius+1+volume.shape[1],
                          radius+1:radius+1+volume.shape[2]]
    
    img2save = np.zeros(body_mask[test_idx,:,:].shape, dtype=np.uint8)
    img2save[body_mask[test_idx,:,:]==1] = 255
    # cv2.imwrite("1.jpg",img2save)
    
    return body_mask


def main():

    base_path = "PATH" # dir that have mhd and raw files.
    in_file_name = os.path.join(base_path, "original.mhd")
    out_file_name = os.path.join(base_path, "original.raw")

    volume = skimage.io.imread(in_file_name, plugin='simpleitk')

    body_mask = body_trunk_extraction(volume)

    with open(out_file_name, "wb") as fp:
        fp.write(body_mask.flatten('C'))

    # create body_mask.mhd
    file_dir, raw_file_name = os.path.split(out_file_name)
    raw_file_root, _ = os.path.splitext(raw_file_name)
    mhd_file_name = os.path.join(file_dir, raw_file_root + ".mhd")

    src_file_root, _ = os.path.splitext(os.path.basename(in_file_name))
    mhd_str = ""

    with open(in_file_name, "r") as fp:
        mhd_str = fp.read()

    mhd_str = mhd_str.replace("MET_SHORT", "MET_UCHAR")
    mhd_str = mhd_str.replace(src_file_root, raw_file_root)

    with open(mhd_file_name, "w") as fp:
        fp.write(mhd_str)


if __name__ == '__main__':
    main()
