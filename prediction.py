import argparse
from datetime import datetime as dt
import imageio
import numpy as np
import os
import sys
import torch
import scipy.ndimage as ndimg
from skimage.measure import label, regionprops
import skimage.io

from overlay import *
from predDataSetCreat import *
import dice
import mhd_io
from attUnet import AttentionUNet
from unet import * 
from Unet_plus import *
from Unet_plus_3ch_in import *

def spherical_structure(radius):
    L = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(L, L, L)
    sphere = (X**2 + Y**2 + Z**2) <= radius**2
    return sphere.astype(np.bool_)

def circular_structure(radius):
    L = np.arange(-int(np.ceil(radius)), int(np.ceil(radius)) + 1)
    X, Y = np.meshgrid(L, L, indexing='ij')
    circle = (X**2 + Y**2) <= radius**2
    return circle

def combine_masks(outputs):
    all_masks = np.stack([output for output in outputs])
    return all_masks

def select_regions_and_filter(mask_volume, volume_threshold=0.8):

    labeled_mask = label(mask_volume)
    regions = regionprops(labeled_mask)

    regions = sorted(regions, key=lambda x: x.area, reverse=True)
    largest_region = regions[0]
    
    selected_volume = np.zeros_like(mask_volume, dtype=np.uint8)
    for z, y, x in largest_region.coords:
        selected_volume[z, y, x] = 1

    # maintain regions add up to 80% of whole volume
    # selected_volume = np.zeros_like(mask_volume).astype(np.uint8)
    # total_volume = np.sum([region.area for region in regions])
    # accumulated_volume = 0

    # for region in regions:
    #     if accumulated_volume / total_volume > volume_threshold:
    #         break
    #     accumulated_volume += region.area
    #     for coords in region.coords:
    #         selected_volume[coords[0], coords[1], coords[2]] = 1

    depth = selected_volume.shape[0]
    for k in range(depth - 1):
        loc_count = np.sum(selected_volume[k])
        next_count = np.sum(selected_volume[k+1])
        if loc_count > 0 and loc_count < next_count / 2:
            print(f"filtered layer k={k}: count={loc_count}, next={next_count}")
            selected_volume[k] = 0
    return selected_volume

def enhance_connectivity(volume):
    depth = volume.shape[0]
    for i in range(1, depth - 1):
        slice1, slice2, slice3 = volume[i - 1], volume[i], volume[i + 1]
        
        dice12 = dice.dice_numpy(slice1, slice2)
        dice23 = dice.dice_numpy(slice2, slice3)
        dice13 = dice.dice_numpy(slice1, slice3)
        
        if dice13 > dice12 + 0.1 and dice13 > dice23 + 0.1:
            intersection_13 = np.logical_and(slice1, slice3)
            volume[i] = intersection_13.astype(int)
    
    return volume

def labeling(outputs, labels, binarize_threshold):
    out = outputs.cpu().numpy()
    out_mask = (out[0, 0, :, :] >= binarize_threshold).astype(np.uint8)
    label_img = labels.cpu().numpy()
    label_img = (label_img[0, 0, :, :] >= binarize_threshold).astype(np.uint8)
    
    # labeling 
    label = ndimg.label(out_mask)
    areas = np.array(ndimg.sum(out_mask, label[0], np.arange(label[0].max()+1)))
    mask = areas > (sum(areas) * 0.25)
    out_mask = mask[label[0]].astype(np.uint8)
    
    return out_mask, label_img

def save_mhd(mhd_masks, case_name, output_path, spacing, dimension):
    out_mhd = np.zeros_like(mhd_masks[0]).astype(np.uint8)
    out_mhd[mhd_masks[1] == 1] = 10
    out_mhd[mhd_masks[0] == 1] = 1
    mhd_io.export_to_mhd_and_raw(f"{output_path}/mhd/{case_name}_prediction.mhd", out_mhd, spacing)

def save_separate_mhd(mhd_mask, case_name, output_path, spacing, dimension, class_id):
    out_mhd = np.zeros_like(mhd_mask).astype(np.uint8)
    out_mhd[mhd_mask == 1] = 1
    
    case_id, case_date = case_name.split("_",1)
    target_dir = os.path.join(output_path, "mhd", case_id, case_date)
    os.makedirs(target_dir, exist_ok=True)
    mhd_file_path = os.path.join(target_dir, f"{class_id}.mhd")
    
    mhd_io.export_to_mhd_and_raw(mhd_file_path, out_mhd, spacing)
    # mhd_io.export_to_mhd_and_raw(f"{output_path}/mhd/{case_name}_{class_id}_prediction.mhd", out_mhd, spacing)

def evaluation(case_name,class_num,all_outputs,all_labels,all_ori,export_mask,result_file_name,output_path,binarize_threshold,general=False):
    print(f"----------------------{case_name}-----------------------------")
    mhd_masks = []
    stent_start = 0
    for i in range(class_num): # out_channels OR number of classes
        mask_volume = combine_masks(all_outputs[i])
        if i == 0:
            structure_element = np.zeros((3, 1, 1))
            structure_element[:, 0, 0] = 1
            structure_3d = ndimg.generate_binary_structure(rank=3, connectivity=3)
            structure = circular_structure(7)
            mask_volume = ndimg.binary_closing(mask_volume, structure=structure_element)
            for z in range(mask_volume.shape[0]):
                mask_volume[z] = ndimg.binary_opening(mask_volume[z], structure=structure)
            # mask_volume = ndimg.binary_opening(mask_volume, structure=structure_3d)
            
        if i == 1:
            structure_element = np.zeros((3, 1, 1))
            structure_element[:, 0, 0] = 1
            mask_volume = ndimg.binary_closing(mask_volume,structure=structure_element)
        mask_volume = enhance_connectivity(mask_volume)
        processed_volume = select_regions_and_filter(mask_volume)

        if i == 0:
            for index in range(0, processed_volume.shape[0]):
                if np.any(processed_volume[index]):
                    print("index: ",index)
                    stent_start = index
                    break
        if i == 1:
            for index in range(0, stent_start):
                if np.any(processed_volume[index]):
                    processed_volume[index][processed_volume[index] > 0] = 0
        
        if export_mask:
            for k in range(processed_volume.shape[0]):
                pre_img = processed_volume[k,:,:]
                label_img = np.zeros_like(pre_img)
                ori_img = all_ori[k]
                # mask_file_name = f'{output_path}/test_images/mask_th{binarize_threshold:.3f}_{k:05d}.png'
                mask_file_name = f'{output_path}/test_images/ch{i}_mask_th{binarize_threshold:.3f}_{k:05d}.png'
                output = overlay(pre_img,label_img,ori_img)
                cv2.imwrite(mask_file_name, output)

        mhd_masks.append(processed_volume)
        
    return mhd_masks

def test(test_data_path, model_file_name, output_path, original_path,
         binarize_threshold=0.51, gpu_id="0",
         export_mask=False, time_stamp="", model=1, data_creation=False, **kwargs):
    if not os.path.isdir(test_data_path):
        print(f"Error: Path of prediction data ({test_data_path}) is not found.")
        sys.exit(1)

    # Check model file is exist
    if not os.path.isfile(model_file_name):
        print(f"Error: Model file ({model_file_name}) is not found.")
        sys.exit(1)   
        
    if time_stamp == "":
        time_stamp = dt.now().strftime('%Y%m%d%H%M%S')
        
    # Set ID of CUDA device
    device = f'cuda:{gpu_id}'
    print(f"Device: {device}")

    test_dataset = CTImagesDataset(root_dir=test_data_path,
                                        transform=transforms.Compose([
                                            ToTensor()
                                        ]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    modelname = 'None'     
    if model == 0:
        modelname = 'Unet'
    if model == 1:
        modelname = 'AttUnet'
    if model == 2:
        modelname = 'UnetPlus'
    if model == 3:
        modelname = 'UnetPlus_3ch'

    sample_batch = next(iter(test_loader))
    in_channels = sample_batch['original'].shape[1]
    out_channels = 2
    class_num = 2
    first_filter_num = kwargs.get('first_filter_num', 16)
    print(f"in-channel: {in_channels} out-channel: {out_channels}")

    # Load the parameter of model
    if model == 0: 
        model = Unet(in_channels, out_channels, first_filter_num)
    if model == 1:
        model = AttentionUNet(in_channels, out_channels, first_filter_num)
    if model == 2:
        model = NestedUNet(in_channels, out_channels, first_filter_num)
    if model == 3:
        model = NestedUNet_3ch(in_channels, out_channels, first_filter_num)
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    model = model.to(device)
    model.eval()

    file_list = []
    case_split_index = [0]
    ori_index = []
    with torch.no_grad():
        all_outputs = []
        all_ori = []
        for i in range(class_num): # multi-channels OR number of classes
            all_outputs.append([])
        
        for batch_idx, data in enumerate(test_loader):
            data = data['original'].to(device)
            outputs = model(data)
            filename = test_dataset.get_filename(batch_idx)
            case_number = filename.split("_")[0] + "_" + filename.split("_")[1]
            if case_number not in file_list:
                file_list.append(case_number)
                if batch_idx != 0:
                    ori_end_slice = int(test_dataset.get_filename(batch_idx-1).split("_")[-1][:-4])
                    ori_index.append((ori_end_slice-(batch_idx-1-case_split_index[-1]), ori_end_slice))
                    case_split_index.append(batch_idx)
            # print(file_list, batch_idx)
            
            all_ori.append((data*255).cpu())
            for i in range(class_num): # multi-channels OR number of classes(outputs.shape[1])
                out_mask, _ = labeling(outputs[:, i:i+1, :, :],outputs[:, i:i+1, :, :],binarize_threshold)
                out_mask = ndimg.binary_fill_holes(out_mask) # hole filling
                
                all_outputs[i].append(out_mask)
                if False:
                    mask_file_name = f'{output_path}/test_images/ch{i}_mask_th{binarize_threshold:.3f}_{batch_idx:05d}.png'
                    out = outputs[:, i:i+1, :, :].cpu().numpy()
                    cv2.imwrite(mask_file_name, out[0, 0, :, :]*255)
                    
    print(f"Model file name: {model_file_name}")

    # multi-channels
    ori_end_slice = int(test_dataset.get_filename(batch_idx).split("_")[-1][:-4])
    ori_index.append((ori_end_slice-(batch_idx-case_split_index[-1]), ori_end_slice))
    case_split_index.append(np.shape(all_outputs)[1])
    all_ori = np.array(all_ori)
    all_outputs = np.array(all_outputs)

    
    print(file_list)
    # print(case_split_index)
    # print(ori_index)
    for case_name in file_list:
        index = file_list.index(case_name)
        
        # find mhd infor
        filename = test_dataset.get_filename(case_split_index[index])
        print(f"filename: {filename}^^^^^^^^^^^^^^^^)")
        name1, name2, z_spacing = filename.split('_')[0], filename.split('_')[1], filename.split("_")[2]
        # origin_mhd_path = os.path.join(original_path, name1, name2, f'plane_{z_spacing}/original.mhd')
        origin_mhd_path = os.path.join(original_path, name1, name2, 'original.mhd')
        origin_mhd = skimage.io.imread(origin_mhd_path, plugin='simpleitk')
        spacing = mhd_io.get_voxel_spacing_from_mhd(origin_mhd_path)
        dimension = mhd_io.get_dimension_from_mhd(origin_mhd_path)
        
        # cut from all test slices
        case_ori = all_ori[case_split_index[index]:case_split_index[index+1]]
        case_masks = all_outputs[:,case_split_index[index]:case_split_index[index+1],:,:]
        case_labels = np.zeros_like(case_masks)
        mhd_masks = evaluation(case_name, class_num,case_masks,case_labels,case_ori,False,"_",output_path,binarize_threshold)
        # print(mhd_masks[0].shape)
        # print(dimension)
        v_before = np.zeros((ori_index[index][0], dimension[0], dimension[1]))
        v_after = np.zeros((dimension[2]-ori_index[index][1]-1, dimension[0], dimension[1]))
        # print(v_before.shape)
        # print(v_after.shape)
        for channel in range(len(mhd_masks)):
            final_mhd = np.concatenate((v_before, mhd_masks[channel]))
            final_mhd = np.concatenate((final_mhd, v_after))
            # print(final_mhd.shape)
            mhd_masks[channel] = final_mhd
        
        if not data_creation:
            save_mhd(mhd_masks, case_name, output_path, spacing, dimension)
        else:
            save_separate_mhd(mhd_masks[0], case_name, output_path, spacing, dimension, "stent")
            save_separate_mhd(mhd_masks[1], case_name, output_path, spacing, dimension, "arterial")
    
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Test segmentation',
        add_help=True)
    parser.add_argument('test_data_path', help='Path of test data')
    parser.add_argument('model_file_name', help='File name of trained model')
    parser.add_argument('output_path', help='Path of output data')
    parser.add_argument('original_data_path', help='Path of output data')
    parser.add_argument('-g', '--gpu_id', help='GPU IDs',
                        type=str, default='0')
    parser.add_argument('-f', '--first_filter_num',
                        help='Number of the first filter in U-Net',
                        type=int, default=16)
    parser.add_argument('-t', '--binarize_threshold',
                        help='Threshold to binarize outputs',
                        type=float, default=0.51)
    parser.add_argument('--export_mask',
                        help='Export output mask as png file',
                        action='store_true')
    parser.add_argument('--time_stamp', help='Time stamp for saved data',
                        type=str, default='')        
    parser.add_argument('-mo','--model', help='Time stamp for saved data',
                        type=int, default=0)        
    parser.add_argument('--data_creation',
                        help='Data creation?',
                        action='store_true')        

    args = parser.parse_args()

    hyperparameters_dict = {"first_filter_num": args.first_filter_num}

    eval_vals = test(args.test_data_path,
                     args.model_file_name,
                     args.output_path,
                     args.original_data_path,
                     args.binarize_threshold,
                     args.gpu_id,
                     args.export_mask,
                     args.time_stamp,
                     args.model,
                     args.data_creation,
                     **hyperparameters_dict)

# python sample/test_segmentation.py validation/ output/best_model_best_20221031194537.pth output/ --export_mask