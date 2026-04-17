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
from dataSetCreat import *
import dice
import mhd_io
from attUnet import AttentionUNet
from unet import * 
from Unet_plus import *
from Unet_plus_3ch_in import *

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

    selected_volume = np.zeros_like(mask_volume).astype(np.uint8)
    total_volume = np.sum([region.area for region in regions])
    accumulated_volume = 0

    for region in regions:
        if accumulated_volume / total_volume > volume_threshold:
            break
        accumulated_volume += region.area
        for coords in region.coords:
            selected_volume[coords[0], coords[1], coords[2]] = 1

    for k in range(selected_volume.shape[0] - 1):
        loc = selected_volume[k,:,:]
        next = selected_volume[k+1,:,:]
        if np.sum(loc) > 0 and np.sum(loc) < np.sum(next)/4:
            print(f"k:{k} loc:{np.sum(loc)}, next:{np.sum(next)}.")
            selected_volume[k,:,:] = np.zeros_like(selected_volume[k,:,:])
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
    out_mhd = np.zeros_like(mhd_masks[0])
    out_mhd[mhd_masks[1] == 1] = 10
    out_mhd[mhd_masks[0] == 1] = 1
    mhd_io.export_to_mhd_and_raw(f"{output_path}/mhd/{case_name}_prediction.mhd", out_mhd, spacing)

def evaluation(case_name,class_num,all_outputs,all_labels,all_ori,export_mask,result_file_name,output_path,binarize_threshold,general=False):
    print(f"----------------------{case_name}-----------------------------")
    mhd_masks = []
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
            mask_volume = ndimg.binary_opening(mask_volume, structure=structure_3d)
            
        if i == 1:
            structure_element = np.zeros((3, 1, 1))
            structure_element[:, 0, 0] = 1
            mask_volume = ndimg.binary_closing(mask_volume,structure=structure_element)
            # mask_volume = ndimg.binary_opening(mask_volume)
        label_volume = combine_masks(all_labels[i])
        mask_volume = enhance_connectivity(mask_volume)
        if not general:
            processed_volume = select_regions_and_filter(mask_volume)
        else:
            processed_volume = mask_volume
        # processed_volume = mask_volume
        fp = np.sum((processed_volume == 1) & (label_volume == 0))
        fn = np.sum((processed_volume == 0) & (label_volume == 1))
        tp = np.sum((processed_volume == 1) & (label_volume == 1))
        total = np.sum(label_volume != 0)
        fp_ratio = fp / total if total != 0 else 0
        fn_ratio = fn / total if total != 0 else 0
        tp_ratio = tp / total if total != 0 else 0
        
        before_dice = dice.dice_numpy(mask_volume,label_volume)
        final_dice = dice.dice_numpy(processed_volume,label_volume)
        if export_mask:
            for k in range(processed_volume.shape[0]):
                pre_img = processed_volume[k,:,:]
                label_img = label_volume[k,:,:]
                ori_img = all_ori[k]
                # mask_file_name = f'{output_path}/test_images/mask_th{binarize_threshold:.3f}_{k:05d}.png'
                mask_file_name = f'{output_path}/test_images/ch{i}_mask_th{binarize_threshold:.3f}_{k:05d}.png'
                output = overlay(pre_img,label_img,ori_img)
                cv2.imwrite(mask_file_name, output)

        with open(result_file_name,'a') as fp:
            fp.write(f"----------------------{case_name}-----------------------------\n")
            fp.write("Mean of Dice coefficient(ch%.1f): %.4f ||" %
            (i, before_dice))
            fp.write("Mean of Dice coefficient after post-processing(ch%.1f): %.4f\n" %
            (i, final_dice))
            fp.write("True Positive Ratio(ch%.1f): %.4f\n" % (i, tp_ratio))
            fp.write("False Positive Ratio(ch%.1f): %.4f\n" % (i, fp_ratio))
            fp.write("False Negative Ratio(ch%.1f): %.4f\n" % (i, fn_ratio))
        mhd_masks.append(processed_volume)
        
        print("Mean of Dice coefficient(ch%.1f): %.4f " %
            (i, before_dice))
        print("Mean of Dice coefficient after post-processing(ch%.1f): %.4f\n" %
            (i, final_dice))
        print("True Positive Ratio(ch%.1f): %.4f" % (i, tp_ratio))
        print("False Positive Ratio(ch%.1f): %.4f" % (i, fp_ratio))
        print("False Negative Ratio(ch%.1f): %.4f\n" % (i, fn_ratio))
        
    return mhd_masks

def test(test_data_path, model_file_name, output_path, original_path,
         binarize_threshold=0.51, gpu_id="0",
         export_mask=False, time_stamp="", model=1, deep_supervision=False, **kwargs):

    if not os.path.isdir(test_data_path):
        print(f"Error: Path of test data ({test_data_path}) is not found.")
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
        if deep_supervision:
            modelname = "UnetPlus_DSV"
    if model == 3:
        modelname = 'UnetPlus_3ch'
    result_file_name = f'{output_path}/test_result_th{binarize_threshold:.3f}_{time_stamp}_{modelname}.csv'

    sample_batch = next(iter(test_loader))
    in_channels = sample_batch['original'].shape[1]
    out_channels = sample_batch['mask'].shape[1]
    class_num = 2
    first_filter_num = kwargs.get('first_filter_num', 16)
    print(f"in-channel: {in_channels} out-channel: {out_channels}")

    # Load the parameter of model
    if model == 0: 
        model = Unet(in_channels, out_channels, first_filter_num)
    if model == 1:
        model = AttentionUNet(in_channels, out_channels, first_filter_num)
    if model == 2:
        model = NestedUNet(in_channels, out_channels, first_filter_num, deep_supervision)
    if model == 3:
        model = NestedUNet_3ch(in_channels, out_channels, first_filter_num)
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    model = model.to(device)
    model.eval()

    # dice_coeff_arr = np.zeros(test_dataset.__len__())
    dice_coeff_arr = np.zeros((out_channels,test_dataset.__len__()))
    file_list = []
    case_split_index = [0]
    ori_index = []
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        all_ori = []
        for i in range(class_num): # multi-channels OR number of classes
            all_labels.append([])
            all_outputs.append([])
        
        for batch_idx, data in enumerate(test_loader):
            data, labels = data['original'].to(device), data['mask'].to(device)
            outputs = model(data)
            if deep_supervision:
                outputs = torch.mean(torch.stack(outputs, dim=0), dim=0)
            filename = test_dataset.get_filename(batch_idx)[0]
            case_number = filename.split("_")[0]
            if case_number not in file_list:
                file_list.append(case_number)
                if batch_idx != 0:
                    ori_end_slice = int(test_dataset.get_filename(batch_idx-1)[0].split("_")[-1][:-4])
                    ori_index.append((ori_end_slice-(batch_idx-1-case_split_index[-1]), ori_end_slice))
                    case_split_index.append(batch_idx)
            # print(file_list, ori_index)
            
            all_ori.append((data*255).cpu())
            # outmask = torch.softmax(outputs,dim=1)
            for i in range(class_num): # multi-channels OR number of classes(outputs.shape[1])
                out_mask, label_img = labeling(outputs[:, i:i+1, :, :],labels[:, i:i+1, :, :],binarize_threshold)
                out_mask = ndimg.binary_fill_holes(out_mask) # hole filling
                # for softmax
                # if i == 0:
                #     out_mask = ndimg.binary_fill_holes(out_mask) # hole filling
                # if i == 1:
                #     out_mask = out_mask - all_outputs[0][-1]
                #     label_img = label_img - all_labels[0][-1]
                
                dice_coeff_arr[i][batch_idx] = dice.dice_numpy(out_mask, label_img)
                # out_mask, label_img = labeling(outputs,labels,binarize_threshold)
                # dice_coeff_arr[batch_idx] = dice.dice_numpy(out_mask, label_img)
                
                all_labels[i].append(label_img)
                all_outputs[i].append(out_mask)
                if False:
                    mask_file_name = f'{output_path}/test_images/ch{i}_mask_th{binarize_threshold:.3f}_{batch_idx:05d}.png'
                    out = outputs[:, i:i+1, :, :].cpu().numpy()
                    cv2.imwrite(mask_file_name, out[0, 0, :, :]*255)

                # print(f"{batch_idx},ch{i},{dice_coeff_arr[i][batch_idx]:.4f}") # multi-channels
                # with open(result_file_name, "a") as fp:
                #     fp.write(f"{batch_idx},ch{i},{dice_coeff_arr[i][batch_idx]:.4f}\n")
                # print(f"{batch_idx},{dice_coeff_arr[batch_idx]:.4f}")
                # with open(result_file_name, "a") as fp:
                #     fp.write(f"{batch_idx},{dice_coeff_arr[batch_idx]:.4f}\n")
    eval_vals = dice_coeff_arr
    print(f"Model file name: {model_file_name}")
    with open(result_file_name,'a') as fp: # multi-channels
        fp.write(f"Model file name: {model_file_name}\n")
        fp.write("Mean of Dice coefficient: ch0:%.4f (%.4f - %.4f) ch1:%.4f (%.4f - %.4f) totalAVG:%.4f\n" %
          (np.mean(eval_vals[0]), np.min(eval_vals[0]), np.max(eval_vals[0]), np.mean(eval_vals[1]), np.min(eval_vals[1]), np.max(eval_vals[1]), np.mean(eval_vals)))
    # with open(result_file_name,'a') as fp:
    #     fp.write("Mean of Dice coefficient: %.4f (%.4f - %.4f)\n" %
    #       (np.mean(eval_vals), np.min(eval_vals), np.max(eval_vals)))
    

    # multi-channels
    ori_end_slice = int(test_dataset.get_filename(batch_idx)[0].split("_")[-1][:-4])
    ori_index.append((ori_end_slice-(batch_idx-case_split_index[-1]), ori_end_slice))
    case_split_index.append(np.shape(all_outputs)[1])
    all_ori = np.array(all_ori)
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    # general
    evaluation("General",class_num,all_outputs,all_labels,all_ori,export_mask,result_file_name,output_path,binarize_threshold,True)
    
    # print(file_list)
    # print(case_split_index)
    # print(ori_index)
    for case_name in file_list:
        index = file_list.index(case_name)
        
        # find mhd infor
        filename = test_dataset.get_filename(case_split_index[index])[0]
        name1, name2 = filename.split('_')[0], filename.split('_')[1]
        origin_mhd_path = os.path.join(original_path, name1, name2, 'stent_mask.mhd')
        origin_mhd = skimage.io.imread(origin_mhd_path, plugin='simpleitk')
        spacing = mhd_io.get_voxel_spacing_from_mhd(origin_mhd_path)
        dimension = mhd_io.get_dimension_from_mhd(origin_mhd_path)
        if in_channels == 3:
            dimension = (dimension[0], dimension[1], int(spacing[2]/2.5*dimension[2]))
            spacing = (spacing[0], spacing[1], 2.5)
            
        # cut from all test slices
        case_ori = all_ori[case_split_index[index]:case_split_index[index+1]]
        case_masks = all_outputs[:,case_split_index[index]:case_split_index[index+1],:,:]
        case_labels = all_labels[:,case_split_index[index]:case_split_index[index+1],:,:]
        mhd_masks = evaluation(case_name, class_num,case_masks,case_labels,case_ori,False,result_file_name,output_path,binarize_threshold)
        # print(mhd_masks[0].shape)
        # print(dimension)
        # print(dimension[2], ori_index[index][1], dimension[2]-ori_index[index][1]-1)
        print(f"ori_index[index][0]: {ori_index[index][0]}")
        v_before = np.zeros((ori_index[index][0], dimension[0], dimension[1]))
        v_after = np.zeros((dimension[2]-ori_index[index][1]-1, dimension[0], dimension[1]))
        # print(v_before.shape)
        # print(v_after.shape)
        for channel in range(len(mhd_masks)):
            final_mhd = np.concatenate((v_before, mhd_masks[channel]))
            final_mhd = np.concatenate((final_mhd, v_after))
            # print(final_mhd.shape)
            mhd_masks[channel] = final_mhd
        save_mhd(mhd_masks, case_name, output_path, spacing, dimension)
    
    
    return dice_coeff_arr


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
    parser.add_argument('--deepsupervision',
                        help='Threshold to binarize outputs',
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
                     args.deepsupervision,
                     **hyperparameters_dict)

# python sample/test_segmentation.py validation/ output/best_model_best_20221031194537.pth output/ --export_mask