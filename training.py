import torch
import os
import argparse
import numpy as np
import random
import time
from datetime import datetime as dt
from torchvision import transforms
import matplotlib.pyplot as plt

from dataSetCreat import *
from attUnet import * 
from Unet_plus import *
from Unet_plus_3ch_in import *
from unet import *
import dice
import diceCE
import test

def show_images(sample,idx):
    original = sample['original'][0].numpy()
    mask1 = sample['mask'][0].numpy()
    mask2 = sample['mask'][1].numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Transformed Original')
    axs[1].imshow(mask1, cmap='gray')
    axs[1].set_title('Transformed Mask Channel 1')
    axs[2].imshow(mask2, cmap='gray')
    axs[2].set_title('Transformed Mask Channel 2')
    plt.savefig(f'aug/transformed_sample_{idx}.png')
    plt.close()

def training(training_data_path, validation_data_path, output_path,
             first_filter_num=64, learning_rate=0.001, beta_1=0.99,
             batch_size=8, max_epoch_num=25,binarize_threshold=0.5,
             gpu_id="0", model=0, augmentation=False, time_stamp="", deep_supervision=False):
    # Fix seed
    seed_num = 234567
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    
    device = f'cuda:{gpu_id}'
    
    # Creation of output folder
    if not os.path.isdir(output_path):
        print(f"Path of output data ({output_path}) is created automatically.")
        os.makedirs(output_path) 
    if time_stamp == "":
            time_stamp = dt.now().strftime('%Y%m%d%H%M%S')
       
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
    
    general_log_file_name = f"{output_path}/csvFiles/general_log.csv"
    loss_log_file_name = f"{output_path}/csvFiles/loss_log_{time_stamp}_{modelname}.csv"
    model_file_name = f"{output_path}/models/model_best_{time_stamp}_{modelname}.pth"
    
    print(f"Device: {device} First_filter={first_filter_num} Beta={beta_1} LR={learning_rate} batch={batch_size} model={modelname} epoch_num={max_epoch_num}")
    with open(loss_log_file_name,'a') as f:
        f.write(f"First_filter={first_filter_num} Beta={beta_1} LP={learning_rate} batch={batch_size} epoch_num={max_epoch_num}\n")
    
    # Use the dataset with DataLoader
    aug_scale = 1
    if augmentation:
        transform = transforms.Compose([ToTensor(),
                                        ApplyElasticTransform(alpha=80.0, sigma=5.0, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
                                        RandomRotation((-10, 10))
                                        ])
        aug_scale = 2
        print("Augmentation active.")
    else:
        transform = transforms.Compose([ToTensor()])
    transformed_dataset = CTImagesDataset(root_dir=training_data_path,
                                        transform=transform, augment=augmentation,aug_scale=aug_scale)
    validation_dataset = CTImagesDataset(root_dir=validation_data_path,
                                        transform=transforms.Compose([ToTensor()]))

    training_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    print(f"Data loaded from {training_data_path} and {validation_data_path}. Length {transformed_dataset.__len__()}")
    
    for i, data in enumerate(training_loader):
        # show_images({'original': data['original'][0], 'mask': data['mask'][0]}, i)
        for j in range(data['original'].size(0)):
            show_images({'original': data['original'][j], 'mask': data['mask'][j]}, j)
        # if i > 100:
        break
    
    sample_batch = next(iter(training_loader))
    in_channels = sample_batch['original'].shape[1]
    out_channels = sample_batch['mask'].shape[1]
    class_num = 2
    # print("Original image dimensions:", sample_batch['original'].shape)  # output: (batch_size, 1, height, width)
    # print("Mask image dimensions:", sample_batch['mask'].shape) 
    print(f"in-channel: {in_channels} out-channel: {out_channels}")
    
    # first filter num?
    if model == 0: 
        model = Unet(in_channels, out_channels, first_filter_num)
    if model == 1:
        model = AttentionUNet(in_channels, out_channels, first_filter_num)
    if model == 2:
        model = NestedUNet(in_channels, out_channels, first_filter_num, deep_supervision)
    if model == 3:
        model = NestedUNet_3ch(in_channels, out_channels, first_filter_num)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=learning_rate,
                            betas=(beta_1, 0.999))
    # criterion = diceCE.DiceCELoss()
    criterion = dice.DiceLoss()
    
    print("Start training...")
    best_validation_loss = float('inf')
    best_train_loss = float('inf')
    previous = 0
    count = 0
    earlystop = 10
    for epoch in range(max_epoch_num):
        training_loss = 0
        validation_loss = 0
        
        # training
        model.train()
        
        for batch_idx, data in enumerate(training_loader):
            inputs, labels = data['original'].to(device), data['mask'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if deep_supervision:
                loss_list = [criterion(out, labels) for out in outputs]
                loss = sum(loss_list) / len(loss_list)
            else:
                loss = criterion(outputs, labels)

            training_loss += loss.item()

            loss = torch.sum(loss)

            loss.backward()
            optimizer.step()
        avg_training_loss = training_loss / (batch_idx + 1)
    
        # validation
        model.eval() 
        dice_coeff_arr = np.zeros((out_channels,validation_dataset.__len__()))
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                inputs, labels = data['original'].to(device), data['mask'].to(device)

                outputs = model(inputs)
                if deep_supervision:
                    loss_list = [criterion(out, labels) for out in outputs]
                    loss = sum(loss_list) / len(loss_list)
                    outputs = torch.mean(torch.stack(outputs, dim=0), dim=0)
                else:
                    loss = criterion(outputs, labels)
                validation_loss += loss.item()
                bi_outputs = torch.argmax(outputs,dim=1)
                
                for i in range(out_channels): # number of classes(outputs.shape[1])
                    # for sigmoid
                    out_mask, label_img = test.labeling(outputs[:,i][:,np.newaxis,:,:],labels[:,i][:,np.newaxis,:,:],binarize_threshold)
                    
                    # for softmax
                    # out_mask = (bi_outputs == i).to(torch.uint8)
                    # out_mask = out_mask.cpu().numpy()
                    
                    # label_img = labels[:,i][:,np.newaxis,:,:].cpu().numpy()
                    # label_img = (label_img[0, 0, :, :] >= binarize_threshold).astype(np.uint8)
                    
                    dice_coeff_arr[i][batch_idx] = dice.dice_numpy(out_mask, label_img)
                # out_mask, label_img = test.labeling(outputs,labels,binarize_threshold)
                # dice_coeff_arr[batch_idx] = dice.dice_numpy(out_mask, label_img)

        avg_validation_loss = validation_loss / (batch_idx + 1)
        # File save
        saved_str = ""
        if best_validation_loss > avg_validation_loss:
            best_validation_loss = avg_validation_loss
            best_train_loss = avg_training_loss
            torch.save(model.state_dict(), model_file_name)
            saved_str = " ==> model saved"
        
        eval_vals = dice_coeff_arr
        print("epoch %d: train_loss:%.4f val_loss:%.4f %s dice_ch0:%.4f (%.4f - %.4f) dice_ch1:%.4f (%.4f - %.4f)" %
            (epoch + 1, avg_training_loss, avg_validation_loss, saved_str, 
            np.mean(eval_vals[0]), np.min(eval_vals[0]), np.max(eval_vals[0]),
            np.mean(eval_vals[1]), np.min(eval_vals[1]), np.max(eval_vals[1])))
        # softmax
        # print("dice_background:%.4f (%.4f - %.4f)" %
        #     (np.mean(eval_vals[2]), np.min(eval_vals[2]), np.max(eval_vals[2])))
        # print("epoch %d: train_loss:%.4f val_loss:%.4f %s dice:%.4f (%.4f - %.4f)" %
        #     (epoch + 1, avg_training_loss, avg_validation_loss, saved_str, 
        #     np.mean(eval_vals), np.min(eval_vals), np.max(eval_vals)))
        with open(loss_log_file_name, "a") as fp:
            fp.write("%d,%.4f,%.4f,%.4f,%.4f\n" %
                    (epoch + 1, avg_training_loss, avg_validation_loss, np.mean(eval_vals[0]), np.mean(eval_vals[1])))
            # fp.write("%d,%.4f,%.4f,%.4f\n" %
            #         (epoch + 1, avg_training_loss, avg_validation_loss, np.mean(eval_vals)))
        
            if abs(previous - avg_validation_loss) < 0.001:
                count += 1
            else:
                count = 0
            if count >= earlystop:
                fp.write(f"Because the val_loss didn't change during {earlystop} epoches. Program was stopped")
                with open(general_log_file_name, "a") as fp:
                    fp.write(f"{time_stamp},{modelname},{best_train_loss},{best_validation_loss},{first_filter_num},{beta_1},{learning_rate},{batch_size},{max_epoch_num},{aug_scale} early stopped\n")
                return model_file_name
            previous = avg_validation_loss
    
    print(f"Best validation loss: {best_validation_loss}")
    with open(loss_log_file_name, "a") as fp:
            fp.write(f"Best validation loss: {best_validation_loss}")
            
    with open(general_log_file_name, "a") as fp:
            fp.write(f"{time_stamp},{modelname},{best_train_loss},{best_validation_loss},{first_filter_num},{beta_1},{learning_rate},{batch_size},{max_epoch_num},{aug_scale}\n")
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AAA detection',
                                     add_help=True)
    parser.add_argument('training_data_path', help='Path of training data')
    parser.add_argument('validation_data_path', help='Path of validation data')
    parser.add_argument('output_path',
                        help='Path of output data')
    parser.add_argument('-g', '--gpu_id', help='GPU ID',
                        type=str, default='0')
    parser.add_argument('-f', '--first_filter_num',
                        help='Number of the first filter in U-Net',
                        type=int, default=16)
    parser.add_argument('-l', '--learning_rate',
                        help='Learning rate',
                        type=float, default=0.01)
    parser.add_argument('--beta_1',
                        help='Beta_1 for Adam',
                        type=float, default=0.9)
    parser.add_argument('-b', '--batch_size',
                        help='Batch size',
                        type=int, default=8)
    parser.add_argument('-m', '--max_epoch_num',
                        help='Maximum number of training epochs',
                        type=int, default=50)
    parser.add_argument('-t', '--binarize_threshold',
                        help='Threshold to binarize outputs',
                        type=float, default=0.51)
    parser.add_argument('-mo', '--model',
                        help='Select the training model',
                        type=int, default=1)
    parser.add_argument('-a', '--augmentation',
                        help='Data augmentation scale',
                        action='store_true')
    parser.add_argument('--time_stamp', help='Time stamp for saved data',
                        type=str, default='')
    parser.add_argument('--deepsupervision',
                        help='Threshold to binarize outputs',
                        action='store_true')

    args = parser.parse_args()

    random.seed(time.time())
    r_lr = 10**random.randint(-5,-1)
    r_f = 2*random.randint(2,8)
    r_beta = random.uniform(0.9,0.99)
    r_batch = 2**random.randint(1,3)
    
    # r_lr = 0.001
    # r_f = 16
    # r_beta = 0.943
    # r_batch = 8
    training(args.training_data_path,
            args.validation_data_path,
            args.output_path,
            r_f,
            r_lr,
            r_beta,
            r_batch,
            args.max_epoch_num,
            args.binarize_threshold,
            args.gpu_id,
            args.model,
            args.augmentation,
            args.time_stamp,
            args.deepsupervision)