import numpy as np
import cv2
import imageio

def overlay(out_mask, label_img, original_img):
    if original_img.ndim == 4:
        original_img = original_img.squeeze(0)
    # original_img = original_img.cpu().numpy()
    original_img = np.transpose(original_img, (1, 2, 0))
    color_image = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    # print(f"img max:{color_image.max()}, min:{color_image.min()}")

    # Define colors for overlay (in BGR)
    mask_color = [0, 255, 255]  # Light blue
    label_color =  [255, 0, 255] # Purple
    overlap_color =  [0, 255, 0] # Yellow

    # Create colored overlays
    # mask_overlay = np.zeros_like(color_image)
    # label_overlay = np.zeros_like(color_image)
    
    # Apply colors to the masks
    
    overlay = np.zeros_like(color_image)
    overlay[out_mask == 1] = mask_color
    overlay[label_img == 1] = label_color
    overlap_mask = (out_mask == 1) & (label_img == 1)
    overlay[overlap_mask] = overlap_color
    
    # Combine overlays with the original image
    alpha = 0.2  # Transparency factor
    color_image[overlap_mask] = color_image[overlap_mask] * (1 - alpha)
    combined_image = cv2.addWeighted(color_image, 1, overlay, alpha, 0)
    # combined_image = cv2.addWeighted(combined_image, 1, label_overlay, alpha, 0)

    return combined_image