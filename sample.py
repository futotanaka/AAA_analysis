from radiomics import featureextractor 

def compute_slice_features(img_slice, mask_slice, pixel_spacing,
                           radiomics_extractor, skin_label=14):
    """
    spacing : (sy, sx)
    mask_slice : label mask (2D)
    """

    # leg mask
    leg_mask = (mask_slice > 0).astype(np.uint8)

    # 周囲長
    perimeter_mm = calc_perimeter(leg_mask, pixel_spacing)

    # 皮膚厚
    sy, sx = pixel_spacing
    skin_area_mm2 = np.sum(mask_slice == skin_label) * sy * sx
    skin_thickness_mm = skin_area_mm2 / perimeter_mm if perimeter_mm > 0 else 0.0
    # skin_thickness_mm = np.sum(mask_slice == skin_label) / perimeter_mm

    # radiomics features in fat region
    img_sitk = sitk.GetImageFromArray(img_slice.astype(np.float32))
    fat_mask_sitk = sitk.GetImageFromArray((mask_slice == 1).astype(np.uint8))

    rad_raw = radiomics_extractor.execute(img_sitk, fat_mask_sitk)
    rad_features = filter_radiomics_features(rad_raw)

    # perimeter, thickness を追加
    rad_features["perimeter_mm"] = perimeter_mm
    rad_features["skin_thickness_mm"] = skin_thickness_mm

    # return rad_features
    return {
        "perimeter_mm": perimeter_mm,
        "skin_thickness_mm": skin_thickness_mm
    }


def extract_leg_features(org_volume, segmented_mask, voxel_spacing, output_file_name):
     # extract leg mask    
     leg_mask, target_slice = extract_leg_mask(mask_img, org_spacing, output_file_name)
    #  dst_img = sitk.GetImageFromArray(leg_mask.astype(np.uint8))
    #  sitk.WriteImage(dst_img, "leg_mask.mhd")
    #  print("leg mask exported to leg_mask.mhd")
     
     # 膝スライス抽出アルゴリズムを加える
     left = segmented_mask * (leg_mask == 2)
     right = segmented_mask * (leg_mask == 1)
     left_pos = find_patella(left, org_volume)
     print("left patella slice: ", left_pos)
     right_pos = find_patella(right, org_volume)
     print("right patella slice: ", right_pos)
     knee_slice = [right_pos, left_pos]
     
     # 各脚の特徴を抽出する
     dst_features = []
     voxels_in_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
     print(f"voxel spacing (mm): {voxel_spacing}")

     # define extractor for radiomnics features
     params = {
         'binWidth': 25,
         'resampledPixelSpacing': None,
         'interpolator': sitk.sitkBSpline,
         'enableCExtensions': True,
         'shape2D': True}
     radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(**params)

     for j, leg in enumerate(["right", "left"]):
        print(j, leg)

        features = {
            "leg": leg,
            "leg_volume": np.sum(leg_mask == j + 1) * voxels_in_mm3,
            "bone_volume": np.sum((leg_mask == j + 1) & (segmented_mask == 5)) * voxels_in_mm3,
            "muscle_volume": np.sum((leg_mask == j + 1) & (segmented_mask == 2)) * voxels_in_mm3,
            "fat_volume": np.sum((leg_mask == j + 1) & (segmented_mask == 1)) * voxels_in_mm3
        }

        #　最初のバージョン
        # slice_diff = int(round(200.0 / voxel_spacing[2]))
        # slice_index = [knee_slice[j] - slice_diff, knee_slice[j] + slice_diff]
    
        # for i, region in enumerate(["thigh", "lower"]):   # 大腿・下腿
         
        #     print(slice_index[i], org_volume.shape)
        #     img_slice = org_volume[slice_index[i]]
        #     mask_slice = segmented_mask[slice_index[i]]\
        #                * (leg_mask[slice_index[i]] == j+1).astype(np.uint8)
        #     spacing_2d = (voxel_spacing[1], voxel_spacing[2])

        #     features[region] = compute_slice_features(img_slice,
        #                                               mask_slice,
        #                                               spacing_2d,
        #                                               radiomics_extractor)         
         
        # dst_features.append(features) 

        thigh_mm = 200.0   # 上 
        lower_mm = 100.0   # 下

        thigh_diff = int(round(thigh_mm / voxel_spacing[2]))
        lower_diff = int(round(lower_mm / voxel_spacing[2]))
       
        slice_index = [
            # knee_slice[j] - thigh_diff  # thigh 的那一张
            knee_slice[j] + lower_diff    # lower 的那一张
        ]

        for i, region in enumerate(["lower"]):   # 大腿・下腿
         
            print(slice_index[i], org_volume.shape)
            img_slice = org_volume[slice_index[i]]
            mask_slice = segmented_mask[slice_index[i]]\
                       * (leg_mask[slice_index[i]] == j+1).astype(np.uint8)
            spacing_2d = (voxel_spacing[1], voxel_spacing[0])
            print(f"processing {leg} {region} slice {slice_index[i]} with spacing {spacing_2d}")

            features[region] = compute_slice_features(img_slice,
                                                      mask_slice,
                                                      spacing_2d,
                                                      radiomics_extractor)  
        dst_features.append(features)
     
     folder_name = os.path.basename(os.path.normpath(output_file_name))
     file_path = os.path.join(output_file_name, f"{folder_name}_leg_features_withoutR_l10.json")
     
     with open(file_path, "w") as fp:
        json.dump(dst_features, fp, indent=2)
     print(f"leg features saved to {file_path}")
