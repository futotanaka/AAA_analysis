import numpy as np
from scipy import ndimage
import math
from skimage.morphology import skeletonize

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


def LRCoutputAnalysis(radiusList, spacing, part_name):
    print("length:", len(radiusList))
    if len(radiusList) == 0:
        print(f"LRC-{part_name} Median: 0")
        print(f"LRC-{part_name} Average different: 0")
        return
    diffAverageBefore = []
    s = 0.0
    for i in range(len(radiusList) - 1):
        d = abs(radiusList[i] - radiusList[i+1])
        if d < 50.0:
            diffAverageBefore.append(d)
        if radiusList[i] < 100.0:
            s += radiusList[i]
        else:
            s += 70
    diffAverage = sum(diffAverageBefore) / len(diffAverageBefore) if diffAverageBefore else 0
    sortedList = sorted(radiusList)
    median = sortedList[len(sortedList)//2]
    average = s / len(radiusList)
    upper20 = sortedList[(len(sortedList)*4)//5]
    lower20 = sortedList[len(sortedList)//5]
    print(f"LRC-{part_name} Median: {median:.2f}")
    # print(f"80% percentiles:{upper20:.2f}")
    # print(f"20% percentiles:{lower20:.2f}")
    # print(f"difference of above two values:{upper20 - lower20:.2f}")
    print(f"LRC-{part_name} Average different: {diffAverage:.2f}")
    # 计算 bounding box 的物理尺寸
    # boxX = (boundingBox[3] - boundingBox[0]) * spacing[0]
    # boxY = (boundingBox[4] - boundingBox[1]) * spacing[1]
    # boxZ = (boundingBox[5] - boundingBox[2]) * spacing[2]

def DSBoutputAnalysis(distanceList, spacing, part_name):
    diffAverageBefore = []
    if len(distanceList) == 0:
        print(f"DSB-{part_name} Median: 0")
        print(f"DSB-{part_name} Average different: 0")
        return
    for i in range(1, len(distanceList)):
        d = abs(distanceList[i] - distanceList[i-1])
        if d < 50.0:
            diffAverageBefore.append(d)
    diffAverage = sum(diffAverageBefore) / len(diffAverageBefore) if diffAverageBefore else 0
    sortedList = sorted(distanceList)
    median = sortedList[len(sortedList)//2]
    upper20 = sortedList[(len(sortedList)*4)//5]
    lower20 = sortedList[len(sortedList)//5]
    print(f"DSB-{part_name} Median: {median:.2f}")
    # print(f"80% percentiles:{upper20:.2f}")
    # print(f"20% percentiles:{lower20:.2f}")
    # print(f"difference of above two values:{upper20 - lower20:.2f}")
    print(f"DSB-{part_name} Average different: {diffAverage:.2f}")

def AreaAnalysis(crossSectionAreas, zAreas):
    print("-------cross-section area")
    crossSectionAverage = sum(crossSectionAreas) / len(crossSectionAreas) if crossSectionAreas else 0
    sortedCS = sorted(crossSectionAreas)
    median = sortedCS[len(sortedCS)//2] if sortedCS else 0
    maxArea = sortedCS[-1] if sortedCS else 0
    vcsRatioMedian = median / maxArea if maxArea != 0 else 0
    print("Median:", median)
    print("Maximum:", maxArea)
    print("VCS ratio median:", vcsRatioMedian)
    print("-------axial area")
    zAreasAverage = sum(zAreas) / len(zAreas) if zAreas else 0
    sortedZ = sorted(zAreas)
    median_z = sortedZ[len(sortedZ)//2] if sortedZ else 0
    maxArea_z = sortedZ[-1] if sortedZ else 0
    vcsRatioMedian_z = median_z / maxArea_z if maxArea_z != 0 else 0
    print("Median:", median_z)
    print("Maximum:", maxArea_z)
    print("VCS ratio median:", vcsRatioMedian_z)

def centerCircle3d(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    a1 = (y1*z2 - y2*z1 - y1*z3 + y3*z1 + y2*z3 - y3*z2)
    b1 = -(x1*z2 - x2*z1 - x1*z3 + x3*z1 + x2*z3 - x3*z2)
    c1 = (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2)
    d1 = -(x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1)
    
    a2 = 2 * (x2 - x1)
    b2 = 2 * (y2 - y1)
    c2 = 2 * (z2 - z1)
    d2 = x1**2 + y1**2 + z1**2 - x2**2 - y2**2 - z2**2
    
    a3 = 2 * (x3 - x1)
    b3 = 2 * (y3 - y1)
    c3 = 2 * (z3 - z1)
    d3 = x1**2 + y1**2 + z1**2 - x3**2 - y3**2 - z3**2
    
    denom = (a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1 + 1e-4)
    x = -(b1 * c2 * d3 - b1 * c3 * d2 - b2 * c1 * d3 + b2 * c3 * d1 + b3 * c1 * d2 - b3 * c2 * d1) / denom
    y = (a1 * c2 * d3 - a1 * c3 * d2 - a2 * c1 * d3 + a2 * c3 * d1 + a3 * c1 * d2 - a3 * c2 * d1) / denom
    z = -(a1 * b2 * d3 - a1 * b3 * d2 - a2 * b1 * d3 + a2 * b3 * d1 + a3 * b1 * d2 - a3 * b2 * d1) / denom
    radius = math.sqrt((x1 - x)**2 + (y1 - y)**2 + (z1 - z)**2)
    return x, y, z, radius

def GetFoot(pt, startp, endp):
    ret = [0, 0, 0]
    dx = startp[0] - endp[0]
    dy = startp[1] - endp[1]
    dz = startp[2] - endp[2]
    if abs(dx) < 1e-8 and abs(dy) < 1e-8 and abs(dz) < 1e-8:
        return startp
    u = ((pt[0]-startp[0])*(startp[0]-endp[0]) +
         (pt[1]-startp[1])*(startp[1]-endp[1]) +
         (pt[2]-startp[2])*(startp[2]-endp[2]))
    u = u / (dx*dx + dy*dy + dz*dz)
    ret[0] = startp[0] + u * dx
    ret[1] = startp[1] + u * dy
    ret[2] = startp[2] + u * dz
    return ret

def argsortInverse(array):
    return sorted(range(len(array)), key=lambda i: array[i], reverse=True)

def calculatePlane(a, b, c):
    # Ax + By + Cz + D = 0
    x1 = b[0] - a[0]
    x2 = b[1] - a[1]
    x3 = b[2] - a[2]
    x4 = x1 * c[0] + x2 * c[1] + x3 * c[2]
    return [x1, x2, x3, -x4]

def boundingBoxCal(stent_mask, spacing, cal_range):
    sub_mask = stent_mask[cal_range[0]:cal_range[1]+1]
    
    nz_z, nz_y, nz_x = np.nonzero(sub_mask)
    
    if nz_z.size == 0:
        print("Did't find non-zero element in bounding box calulation range.")
        return None
    
    # boundingBox = [min_x, min_y, min_z, max_x, max_y, max_z]
    min_z = np.min(nz_z)
    max_z = np.max(nz_z)
    min_x = np.min(nz_x)
    max_x = np.max(nz_x)
    min_y = np.min(nz_y)
    max_y = np.max(nz_y)
    
    boundingBox = [min_x, min_y, min_z, max_x, max_y, max_z]
    # print("Bounding box of main branch -> start: ({},{},{}) end: ({},{},{})".format(
    #     min_x, min_y, min_z, max_x, max_y, max_z))
    
    boxX = (max_x - min_x) * spacing[0]
    boxY = (max_y - min_y) * spacing[1]
    boxZ = (max_z - min_z) * spacing[2]
    
    print("difference of x(mm):", boxX)
    print("difference of y(mm):", boxY)
    print("difference of z(mm):", boxZ)
    print(f"Bounding box volume(mm3):{boxX * boxY * boxZ:.2f}")
    
    return boundingBox
    
def get_first_coord(slice_2d, t):
    """
        Find the first position (column-first order) in the 2D array slice_2d
        where the value equals t. To simulate the original code’s iteration 
        with outer loop j (y) and inner loop k (x), we first transpose the array 
        so that the first dimension is y and the second dimension is x.
        Return the (x, y) coordinate; if not found, return None.
    """
    # After transpose, shape is (y, x)
    coords = np.argwhere(slice_2d.T == t)
    if coords.size == 0:
        return None
    # Here argwhere returns in row-major order (i.e., y first, then x)
    y, x = coords[0]
    return float(x), float(y)

def postProcessingForStent(result, stent_mask, spacing, arterial_branchpoint):
    """
    Parameter description:
      result: 3D NumPy array (dtype short), with shape (z, x, y), storing labels of each branch (1, 2, 3, ...).
      stent_mask: 3D NumPy array (dtype unsigned char), shape (z, x, y).
      interpolationMasks: A list of 3D NumPy arrays (each with shape (z, x, y)), each corresponding to one branch.
      spacing: [spacing_x, spacing_y, spacing_z]
    Returns: branchPoint (the z index where the branch point is located).
    """
    # Compute the starting and ending slices (along the z-axis)
    startSlice = 0
    endSlice = 0
    flag = False
    print(result.shape)
    nonzero_indices = np.nonzero(result)  # Returns a tuple; the first element corresponds to the z-axis
    if nonzero_indices[0].size > 0:
        startSlice = np.min(nonzero_indices[0])
        endSlice = np.max(nonzero_indices[0])
    else:
        startSlice = endSlice = None  # Or any default value you prefer
    
    # Detect the branch point: iterate from startSlice to endSlice; for the first slice where result > 1 is satisfied,
    # take the previous slice as branchPoint
    sub_array = result[startSlice:endSlice+1]  # shape: (n, result.shape[1], result.shape[2])
    condition = np.any(sub_array > 1, axis=(1, 2))
    if np.any(condition):
        # np.argmax returns the index of the first True
        first_index = np.argmax(condition)
        branchPoint = startSlice + first_index - 1
    else:
        branchPoint = 0
    
    print("branch point of stent: ", branchPoint)
    
    # # Compute the bounding box for the main branch and the whole body, and draw the bounding box
    # print("-----------Bounding box information above arterial bifucation-----------")
    # boundingBoxM = boundingBoxCal(stent_mask, spacing, [0,arterial_branchpoint])
    # print("-----------Bounding box information whole body-----------")
    # boundingBox = boundingBoxCal(stent_mask, spacing, [0,endSlice])
    # # drawBoundingBox(stent_mask, boundingBox)
    
    # Local curvature (local radius) calculation
    radiusListBefore = []
    radiusListAfter1 = []
    radiusListAfter2 = []
    
    # Compute the number of slices apart on the z-axis for a physical distance of 5 mm
    sliceThick = int(5.0 / spacing[2])
    
    # Iterate t = 1, 2, 3
    for t in range(1, 4):
        i_val = startSlice + sliceThick
        # Ensure indexFront does not exceed endSlice
        while i_val + sliceThick <= endSlice:
            indexBehind = int(i_val - sliceThick)
            indexMid = int(i_val)
            indexFront = int(i_val + sliceThick)
            
            # Directly extract the corresponding slices (note: result has shape (z, x, y))
            slice_behind = result[indexBehind, :, :]
            slice_mid = result[indexMid, :, :]
            slice_front = result[indexFront, :, :]
            
            # Use the helper function to find the first pixel that meets the condition
            coord_behind = get_first_coord(slice_behind, t)
            coord_mid = get_first_coord(slice_mid, t)
            coord_front = get_first_coord(slice_front, t)
            
            # If any of the slices does not contain a valid pixel, skip this position
            if coord_behind is None or coord_mid is None or coord_front is None:
                i_val += sliceThick
                continue
            
            # Assemble the pixel coordinates (x, y, z) of the three points
            # Note: the z coordinate uses the slice index
            point1 = [coord_behind[0], coord_behind[1], float(indexBehind)]
            point2 = [coord_mid[0],   coord_mid[1],   float(indexMid)]
            point3 = [coord_front[0], coord_front[1], float(indexFront)]
            
            # Convert pixel coordinates to physical coordinates
            # Use spacing[0] and spacing[1] for x and y respectively, and spacing[2] for z
            for idx in range(3):
                if idx < 2:
                    point1[idx] *= spacing[idx]
                    point2[idx] *= spacing[idx]
                    point3[idx] *= spacing[idx]
                else:
                    point1[idx] *= spacing[2]
                    point2[idx] *= spacing[2]
                    point3[idx] *= spacing[2]
            
            # Compute the center and radius of the circle defined by three points
            cx, cy, cz, r = centerCircle3d(point1[0], point1[1], point1[2],
                                           point2[0], point2[1], point2[2],
                                           point3[0], point3[1], point3[2])
            # Save to the corresponding list based on t
            if t == 1:
                radiusListBefore.append(r)
            elif t == 2:
                radiusListAfter1.append(r)
            elif t == 3:
                radiusListAfter2.append(r)
            
            i_val += sliceThick
        
    # Compute deviation from a straight line (curvature—deviation from a straight line)
    endpoints = {1: {"first": None, "last": None},
                 2: {"first": None, "last": None},
                 3: {"first": None, "last": None}}
    
    z_indices = np.arange(startSlice, endSlice + 1, sliceThick)
    for z in z_indices:
        index = int(z)
        slice_2d = result[index, :, :]  # shape: (x, y)
        # For each branch t = 1, 2, 3 in this slice (transpose to iterate y first, then x)
        for t in [1, 2, 3]:
            coords = np.argwhere(slice_2d.T == t)  # Returns an array; each row is [y, x]
            if coords.size > 0:
                # Record the endpoint at the first occurrence (only if the endpoint is not set)
                if endpoints[t]["first"] is None:
                    first = coords[0]
                    endpoints[t]["first"] = [float(first[1]), float(first[0]), float(index)]
                # Update the last occurrence endpoint each time
                last = coords[-1]
                endpoints[t]["last"] = [float(last[1]), float(last[0]), float(index)]
    
    # Print endpoint info
    # print("point1 (branch 1 first):", endpoints[1]["first"])
    # print("point2 (branch 1 last):", endpoints[1]["last"])
    # print("point3 (branch 2 first):", endpoints[2]["first"])
    # print("point4 (branch 2 last):", endpoints[2]["last"])
    # print("point5 (branch 3 first):", endpoints[3]["first"])
    # print("point6 (branch 3 last):", endpoints[3]["last"])
    # print()
    
    # ----------------------------
    # 2. Compute the distance from the straight line based on endpoints
    distanceListBefore = []
    distanceListAfter1 = []
    distanceListAfter2 = []
    
    # For each branch t = 1, 2, 3
    for t in [1, 2, 3]:
        if endpoints[t]["first"] is None or endpoints[t]["last"] is None:
            continue  # Skip if this branch is not found within the scanning range
        
        # Save the original slice indices to determine the sampling range along the z-axis
        z_start = endpoints[t]["first"][2]
        z_end   = endpoints[t]["last"][2]
        
        # Copy endpoints (note: endpoint coordinates are [x, y, z], but here z is the slice index)
        p1 = endpoints[t]["first"].copy()
        p2 = endpoints[t]["last"].copy()
        
        # Compute the physical coordinates of the two endpoints of the line
        # (multiply x, y by spacing[0], spacing[1]; multiply z by spacing[2])
        p1_phys = p1.copy()
        p2_phys = p2.copy()
        for idx in range(3):
            if idx < 2:
                p1_phys[idx] *= spacing[idx]
                p2_phys[idx] *= spacing[idx]
            else:
                p1_phys[idx] *= spacing[2]
                p2_phys[idx] *= spacing[2]
        
        # Within the z-range of the line, sample every sliceThick slices
        for z in np.arange(z_start + sliceThick, z_end, sliceThick):
            index = int(z)  # index is the slice number
            slice_2d = result[index, :, :]
            # Use the helper to get the first pixel coordinate ([x, y]) in this slice that meets the condition
            coord = get_first_coord(slice_2d, t)
            if coord is None:
                continue
            # Assemble the pixel coordinate of this point; use the current slice index for z
            pt = [coord[0], coord[1], float(index)]
            # Convert to physical coordinates
            pt_phys = pt.copy()
            pt_phys[0] *= spacing[0]
            pt_phys[1] *= spacing[1]
            pt_phys[2] *= spacing[2]
            
            # Compute the foot of the perpendicular from pt_phys to the line (p1_phys, p2_phys); GetFoot is implemented elsewhere
            foot = GetFoot(pt_phys, p1_phys, p2_phys)
            # Compute Euclidean distance
            distance = math.sqrt((pt_phys[0] - foot[0])**2 +
                                 (pt_phys[1] - foot[1])**2 +
                                 (pt_phys[2] - foot[2])**2)
            # If the distance is extremely small, set it to 0
            if distance < 0.01:
                distance = 0
            # Save to the corresponding distance list for this branch
            if t == 1:
                distanceListBefore.append(distance)
            elif t == 2:
                distanceListAfter1.append(distance)
            elif t == 3:
                distanceListAfter2.append(distance)
    

    # Compute cross-sectional area
    # crossSectionAreasList = []
    # zAreasList = []
    # for t in range(1, 4):
    #     crossSectionAreas = []
    #     zAreas = []
    #     i_val = startSlice + sliceThick
    #     while i_val + sliceThick <= endSlice:
    #         indexBehind = int(i_val - sliceThick)
    #         indexMid = int(i_val)
    #         indexFront = int(i_val + sliceThick)
    #         point1_cs = [0.0, 0.0, 0.0]
    #         point2_cs = [0.0, 0.0, 0.0]
    #         point3_cs = [0.0, 0.0, 0.0]
    #         found1 = found2 = found3 = False
    #         for j in range(result.shape[2]):
    #             for k in range(result.shape[1]):
    #                 if not found1 and result[indexBehind, k, j] == t:
    #                     point1_cs = [float(k), float(j), float(indexBehind)]
    #                     found1 = True
    #                 if not found2 and result[indexMid, k, j] == t:
    #                     point2_cs = [float(k), float(j), float(indexMid)]
    #                     found2 = True
    #                 if not found3 and result[indexFront, k, j] == t:
    #                     point3_cs = [float(k), float(j), float(indexFront)]
    #                     found3 = True
    #         if not (found1 and found2 and found3):
    #             i_val += sliceThick
    #             continue
    #         plane = calculatePlane(point1_cs, point3_cs, point2_cs)
    #         area = 0
    #         zArea = 0
    #         # Note: in the inner loop here, j corresponds to y (result.shape[2]), k corresponds to x (result.shape[1])
    #         for z in range(int(i_val - 50), int(i_val + 50)):
    #             for j in range(result.shape[2]):
    #                 for k in range(result.shape[1]):
    #                     if z < 0 or z >= interpolationMasks[t-1].shape[0]:
    #                         continue
    #                     # When accessing interpolationMasks, convert indices to [z, k, j]
    #                     if interpolationMasks[t-1][z, k, j] == 0:
    #                         continue
    #                     distanceTop = abs(plane[0]*k + plane[1]*j + plane[2]*z + plane[3])
    #                     distanceDown = math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
    #                     if distanceTop <= 50 and (distanceTop / distanceDown) <= 0.5:
    #                         area += 1
    #                     if z == int(i_val):
    #                         if result[z, k, j] == 0:
    #                             result[z, k, j] = 6
    #                         zArea += 1
    #         if area != 0 and zArea != 0:
    #             area = area * spacing[0] * spacing[1]
    #             zArea = zArea * spacing[0] * spacing[1]
    #             crossSectionAreas.append(area)
    #             zAreas.append(zArea)
    #         i_val += sliceThick
    #     crossSectionAreasList.append(crossSectionAreas)
    #     zAreasList.append(zAreas)
    
    print("-----------local radius of curvature(main, left leg, right leg)-----------")
    LRCoutputAnalysis(radiusListBefore, spacing, "main")
    LRCoutputAnalysis(radiusListAfter1, spacing, "left")
    LRCoutputAnalysis(radiusListAfter2, spacing, "right")
    print("-----------distance from straight branch-----------")
    DSBoutputAnalysis(distanceListBefore, spacing,"main")
    DSBoutputAnalysis(distanceListAfter1, spacing,"left")
    DSBoutputAnalysis(distanceListAfter2, spacing,"right")
    # print("--------------area analysis--------------")
    # if len(crossSectionAreasList) >= 3:
    #     AreaAnalysis(crossSectionAreasList[0], zAreasList[0])
    #     AreaAnalysis(crossSectionAreasList[1], zAreasList[1])
    #     AreaAnalysis(crossSectionAreasList[2], zAreasList[2])
    return branchPoint


def AAA_part_stent_analysis(stent_mask, centerline, aaa_range, spacing, array_stent_ori, spacing_ori, zoom_factor):
    # structure = _circle_structure(5)
    # structure_3d = _spherical_structure(3)
    # structure_element = np.zeros((3, 1, 1))
    # structure_element[:, 0, 0] = 1
    # stent_mask = ndimage.binary_opening(stent_mask, structure=structure_3d)
    # # sub_array_aorta = ndimage.binary_opening(sub_array_aorta, structure=structure_element)
    # stent_mask_eroded = np.zeros_like(stent_mask, dtype=stent_mask.dtype)
    # for z in range(stent_mask.shape[0]):
    #     stent_mask_eroded[z] = ndimage.binary_erosion(stent_mask[z], structure=structure)
    # stent_mask_eroded = ndimage.binary_closing(stent_mask_eroded, structure=structure_3d)
    # skel = skeletonize(stent_mask_eroded) 
    # skel = np.where(skel, 1, 0).astype(np.uint8)
    
    print("Bounding box of stent in AAA:  ")
    boundingBoxCal(stent_mask, spacing, aaa_range) # if AAA range is calulated from non-interpolation data, using ori version input.
    
    z0, z1 = aaa_range
    # if AAA range is calulated from non-interpolation data, using the next 2 lines
    # z0 = round(z0*zoom_factor) 
    # z1 = round(z1*zoom_factor)
    print("bounding box AAA: ",z0,z1)
    roi = centerline[z0:z1+1]
    D, H, W = roi.shape

    visited = set()

    def length_of_branch(binary_skel):
        def dfs(z, x, y):
            visited.add((z, x, y))
            length = 0.0
            for dz in (0, 1):  # Only scan the same layer (dz=0) and the previous layer (dz=1)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dz == dx == dy == 0:
                            continue
                        nz, nx, ny = z+dz, x+dx, y+dy
                        if (0 <= nz < D and 0 <= nx < H and 0 <= ny < W
                            and binary_skel[nz, nx, ny]
                            and (nz, nx, ny) not in visited):
                            # Accumulate Euclidean distance
                            dist = math.sqrt((dz*spacing[2])**2 + (dx*spacing[0])**2 + (dy*spacing[1])**2)
                            length += dist
                            length += dfs(nz, nx, ny)
            return length

        total = 0.0
        # Start with the pixels at the top layer where z == 0
        for x in range(H):
            for y in range(W):
                if binary_skel[0, x, y] and (0, x, y) not in visited:
                    total += dfs(0, x, y)
        # Then traverse the remaining unvisited skeleton points (isolated branches)
        for z, x, y in np.argwhere(binary_skel):
            if (z, x, y) not in visited:
                total += dfs(z, x, y)
        return total

    total_length = 0.0
    # Count separately for each label, then sum them up
    for label in np.unique(roi):
        if label == 0:
            continue
        binary_branch = (roi == label)
        total_length += length_of_branch(binary_branch)

    visited_mask = centerline.copy()
    for z_local, x, y in visited:
        visited_mask[z0 + z_local, x, y] = 20
    
    print("stent length in AAA(mm): ",total_length)
    return total_length, visited_mask