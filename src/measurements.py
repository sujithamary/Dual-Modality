import numpy as np
import scipy.ndimage as ndi

def binary_mask_from_prob(prob, thr=0.5):
    return (prob >= thr).astype(np.uint8)

def compute_area_mm2(mask, affine, pixel_spacing=None):
    
    if pixel_spacing is None:
        
        sx = np.linalg.norm(affine[0:3,0])
        sy = np.linalg.norm(affine[0:3,1])
    else:
        sx, sy = pixel_spacing
    return mask.sum() * (sx * sy)

def compute_volume_cm3(seg_slices_mask, affine):
    # seg_slices_mask: binary 3D volume (Z, H, W) or list -> returns cm^3
    # voxel volume from affine
    voxel_volume_mm3 = abs(np.linalg.det(affine))
    total_voxels = int(seg_slices_mask.sum())
    return (total_voxels * voxel_volume_mm3) / 1000.0  # mm3 -> cm3

def centroid_voxel(mask):
    # mask: binary 2D or 3D
    if mask.ndim == 2:
        coords = ndi.center_of_mass(mask)
        return tuple(coords)
    else:
        return tuple(ndi.center_of_mass(mask))

def bounding_box(mask):
    # mask: binary 2D array
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin, rmax, cmin, cmax)

def max_diameter(mask):
    # approximate max diameter using pairwise distances on boundary points
    from scipy.spatial.distance import pdist
    pts = np.column_stack(np.nonzero(mask))
    if len(pts) < 2:
        return 0.0
    d = pdist(pts)
    return d.max()
