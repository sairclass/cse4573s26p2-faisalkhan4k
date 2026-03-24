'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    keys = list(imgs.keys())

    # Load & normalize input imgs
    img1 = imgs[keys[0]].float() / 255.0
    img2 = imgs[keys[1]].float() / 255.0

    # make batches
    img1_b = img1.unsqueeze(0)#1x C x H x W
    img2_b = img2.unsqueeze(0)

    #grayscale 
    img1_gray = K.color.rgb_to_grayscale(img1_b)
    img2_gray = K.color.rgb_to_grayscale(img2_b)

    detector = K.feature.KeyNetAffNetHardNet(num_features=2048, upright=True)
    # get the keypoint and the detectors
    with torch.no_grad():
        kp1,_, desc1 = detector(img1_gray)
        kp2,_, desc2 = detector(img2_gray)
    


    desc1 = desc1.squeeze(0)
    desc2 = desc2.squeeze(0)

    _, idxs = K.feature.match_smnn(desc1, desc2, th=0.95)

    pts1 = K.feature.get_laf_center(kp1).squeeze(0)  
    pts2 = K.feature.get_laf_center(kp2).squeeze(0) 

    pts1 = pts1[idxs[:, 0]].unsqueeze(0)  
    pts2 = pts2[idxs[:, 1]].unsqueeze(0)  


    H = K.geometry.find_homography_dlt_iterated(pts1, pts2,n_iter=100,weights=None)

    _, _, H2, W2 = img2_b.shape
    _, _, H1, W1 = img1_b.shape

    corners1 = torch.tensor([[0, 0, 1],[W1, 0, 1],[0, H1, 1],[W1, H1, 1]],dtype=torch.float32).T

    H_mat = H.squeeze(0)
    warped = H_mat @ corners1
    warped = warped[:2] / warped[2]

    corners2 = torch.tensor([[0, 0],[W2, 0],[0, H2],[W2, H2]],dtype=torch.float32)


    # all the points
    all_pts = torch.cat([warped.T, corners2], dim=0)

    min_xy = all_pts.min(dim=0).values
    max_xy = all_pts.max(dim=0).values

    out_w = int((max_xy[0] - min_xy[0]).clamp(max=4000))
    out_h = int((max_xy[1] - min_xy[1]).clamp(max=4000))

    T = torch.eye(3,3)
    T[0, 2] = -min_xy[0]
    T[1, 2] = -min_xy[1]

    H_combined = T @ H_mat

    img1_warped = K.geometry.warp_perspective(img1_b, H_combined.unsqueeze(0), (out_h, out_w))
    img2_warped = K.geometry.warp_perspective(img2_b, T.unsqueeze(0), (out_h, out_w))

    

    ones1 = torch.ones_like(img1_b[:, :1])
    ones2 = torch.ones_like(img2_b[:, :1])


    mask1 = K.geometry.warp_perspective(ones1, H_combined.unsqueeze(0), (out_h, out_w))

    mask2 = K.geometry.warp_perspective(ones2, T.unsqueeze(0), (out_h, out_w))




    mask1 = (mask1 > 0.5).float().squeeze()
    mask2 = (mask2 > 0.5).float().squeeze()
    overlap = mask1 * mask2


    img1_w = img1_warped.squeeze(0)
    img2_w = img2_warped.squeeze(0)

    canvas = torch.zeros_like(img1_w)

    canvas += img1_w * (mask1 * (1 - mask2)).unsqueeze(0)
    canvas += img2_w * (mask2 * (1 - mask1)).unsqueeze(0)


    blur1 = K.filters.gaussian_blur2d(img1_warped, (21, 21), (5.0, 5.0)).squeeze(0)
    blur2 = K.filters.gaussian_blur2d(img2_warped, (21, 21), (5.0, 5.0)).squeeze(0)

    score1 = (img1_w - blur1).abs().mean(dim=0)
    score2 = (img2_w - blur2).abs().mean(dim=0)


    use1 = (score1 <= score2).float()
    use2 = 1 - use1

    blend = (img1_w * use1.unsqueeze(0) + img2_w * use2.unsqueeze(0)) * overlap.unsqueeze(0)

    canvas += blend


    img = (canvas.clamp(0, 1) * 255).byte().cpu()

    return img
    
# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    keys = list(imgs.keys())
    N = len(keys)


    # normalize
    images = [imgs[k].float() / 255.0 for k in keys]


    detector = K.feature.KeyNetAffNetHardNet(num_features=2048, upright=True)
    detector.eval()


    all_pts, all_descs = [], []
    with torch.no_grad():
        for img in images:
            gray = K.color.rgb_to_grayscale(img.unsqueeze(0))
            l, _, desc = detector(gray)

            pts = K.feature.get_laf_center(l).squeeze(0)
            all_pts.append(pts)
            all_descs.append(desc.squeeze(0))

    return img, overlap
