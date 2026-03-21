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
    
    print(kp1)


    desc1 = desc1.squeeze(0)
    desc2 = desc2.squeeze(0)

    _, idxs = K.feature.match_smnn(desc1, desc2, th=0.95)

    pts1 = kp1.squeeze(0)[idxs[:, 0]]
    pts2 = kp2.squeeze(0)[idxs[:, 1]]

    print('Points 1:', pts1)

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

    return img, overlap
