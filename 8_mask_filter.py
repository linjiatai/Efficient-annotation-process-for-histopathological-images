import os
from PIL import Image
from skimage import morphology
import cv2
import numpy as np
from palette import palette

Image.MAX_IMAGE_PIXELS = None
def mask_filter(mask):
    mask_np = np.asarray(mask)
    mask_np_cp = mask_np.copy()
    mask_np = np.array(mask_np).astype(np.uint8)

    ## del small noise in region
    for i in range(mask_np.max()):
        dst = morphology.remove_small_objects(mask_np!=i+1,min_size=224*224*2,connectivity=1)
        mask_np[dst==False]=i+1
    
    ## del small tumor region
    dst = morphology.remove_small_objects(mask_np==2,min_size=224*224*0.5,connectivity=1)
    mask_np[dst!=(mask_np==2)]=1

    ## Do not del BG, stroma and nomal region
    mask_np[mask_np_cp==3]=3
    mask_np[mask_np_cp==1]=1
    mask_np[mask_np_cp==0]=0
    
    ## del small tissue region of any type.
    dst = morphology.remove_small_objects(mask_np!=0,min_size=224*224*10,connectivity=1)
    mask_np[dst==False]=0
    
    return mask_np

maskpath = 'WSI/mask/'
maskpath_new = 'WSI/seg_mask_filter/'
if not os.path.exists(maskpath_new):
    os.mkdir(maskpath_new)

for root, _, files in os.walk(maskpath):
    for file in sorted(files):
        print(file)
        mask = Image.open(root+'/'+file)
        mask_np = mask_filter(mask)
        mask_new = Image.fromarray(np.uint8(mask_np), 'P')
        mask_new.putpalette(palette)
        mask_new.save(maskpath_new+file)
