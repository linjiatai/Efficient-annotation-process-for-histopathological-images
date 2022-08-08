import os
import numpy as np
from PIL import Image

labeldir = 'label_v4_RGB/'
label_RGB_dir = 'label_v4_P/'
annotations = os.listdir(labeldir)
palette = [0]*100
palette[0:3] = [255,255,255]
palette[3:6] = [120,120,120]
palette[6:9] = [255,0,0]
palette[9:12] = [0,255,0]
palette[12:15] = [0,255,255]
palette[15:18] = [255,0,255]
palette[18:21] = [237,145,33]



for annotation in sorted(annotations):
    mask = np.array(Image.open(labeldir+annotation).convert('RGB'))
    mask_P = np.zeros((mask.shape[0],mask.shape[1]))

    for i in range(6+1):
        mask_tmp = np.zeros(mask.shape)
        for j in range(3):
            mask_tmp[:,:,j]=palette[i*3+j]
        mask_P += np.min(mask_tmp == mask,2)*i
    mask_P = Image.fromarray(np.uint8(mask_P), 'P')
    mask_P.putpalette(palette)
    mask_P.save(label_RGB_dir+annotation)
