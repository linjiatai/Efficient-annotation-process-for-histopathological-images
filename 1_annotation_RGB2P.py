import os
from turtle import ontimer
import numpy as np
from PIL import Image
import argparse
from palette import palette
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v1')
args = parser.parse_args()

labeldir = 'label_'+str(args.version)+'_RGB/'
label_RGB_dir = 'label_'+str(args.version)+'_P/'
if not os.path.exists(label_RGB_dir):
    os.mkdir(label_RGB_dir)
annotations = os.listdir(labeldir)
for annotation in sorted(annotations):
    if annotation[-3:] != 'png':
        continue
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
