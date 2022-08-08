import os
from matplotlib.pyplot import annotate
import numpy as np
from PIL import Image

annotations_dir = 'label_v4_P/'
regions = '../SAVE/representative_region_20x/'

files = os.listdir(annotations_dir)
step = 120
for file in sorted(files):
    mask = Image.open(annotations_dir+file)
    img = Image.open(regions+file)
    H,W = mask.size
    for x in range(0,H,step):
        if x+224>H:
            x = H-224
        for y in range(0,W,step):
            if y+224>W:
                y = W-224
            subimg = img.crop((x,y,x+224,y+224))
            submask = mask.crop((x,y,x+224,y+224))
            tag = []
            if np.sum(np.array(submask)==0) > 224*224*0.1:
                continue
            if np.sum(np.array(submask)==1) > 224*224*0.3:
                tag.append(1)
            if np.sum(np.array(submask)==2) > 224*224*0.3:
                tag.append(2)
            if np.sum(np.array(submask)==3) > 224*224*0.3:
                tag.append(3)
            if np.sum(np.array(submask)==4) > 224*224*0.3:
                tag.append(4)
            if np.sum(np.array(submask)==5) > 224*224*0.3:
                tag.append(5)
            if np.sum(np.array(submask)==6) > 224*224*0.3:
                tag.append(6)
            label = np.zeros(6)
            for i in range(len(tag)):
                label[tag[i]-1] = 1
            label = np.uint8(label)
            subimg.save('dataset_stage2_v4/img/'+file[:-4]+'_'+str(x)+'_'+str(y)+'_'+str(label)+'.png')
            submask.save('dataset_stage2_v4/mask/'+file[:-4]+'_'+str(x)+'_'+str(y)+'_'+str(label)+'.png')
            