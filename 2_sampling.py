import os
import shutil
import numpy as np
from PIL import Image
from palette import palette
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v1')
args = parser.parse_args()

annotations_dir = 'label_'+str(args.version)+'_P/'
regions = 'ROIs/'

files = os.listdir(annotations_dir)
step = 120
if not os.path.exists('dataset_stage2_v'+str(int(args.version[1])-1)+'/'):
    os.mkdir('dataset_stage2_'+str(args.version)+'/')
    os.mkdir('dataset_stage2_'+str(args.version)+'/img/')
    os.mkdir('dataset_stage2_'+str(args.version)+'/mask/')
else:
    shutil.copytree(src='dataset_stage2_v'+str(int(args.version[1])-1)+'/', dst='dataset_stage2_'+args.version+'/')

for file in sorted(files):
    if file[-3:] != 'png':
        continue
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
            subimg.save('dataset_stage2_'+args.version+'/img/'+file[:-4]+'_'+str(x)+'_'+str(y)+'_'+str(label)+'.png')
            submask.save('dataset_stage2_'+args.version+'/mask/'+file[:-4]+'_'+str(x)+'_'+str(y)+'_'+str(label)+'.png')
            
