import os
import shutil
from tool.concave_hull import concave_hull

mask_dir = '/home/linjiatai/14TB/兵兵/Multi-step/SAVE/Original_dataset/seg_mask/'
bulk_dir = '/home/linjiatai/14TB/兵兵/Multi-step/SAVE/Original_dataset/bulks/'
bad_dir = '/home/linjiatai/14TB/兵兵/Multi-step/SAVE/Original_dataset/bad_case/'
for root, _, files in os.walk(mask_dir):
    for file in sorted(files):
        print(file)
        if os.path.isfile(os.path.join(bulk_dir, file)):
            continue
        bulk = concave_hull(root+'/'+file, spacing=0.5, alpha=0.1, bulk_class=2)
        palette = [0]*6
        palette[0:3] = [0,0,0]
        palette[3:6] = [255,255,255]
        if bulk == None:
            shutil.move(os.path.join(mask_dir, file),os.path.join(bad_dir, file) )
            continue
        bulk.putpalette(palette)
        bulk.save(os.path.join(bulk_dir, file))