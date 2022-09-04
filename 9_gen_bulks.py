import os
import shutil
from tool.concave_hull import concave_hull

mask_dir = 'WSI/mask/'
bulk_dir = 'WSI/bulks/'
nobulk_dir = 'WSI/no_bulks/'
if not os.path.exists(nobulk_dir):
    os.mkdir(nobulk_dir)
if not os.path.exists(bulk_dir):
    os.mkdir(bulk_dir)
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
            shutil.move(os.path.join(mask_dir, file),os.path.join(nobulk_dir, file) )
            continue
        bulk.putpalette(palette)
        bulk.save(os.path.join(bulk_dir, file))
