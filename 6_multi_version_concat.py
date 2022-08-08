import os
from PIL import Image

dataroot = '../SAVE/representative_region_20x/'

files = os.listdir(dataroot)
tag = ['v0','v1','v2','v3']

for file in sorted(files):
    img = Image.new('RGB', (2000*3,2000*len(tag)))
    for i in range(len(tag)):
        tmp = Image.open('stage2_result_'+tag[i]+'/concat/'+file)
        img.paste(tmp,(0,2000*i))
    img.save('stage2_multiple_concat/'+file)