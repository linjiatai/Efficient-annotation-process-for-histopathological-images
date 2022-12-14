import os
from PIL import Image

dataroot = 'ROIs/'

files = os.listdir(dataroot)
tag = ['v0','v1','v2','v3']
if not os.path.exists('stage2_multiple_concat/'):
    os.mkdir('stage2_multiple_concat/')
for file in sorted(files):
    img = Image.new('RGB', (2000*3,2000*len(tag)))
    for i in range(len(tag)):
        tmp = Image.open('stage2_result_'+tag[i]+'/concat/'+file)
        img.paste(tmp,(0,2000*i))
    img.save('stage2_multiple_concat/'+file)
