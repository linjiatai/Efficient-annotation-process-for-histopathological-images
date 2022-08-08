import os
from PIL import Image

dataroot = '../SAVE/representative_region_20x/'

files = os.listdir(dataroot)
tag = 'v4'

for file in sorted(files):
    img = Image.open(dataroot+file).convert('RGB')
    seg = Image.open('stage2_result_'+tag+'/seg/'+file).convert('RGB')
    mask = Image.open('stage2_result_'+tag+'/mask/'+file).convert('RGB')
    concat = Image.new('RGB', (2000*3,2000))
    concat.paste(img,(0,0))
    concat.paste(seg,(2000,0))
    concat.paste(mask,(4000,0))
    concat.save('stage2_result_'+tag+'/concat/'+file)
