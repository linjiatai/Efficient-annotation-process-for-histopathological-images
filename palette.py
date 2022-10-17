## You can modify the palette to suit your dataset.

##  CRC
# palette = [0]*100
# palette[0:3] = [255,255,255]    # 白色 背景
# palette[3:6] = [120,120,120]    # 灰色 正常
# palette[6:9] = [255,0,0]        # 红色 肿瘤
# palette[9:12] = [0,255,0]       # 绿色 间质
# palette[12:15] = [0,255,255]    # 青色 粘液
# palette[15:18] = [255,0,255]    # 紫色 坏死
# palette[18:21] = [237,145,33]   # 土黄 肌肉

##  Breast
palette = [0]*768
palette[0:3] = [255,255,255]    ## 纯白    Exclude                  (label=0)
palette[3:6] = [255,0,0]        ## 纯红    invasive tumor           (label=1)
palette[6:9] = [0,255,0]        ## 绿色    tumor-associated stroma  (label=2)
palette[9:12]= [244,164,96]     ## 沙棕色  in-situ tumor            (label=3)
palette[12:15]=[255,255,0]      ## 纯黄    healthy glands           (label=4)
palette[15:18]=[255,0,255]      ## 紫红色  necrosis not in-situ     (label=5)      
palette[18:21]=[0,0,225]        ## 蓝色    inflamed stroma          (label=6)
palette[21:23]=[0,0,0]          ## 黑色    rest                     (label=7)
