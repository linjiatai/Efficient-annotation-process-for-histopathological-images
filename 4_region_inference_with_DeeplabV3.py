import openslide
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from skimage import morphology
import torch
import argparse
import os
from torch import nn
from PIL import Image
import time
from network.deeplab import *
from tool import custom_transforms as tr

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample
        # print(mask.shape)
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

class WSI_seg(object):
    def __init__(self, args):
        self.args = args
        self.nclass = 6
        palette = [0]*100
        palette[0:3] = [255,255,255]    # 白色 背景
        palette[3:6] = [120,120,120]    # 灰色 正常
        palette[6:9] = [255,0,0]        # 红色 肿瘤
        palette[9:12] = [0,255,0]       # 绿色 间质
        palette[12:15] = [0,255,255]    # 青色 粘液
        palette[15:18] = [255,0,255]    # 紫色 坏死
        palette[18:21] = [237,145,33]   # 土黄 肌肉
        self.palette = palette

        model = DeepLab(num_classes=6,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
        checkpoint = torch.load('checkpoints/checkpoint_stage2_v4_ep_29.pth')
        model.load_state_dict(checkpoint)
        # Using cuda
        self.model = model.cuda()
        self.model.eval()
        
    def gen_bg_mask(self,orig_img):
        orig_img = np.asarray(orig_img)
        img_array = np.array(orig_img).astype(np.uint8)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary = np.uint8(binary)    
        dst = morphology.remove_small_objects(binary!=255,min_size=10000,connectivity=1)
        dst = morphology.remove_small_objects(dst==False,min_size=10000,connectivity=1)
        bg_mask = np.ones(orig_img.shape[:2]) * -10000
        bg_mask[dst==True]=10000
        return bg_mask

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([Normalize(),ToTensor()])
        return composed_transforms(sample)

    def transform_BLS(self, sample,size):
        composed_transforms = transforms.Compose([transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04),transforms.Resize((size,size))])
        return composed_transforms(sample)
    
    def read_img(self, img_dir):
        img = cv2.imread(img_dir)
        return img

    def gain_network_output(self,WSI):
        H = WSI.size[1]
        W = WSI.size[0]
        G = np.zeros((6, H, W))
        D = np.zeros((6, H, W))
        for y in range(0, H, 224-self.args.overlap):
            if y+224 > H:
                y = H-224
            for x in range(0, W, 224-self.args.overlap):
                if x+224 > W:
                    x = W-224
                patch_cv2 = WSI.crop((x,y,x+224,y+224))
                patch = self.transform_val(patch_cv2) # tensor
                patch = patch.unsqueeze(0)
                if self.args.cuda:
                    patch = patch.cuda()
                with torch.no_grad():
                    output = self.model(patch)
                G[:, y:y+224, x:x+224] += output.squeeze().detach().cpu().numpy()
                D[:, y:y+224, x:x+224] += 1
        G /=D
        G = torch.from_numpy(G)
        mask = self.gen_bg_mask(WSI)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        G = torch.cat((mask,G), 0).numpy()
        return G

    def fuse_mask_and_img(self, mask, img):
        mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        Combine = cv2.addWeighted(mask,0.3,img,0.7,0)
        return Combine

    def seg_png(self, WSI_dir):
        img = Image.open(WSI_dir).convert('RGB')
        pred = self.gain_network_output(img)
        pred = np.argmax(pred,0)
        visualimg = Image.fromarray(pred.astype(np.uint8), "P")
        visualimg.putpalette(self.palette)
        mask = visualimg
        visualimg = visualimg.convert("RGB")
        mask_on_img = self.fuse_mask_and_img(visualimg, img)
        mask_on_img = Image.fromarray(cv2.cvtColor(mask_on_img, cv2.COLOR_BGR2RGB),'RGB')
        return mask_on_img,mask

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--overlap', type=int, default=130, help='overlap')
    parser.add_argument('--save_dir', type=str, default='stage2_result_v4/', help='save path')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    print(args)
    torch.manual_seed(args.seed)
    WSI_seger = WSI_seg(args)
    begin_time = time.time()
    dataroot = '../SAVE/representative_region_20x/'
    for root,_,files in os.walk(dataroot):
        files = sorted(files)
        for file in files:
            print(file)
            if not (file.split('.')[-1] == 'png'):
                continue
            if os.path.exists(os.path.join('mask', file)):
                continue
            img_dir = os.path.join(root,file)
            mask_on_img, mask = WSI_seger.seg_png(img_dir)
            end_time = time.time()
            run_time = end_time-begin_time
            print ('time consumption:',run_time)
            mask_on_img.save(os.path.join(args.save_dir+'seg/', file))
            mask.save(os.path.join(args.save_dir+'mask/', file))

if __name__ == "__main__":
   main()     




