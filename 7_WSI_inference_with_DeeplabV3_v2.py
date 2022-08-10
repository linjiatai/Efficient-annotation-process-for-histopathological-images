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
from torch.utils.data import DataLoader, Dataset
class img_Dataset(Dataset):
    def __init__(self, WSI, overlap, bg, transform_val):
        self.slide = WSI
        self.bg = bg
        self.overlap = overlap
        self.xs, self.ys = self.__coordinate_generation__(WSI)
        self.transform_val = transform_val
        
    def transform_BLS(self, sample,size):
        composed_transforms = transforms.Compose([transforms.Resize((size,size)),Normalize(),ToTensor()])
        return composed_transforms(sample)

    def __getitem__(self, index):
        x = self.xs[index]
        y = self.ys[index]
        patch_cv2 = self.slide.crop((x,y,x+224,y+224))
        patch = self.transform_val(patch_cv2) # tensor
        return patch, x, y 
    def __coordinate_generation__(self,WSI):
        xs = []
        ys = []
        H = WSI.size[0]
        W = WSI.size[1]
        for x in range(0, H, 224-self.overlap):
            if x+224 > H:
                x = H-224
            for y in range(0, W, 224-self.overlap):
                if y+224 > W:
                    y = W-224
                bg_tmp = np.array(self.bg.crop([x,y,x+224,y+224]))
                if np.sum(bg_tmp) > 224*224*0.8:
                    continue
                xs.append(x)
                ys.append(y)
        return xs,ys

    def __len__(self):
        return len(self.xs)

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
        checkpoint = torch.load('/media/linjiatai/linjiatai-16TB/兵兵/Multi-step/2_stage_DeeplabV3/checkpoints/checkpoint_stage2_v4_ep_29.pth')
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
        bg_mask = np.zeros(orig_img.shape[:2])
        bg_mask[dst==True]=1
        bg_mask = Image.fromarray(np.uint8(bg_mask), 'P')
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
    def gain_network_output_use_dataloader(self,WSI):
        H = WSI.size[1]
        W = WSI.size[0]
        G = np.zeros((6, H, W))
        D = np.zeros((6, H, W))
        bg = self.gen_bg_mask(WSI)
        dataset = img_Dataset(WSI,self.args.overlap, bg, self.transform_val)
        data_loader = DataLoader(dataset,batch_size=60,shuffle=False,num_workers=8,pin_memory=False,drop_last=False)
        for iter, (patch, xs, ys) in (enumerate(data_loader)):
            with torch.no_grad():
                patch = patch.cuda()
                output = self.model(patch)
            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                pred = output[i]
                G[:, y:y+224, x:x+224] += pred.detach().cpu().numpy()
                D[:, y:y+224, x:x+224] += 1
        D[D==0] = 1
        G /=D
        G = torch.from_numpy(G)
        bg = np.array(bg)*100
        bg = torch.from_numpy(bg)
        bg = bg.unsqueeze(0)
        G = torch.cat((bg,G), 0).numpy()
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
    
    def seg_WSI(self, WSI_dir):
        slide = openslide.open_slide(WSI_dir)
        H_40x,W_40x = slide.dimensions
        H_20x,W_20x = int(H_40x/2),int(W_40x/2)
        step_x_20x = int(H_20x/5)+1
        step_y_20x = int(W_20x/5)+1
        mask = Image.new('P', (int(H_20x), int(W_20x)))
        # img_20x = Image.new('RGB', (int(H_40x/2), int(W_40x/2)))
        for x_20x in range(0,H_20x,step_x_20x):
            if x_20x+step_x_20x>H_20x:
                x_20x = H_20x - step_x_20x
            for y_20x in range(0,W_20x,step_y_20x):
                if y_20x+step_y_20x>W_20x:
                    y_20x = W_20x - step_y_20x
                img = slide.read_region((x_20x*2,y_20x*2), 0, (step_x_20x*2, step_y_20x*2)).convert('RGB')
                img = img.resize((int(step_x_20x), int(step_y_20x)))
                pred = self.gain_network_output_use_dataloader(img)
                pred = np.argmax(pred,0)
                visualimg = Image.fromarray(pred.astype(np.uint8), "P")
                # x_20x, y_20x = int(x_40x/2), int(y_40x/2)
                # step_x_20x, step_y_20x = int(step_x_40x/2), int(step_y_40x/2)
                mask.paste(visualimg,(x_20x,y_20x, x_20x+step_x_20x, y_20x+step_y_20x))
                # img_20x.paste(img,(x_20x,y_20x, x_20x+step_x_20x, y_20x+step_y_20x))
        mask.putpalette(self.palette)
        img_20x = None
        return mask, img_20x

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
    parser.add_argument('--overlap', type=int, default=120, help='overlap')
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
    dataroot = '/media/linjiatai/linjiatai-16TB/兵兵/Multi-step/SAVE/Original_dataset/WSIs/'
    for root,_,files in os.walk(dataroot):
        files = sorted(files)
        for file in files:
            print(file)
            if not (file.split('.')[-1] == 'svs'):
                continue
            if os.path.exists(os.path.join('/media/linjiatai/linjiatai-16TB/兵兵/Multi-step/SAVE/Original_dataset/seg_mask/', file[:-4]+'.png')):
                continue
            WSI_dir = os.path.join(root,file)
            mask, img_20x = WSI_seger.seg_WSI(WSI_dir)
            end_time = time.time()
            run_time = end_time-begin_time
            print ('time consumption:',run_time)
            mask.save(os.path.join('/media/linjiatai/linjiatai-16TB/兵兵/Multi-step/SAVE/Original_dataset/seg_mask/', file[:-4]+'.png'))
            # img_20x.save(os.path.join('/media/linjiatai/linjiatai-16TB/兵兵/Multi-step/SAVE/Original_dataset/seg_img/', file[:-4]+'.jpg'))
if __name__ == "__main__":
   main()
