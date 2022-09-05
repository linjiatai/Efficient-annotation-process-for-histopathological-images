import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import *
from tool.loss import SegmentationLosses
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator
from palette import palette
class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define
        self.saver = Saver(args)
        self.summary = TensorboardSummary('logs')
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.ft:
                self.model.module.load_state_dict(checkpoint)
            elif args.cuda:
                W = checkpoint['state_dict']
                if not args.ft:
                    del W['decoder.last_conv.8.weight']
                    del W['decoder.last_conv.8.bias']
                self.model.module.load_state_dict(W, strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' ".format(args.resume))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        loss_record = 100
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            one = torch.ones((output.shape[0],1,224,224)).cuda()
            output = torch.cat([(100 * one * (target==0).unsqueeze(dim = 1)), output],dim = 1)
            loss_o = self.criterion(output, target)
            loss = loss_o
            if loss < loss_record:
                loss_record = loss.data.cpu()
                self.saver.save_checkpoint(state=self.model.module.state_dict(), filename='checkpoint_stage2_'+self.args.version+'.pth')
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        

def main():
    parser = argparse.ArgumentParser(description="WSSS Stage2")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--version', type=str, default='v1')
    
    parser.add_argument('--savepath', type=str, default='checkpoints/')
    parser.add_argument('--workers', type=int, default=10, metavar='N')
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--n_class', type=int, default=6)
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N')
    parser.add_argument('--batch-size', type=int, default=5, metavar='N')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    parser.add_argument('--nesterov', action='store_true', default=False )
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    # checking point
    parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    parser.add_argument('--eval-interval', type=int, default=1)
    args = parser.parse_args()
    args.dataroot = 'dataset_stage2_'+args.version
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if args.version == 'v1':
        args.resume = 'init_weights/deeplab-resnet.pth.tar'
        args.ft = False
    else:
        args.resume = 'checkpoints/checkpoint_stage2_v'+str(int(args.version[1])-1)+'.pth'
        args.ft = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    print(args)
    trainer = Trainer(args)
    for epoch in range(trainer.args.epochs):
        trainer.training(epoch)
    
    trainer.writer.close()

if __name__ == "__main__":
   main()
