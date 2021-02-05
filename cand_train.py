'''
This file is for training candidate models for SWEEN
'''
import argparse
import time
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from training.std import std_train
from training.macer import macer_train
from rs.certify import certify
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture

TRAIN_SCHEME = ['std', 'macer']

parser = argparse.ArgumentParser(description='SWEEN Base Classifiers Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('train_scheme', type=str, choices=TRAIN_SCHEME)
parser.add_argument('--name', default='none', type=str, help='Name of the output ckpt/file')
parser.add_argument('--ckptdir', default='./checkpoint', type=str,
                    help='Checkpoints save directory')
parser.add_argument('--outdir', default='./certout', type=str,
                    help='Outputs of certification save directory')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

# model params
parser.add_argument('--sigma', default=0.25, type=float, help='Standard variance of Gaussian noise')

# params for train
parser.add_argument('--resume_ckpt', default='none', type=str, help='Checkpoint path to resume')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size (default: 64)')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate (default: 0.01)')
parser.add_argument('--lbd', default=12.0, type=float,
                    help='Weight for robustness loss (only used in MACER)')
parser.add_argument('--val_num', default=2000, type=int, help='Nums of images for solving weight')

# params for test
parser.add_argument('--start_img', default=0, 
                    type=int, help='Image index to start (choose it randomly)')
parser.add_argument('--num_img', default=1000, type=int, help='Number of test images')
parser.add_argument('--skip', default=1, type=int, help='Number of skipped images per test image')

args = parser.parse_args()


def main():
    ckptdir = None if args.ckptdir == 'none' else args.ckptdir
    if ckptdir is not None and not os.path.isdir(ckptdir):
        os.makedirs(ckptdir)
    outdir = None if args.outdir == 'none' else args.outdir
    if outdir is not None and not os.path.isdir(outdir):
        os.makedirs(outdir)
    checkpoint = None if args.resume_ckpt == 'none' else args.resume_ckpt
    
    ########## models to train ##########
    model = get_architecture(args.arch)
    name = args.arch if args.name == 'none' else args.name
    
    ########## local args ##########
    start_epoch = 0
    if args.train_scheme == 'std':
        train_epochs = 400
        milestones=[150, 300]
    if args.train_scheme == 'macer':
        train_epochs = 440
        milestones=[200, 400]
        
    ########## dataset ##########
    trainset, testset, transform_test = get_dataset(args.dataset)
    
    train_set = torch.utils.data.Subset(trainset, [i for i in range(len(trainset)-args.val_num)])
    trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    num_classes = 10
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    T0 = time.time()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)
    filename = '{}/{}'.format(outdir,name)
    
    # Resume from checkpoint if required
    if checkpoint is not None:
        print('==> Resuming from checkpoint..')
        print(checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch)

    for epoch in range(start_epoch + 1, train_epochs + 1):
        print('===train(epoch={}, model={})==='.format(epoch, name))
        t1 = time.time()
        model.train()
        
        if args.train_scheme == 'std':
            std_train(args.sigma, 1, num_classes, model, trainloader, optimizer, device)    
        elif args.train_scheme == 'macer':
            macer_train(args.sigma, args.lbd, 16, 16.0, 8.0, num_classes, model, trainloader, optimizer, device)
        else:
            raise ValueError('train_scheme must be either std or macer')
        
        
        scheduler.step()
        t2 = time.time()
        print('Elapsed time: {}'.format(t2 - t1))
    T1 = time.time()
    print('Total elapsed time for training: {}'.format(T1 - T0))
        
    # Certify test
    print('===test(model={})==='.format(name))
    t1 = time.time()
    model.eval()
    certify(model, device, testset, transform_test, num_classes,
            filename, start_img=args.start_img, num_img=args.num_img,
            skip=args.skip, sigma=args.sigma)
    t2 = time.time()
    print('Elapsed time: {}'.format(t2 - t1))
    
    if ckptdir is not None:
      # Save checkpoint
      print('==> Saving model {}.pth..'.format(name))
      try:
        state = {
            'net': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, '{}/{}.pth'.format(ckptdir, name))
      except OSError:
        print('OSError while saving {}.pth'.format(name))
        print('Ignoring...')


if __name__ == "__main__":
    main()