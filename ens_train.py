'''
This file is for solving weight
'''
import argparse
import time
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from training.ens import ens_train
from rs.certify import certify
from datasets import get_dataset, DATASETS
from architectures import get_architecture

from model.ensemble import Ensemble
from model.softmaxize import Softmaxize

ENS_COMP = ['3-model-ensemble', '7-model-ensemble']

parser = argparse.ArgumentParser(description='SWEEN Base Classifiers Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('ens_comp', type=str, choices=ENS_COMP)
parser.add_argument('--name', default='none', nargs='+', type=str, help='Name of the ckpts')
parser.add_argument('--ens_name', default='ensemble', type=str, help='Name of the output ckpt/file')
parser.add_argument('--ckptdir', default='./checkpoint', type=str,
                    help='Checkpoints save directory')
parser.add_argument('--outdir', default='./certout', type=str,
                    help='Outputs of certification save directory')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

# model params
parser.add_argument('--sigma', default=0.25, type=float, help='Standard variance of Gaussian noise')

# params for train
parser.add_argument('--batch_size', default=64, type=int, help='Batch size (default: 64)')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate (default: 0.01)')
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
    
    ########## models to train ##########
    if args.ens_comp == '3-model-ensemble':
        model_list = ['resnet20', 'resnet26', 'resnet32']
    elif args.ens_comp == '7-model-ensemble':
        model_list = ['lenet', 'alexnet', 'resnet20', 'resnet110', 'densenet', 'vgg16', 'vgg19']
    name = model_list if args.name == 'none' else args.name
    
    assert isinstance(name, list), 'name must be a list'
    assert len(name) == len(model_list), 'the lengths of name and model_list must be equal'
    
    ########## local args ##########
    start_epoch = 0
    train_epochs = 150
    milestones=[60, 90]
        
    ########## dataset ##########
    trainset, testset, transform_test = get_dataset(args.dataset)

    val_set = torch.utils.data.Subset(trainset, 
                                      [i for i in range(len(trainset)-args.val_num,len(trainset))])
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    num_classes = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ########### load trained models ##############
    trained_models = []
    for model_id in range(len(model_list)):
        checkpoint = torch.load('{}/{}.pth'.format(ckptdir, name[model_id]))
        model = get_architecture(model_list[model_id])
        model.load_state_dict(checkpoint['net'])
        model = Softmaxize(model)
        model = model.to(device)
        model.eval()
        trained_models.append(model)

    ############### Training Ensemble Paras ###############
    ens_model = Ensemble(trained_models, device)
    ens_model.set_training_paras()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, ens_model.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    T0 = time.time()
    for epoch in range(start_epoch + 1, train_epochs + 1):
        print('===train(epoch={}, model=ensemble)==='.format(epoch))
        t1 = time.time()
        scheduler.step()
        ens_model.eval()
        
    
        ens_train(args.sigma, 1, num_classes, ens_model, valloader, optimizer, device)
    
        t2 = time.time()
        print('Elapsed time: {}'.format(t2 - t1))
        print('current w:',ens_model.w.data)
        print('current alpha:',(ens_model.w.exp()/ens_model.w.exp().sum()).data)
        
        
    T1 = time.time()
    print('Total elapsed time for weight solving: {}'.format(T1 - T0))
    # Certify test
    print('===test(model=ensemble)===')
    t1 = time.time()
    ens_model.eval()
    certify(ens_model, device, testset, transform_test, num_classes,
            '{}/{}'.format(outdir, args.ens_name),
            start_img=args.start_img, num_img=args.num_img, 
            skip=args.skip, sigma=args.sigma)
    t2 = time.time()
    print('Elapsed time: {}'.format(t2 - t1))
    
    
    if ckptdir is not None:
      # Save checkpoint
      print('==> Saving Ens model.pth..')
      try:
        state = {
            'w': ens_model.w.data,
        }
        torch.save(state, '{}/{}.pth'.format(ckptdir, args.ens_name))
      except OSError:
        print('OSError while saving model {}.pth'.format(args.ens_name))
        print('Ignoring...')


if __name__ == "__main__":
    main()