'''
This file is for evaluation of SWEEN models
'''
import argparse
import time
import os

import torch

from rs.certify import certify
from datasets import get_dataset, DATASETS
from architectures import get_architecture

from model.ensemble import Ensemble, AEnsemble
from model.softmaxize import Softmaxize

ENS_COMP = ['3_model_ensemble', '7_model_ensemble']

parser = argparse.ArgumentParser(description='SWEEN Base Classifiers Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('ens_comp', type=str, choices=ENS_COMP)
parser.add_argument('--adp', default=True, type=bool, help='whether to use adaptive prediction')
parser.add_argument('--name', default='none', nargs='+', type=str, help='Name of the base model ckpts')
parser.add_argument('--ens_ckpt', default='ensemble', type=str, help='ckpt file name for weight')
parser.add_argument('--ens_name', default='ens_eval', type=str, help='Name of the output file')
parser.add_argument('--ckptdir', default='./checkpoint', type=str,
                    help='Checkpoints save directory')
parser.add_argument('--outdir', default='./certout', type=str,
                    help='Outputs of certification save directory')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

# model params
parser.add_argument('--sigma', default=0.25, type=float, help='Standard variance of Gaussian noise')


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
      
    ########## dataset ##########
    trainset, testset, transform_test = get_dataset(args.dataset)
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
    
    ens_ckpt = torch.load('{}/{}.pth'.format(ckptdir, args.ens_ckpt))
    if args.adp:
        ens_model = AEnsemble(trained_models, device, w=ens_ckpt['w'],  rt_num=True)
    else:
        ens_model = Ensemble(trained_models, device, w=ens_ckpt['w'])
    ens_model.set_training_paras(weight = False)

    # Certify test
    print('===test(model=ensemble)===')
    t1 = time.time()
    ens_model.eval()
    certify(ens_model, device, testset, transform_test, num_classes,
            '{}/{}'.format(outdir, args.ens_name),
            start_img=args.start_img, num_img=args.num_img, 
            skip=args.skip, sigma=args.sigma, adp=args.adp)
    t2 = time.time()
    print('Elapsed time: {}'.format(t2 - t1))


if __name__ == "__main__":
    main()