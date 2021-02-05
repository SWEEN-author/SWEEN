from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN

DATASETS = ['cifar10', 'svhn']

root = '../data'


def get_dataset(dataset):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        trainset = CIFAR10(
            root=root, train=True, download=True, transform=transform_train)
        testset = CIFAR10(
            root=root, train=False, download=True, transform=transform_test)
        
        return trainset, testset, transform_test
    elif dataset == 'svhn':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

        trainset = SVHN(
            root=root, split='train', download=True, transform=transform_train)
        testset = SVHN(
            root=root, split='test', download=True, transform=transform_test)
        
        return trainset, testset, transform_test
        