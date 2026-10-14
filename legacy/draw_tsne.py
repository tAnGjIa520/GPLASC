'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
'''

from __future__ import print_function

import os
import copy
import sys
import argparse
import shutil
import time
import math
import random
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from datasets import TinyImagenet
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, load_model
from networks.resnet_big import SupConResNet
from losses_negative_only import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--target_task', type=int, default=0)

    parser.add_argument('--resume_target_task', type=int, default=None)

    parser.add_argument('--end_task', type=int, default=None)

    parser.add_argument('--replay_policy', type=str, choices=['random'], default='random')

    parser.add_argument('--mem_size', type=int, default=200)

    parser.add_argument('--cls_per_task', type=int, default=2)

    parser.add_argument('--distill_power', type=float, default=1.0)

    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')

    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100','tiny-imagenet', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')


    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--exp_name', type=str, default=None,
                        help="exp's name ")




    opt = parser.parse_args()

    opt.save_freq = opt.epochs // 2


    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 10
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 20
        opt.size = 32
    else:
        pass


    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.model_path = './save_{}_{}/{}_models'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.tb_path = './save_{}_{}/{}_tensorboard'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.log_path = './save_{}_{}/logs'.format(opt.replay_policy, opt.mem_size, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_{}_{}_{}_{}'. \
        format(opt.dataset, opt.size, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp,
               opt.trial,
               opt.start_epoch if opt.start_epoch is not None else opt.epochs, opt.epochs,
               opt.current_temp,
               opt.past_temp,
               opt.distill_power
               )



    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    # todo 加入exp名字

    if opt.exp_name is not None:
        opt.model_name = opt.exp_name


    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_replay_samples(opt, model, prev_indices=None):

    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # construct data loader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'cifar100':
        subset_indices = []
        print(opt.data_folder)
        print(os.getcwd())
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = val_dataset.targets

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
    else:
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                prev_indices += torch.tensor(_prev_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices

    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()


    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)


    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float -  ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        mask = val_targets[observed_indices] == c
        selected_observed_indices += torch.tensor(observed_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()
    print(np.unique(val_targets[selected_observed_indices], return_counts=True))


    model.is_training = is_training

    return prev_indices + selected_observed_indices


def set_loader(opt, replay_indices,target_task):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'cifar100':
        mean = (0.5153, 0.4961, 0.4497)
        std = (0.2608, 0.2551, 0.2779)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])


    target_classes = list(range(target_task*opt.cls_per_task, (target_task+1)*opt.cls_per_task))
    print(target_classes)

    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True,
                                          train=True)
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset =  Subset(_train_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])
    elif opt.dataset == 'cifar100':
        subset_indices = []


        _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                           transform=TwoCropTransform(train_transform),
                                           download=True)
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list


        subset_indices += replay_indices

        train_dataset = Subset(_train_dataset, subset_indices)
        print('dataset length', len(_train_dataset), len(train_dataset))
        print('Dataset size: {}'.format(len(subset_indices)))

        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print('uc[np.argsort(uk)]', uc[np.argsort(uk)])


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices



def set_model(opt,model_name='./save_random_200/cifar10_models/contrastive_learning_representation_no_distall/last_random_4.pth'):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    ckpt = torch.load(
          model_name
        ,
        map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("encoder.module.", "encoder.")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model = model.cuda()
    model.load_state_dict(state_dict,strict=False)
    model=model.cpu()
    return model, criterion


def train(train_loader, model, model2, criterion, optimizer, epoch, opt):


    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)


        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, encoded = model(images, return_feat=True)

        # IRD (current)
        if opt.target_task > 0:
            features1_prev_task = features

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        # Asym SupCon
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))

        # IRD (past)
        if opt.target_task > 0:
            with torch.no_grad():
                features2_prev_task = model2(images)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)



            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += opt.distill_power * loss_distill
            distill.update(loss_distill.item(), bsz)


        # update metric
        losses.update(loss.item(), bsz)

        # SGDx
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, distill=distill))
            sys.stdout.flush()



    return losses.avg, model2
import random
def cal_distance_(model,imagesA,imagesB,equals=False):
    # 计算A和B两个类在dataloader上的余弦相似度
    similarity=0

    with torch.no_grad():

        if torch.cuda.is_available():
            imagesA = imagesA.cuda(non_blocking=True)
            imagesB = imagesB.cuda(non_blocking=True)

        featuresA, encodedA = model(imagesA, return_feat=True)


        featuresB, encodedB = model(imagesB, return_feat=True)

        featuresA=encodedA
        featuresB=encodedB


        if equals:
            # random.shuffle(featuresA)
            # featuresA.mean(dim=0)
            return torch.cosine_similarity(featuresA, torch.unsqueeze(featuresA.mean(dim=0),dim=0), dim=-1).mean().item()
        else:
            similarity=torch.cosine_similarity(featuresA, torch.unsqueeze(featuresB.mean(dim=0),dim=0), dim=-1).mean().item()
            return similarity




from tqdm import tqdm

def print_array(arr):
    for row in arr:
        for element in row:
            print('%.3f'% element, end=" ")
        print()

# def get_tsne():


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def main():
    opt = parse_option()

    replay_indices = []




    if opt.end_task is not None:
        if opt.resume_target_task is not None:
            assert opt.end_task > opt.resume_target_task
        opt.end_task = min(opt.end_task+1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task

    # nodistall
    list_={
        "SupCon":"/home/algroup/parnec/tangjia/Co2L-main/save_random_200/cifar10_models/contrastive_learning_representation_no_distall/last_random_4.pth",
        "Co2l":"/home/algroup/parnec/tangjia/Co2L-main/save_random_200/cifar10_models/contrastive_learning_representation_distall/last_random_4.pth",
        'our method':'save_random_200/cifar10_models/cifar10_drift_0.05_etf_1/last_random_4.pth',
        # 'cclis':"/home/algroup/parnec/tangjia/CCLIS-main-etf-2/CCLIS/save_weight_500_/cifar10_models/cifar10_distill_lamda_0.1_local_lamda_1/last_weight_4.pth",
        'ours': "/home/algroup/parnec/tangjia/CCLIS-main-etf-2/CCLIS/save_weight_500_/cifar10_models/cifar10_distill_lamda_0.1_local_lamda_1/last_weight_4.pth",

        # 'oracle':"/home/algroup/parnec/tangjia/Co2L-main/save_random_200/cifar10_models/oracle_cifar10/last_random_0.pth"

    }
    plt.rcParams.update({'font.size': 15})
    for key,model_name in list_.items():

        plt.figure(figsize=(9, 6))
        model, criterion = set_model(opt, model_name=model_name)
        # from tangjia.tsne import draw_simple_tsne_
        tsne = TSNE(n_components=2, n_iter=3000,perplexity=10, random_state=42)
        palette = sns.color_palette("bright", n_colors=10)  #
        # encoded_list=[]
        # labels_list=[]
        for target_task in tqdm(range(0 if opt.resume_target_task is None else opt.resume_target_task+1, opt.end_task)):
            train_loader, subset_indices = set_loader(opt, replay_indices,target_task)
            for idx, (images, labels) in enumerate(train_loader):

                # images = torch.cat([, images[1]], dim=0)
                features,encoded = model(images[0], return_feat=True)
                encoded= F.normalize(encoded, p=2, dim=1)
                encoded=encoded.detach().cpu().numpy()
                labels=labels.detach().cpu().numpy()
                x_value=encoded
                y_value=labels
                # 01 23 45 67 89
                target_values = torch.tensor([1,2,5,7,8])
                filter=np.isin(y_value,target_values)
                x_value=x_value[filter]
                y_value=y_value[filter]

                x_tsne = tsne.fit_transform(x_value)
                sns.scatterplot(x=x_tsne[:, 0], y=x_tsne[:, 1], hue=y_value, palette=palette, s=50)

                break

        legend = plt.legend(loc='upper right')
        new_labels = ['automobile(task1)', 'bird(task2)', 'dog(task3)', 'horse(task4)', 'ship(task5)']
        for text, new_label in zip(legend.texts[0:], new_labels):  # 跳过第一个标题文本
            text.set_text(new_label)

        plt.gca().axes.xaxis.set_visible(False)  # 隐藏 X 轴
        plt.gca().axes.yaxis.set_visible(False)

        plt.title("t-SNE with {}".format(key))
        plt.show()

if __name__ == '__main__':
    main()
