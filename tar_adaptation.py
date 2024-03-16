import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from time import time
import statistics as st
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from collections import Counter

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.distributed as dist
torch.autograd.set_detect_anomaly(True)

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()


    txt_src = [folder + args.dset + '/' + s for s in txt_src]
    txt_tar = [folder + args.dset + '/' + s for s in txt_tar]
    txt_test = [folder + args.dset + '/' + s for s in txt_test]

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def cal_acc(loader, netF, netB, netC):
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            outputs = netC(fea)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    harmonic = st.harmonic_mean(acc)
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = " ".join(aa)
    return aacc, acc, harmonic

def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    modelpath = args.output_dir_src + f"/{str(args.seed)}/source_F.pt"
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + f"/{str(args.seed)}/source_B.pt"
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + f"/{str(args.seed)}/source_C.pt"
    netC.load_state_dict(torch.load(modelpath))

    netF_ = network.ResBase(res_name=args.net).cuda()
    netB_ = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC_ = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    for param_q, param_k in zip(netF.parameters(), netF_.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient
    for param_q, param_k in zip(netB.parameters(), netB_.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient
    for param_q, param_k in zip(netC.parameters(), netC_.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        param_group += [{"params": v, "lr": args.lr * args.lr_F}]  # 0.1
    for k, v in netB.named_parameters():
        param_group += [{"params": v, "lr": args.lr * args.lr_B}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * args.lr_C}]  # 1

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, 12)
    label_bank = torch.randn(num_sample)
    pseudo_bank = torch.randn(num_sample).long()

    netF.eval()
    netB.eval()
    netC.eval()

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            inputs, labels, indx = next(iter_test)
            inputs = inputs.cuda()
            labels = labels.type(torch.FloatTensor)
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs_ = netC(output)
            outputs = nn.Softmax(-1)(outputs_)
            pseudo_label = torch.argmax(outputs, 1)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone().cpu()
            label_bank[indx] = labels
            pseudo_bank[indx] = pseudo_label.detach().clone().cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    real_max_iter = max_iter
    rho = torch.ones([args.class_num]).cuda() / args.class_num
    Cov = torch.zeros(args.class_num, 256, 256).cuda()
    Ave = torch.zeros(args.class_num, 256).cuda()
    Amount = torch.zeros(args.class_num).cuda()
    epoch = 0
    with tqdm(total=real_max_iter) as pbar:
        while iter_num < real_max_iter:
            start = time()
            try:
                inputs_test, _, tar_idx = next(iter_test)
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = next(iter_test)

            if inputs_test.size(0) == 1:
                continue

            inputs_test = inputs_test.cuda()
            if True:
                alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
            else:
                alpha = args.alpha

            iter_num += 1
            pbar.update(1)
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

            features_test = netB(netF(inputs_test))
            output_f_norm = F.normalize(features_test)
            w = F.normalize(netC.fc.weight_v) * netC.fc.weight_g
            pred_test = netC(features_test)
            score_test = nn.Softmax(dim=1)(pred_test)

            pseudo_label = torch.argmax(score_test, 1).detach()
            top2 = torch.topk(score_test, 2).values
            margin = top2[:,0] - top2[:,1]

            with torch.no_grad():
                output_f_ = output_f_norm.cpu().detach().clone()
                fea_bank[tar_idx] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = score_test.detach().clone().cpu()
                pseudo_bank[tar_idx] = pseudo_label.detach().clone().cpu()
                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]

            ## SNC
            rho_batch = torch.histc(pseudo_label, bins=args.class_num, min=0, max=args.class_num - 1) / inputs_test.shape[0]
            rho = 0.95*rho + 0.05*rho_batch

            softmax_out_un = score_test.unsqueeze(1).expand(
                -1, args.K, -1
            ).cuda()

            loss_pos = torch.mean(
                (F.kl_div(softmax_out_un, score_near.cuda(), reduction="none").sum(-1)).sum(1)
            )
            loss = loss_pos
            loss_pos_print = loss_pos.clone().detach()

            mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy = score_test.T

            dot_neg = score_test @ copy

            dot_neg = ((dot_neg**2) * mask.cuda()).sum(-1)
            neg_pred = torch.mean(dot_neg)
            loss_neg = neg_pred * alpha
            loss_neg_print = loss_neg.clone().detach()
            loss += loss_neg

            ## IFA
            ratio = args.lambda_0 * (iter_num / max_iter)
            maxprob,_=torch.max(score_test,dim=1)
            Amount, Ave, Cov = update_CV(features_test, pseudo_label, Amount, Ave, Cov)
            loss_ifa_, sigma2 = IFA(w, features_test, pred_test, Cov, ratio)
            loss_ifa = args.alpha_1 * torch.mean(loss_ifa_)
            loss_ifa_print = loss_ifa.clone().detach()
            loss += loss_ifa

            ## FD
            mean_score = torch.stack([torch.mean(score_bank[pseudo_bank==i], dim=0) for i in range(args.class_num)])
            cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(args.class_num))
            Cov1 = Cov.view(args.class_num,-1).unsqueeze(1)
            Cov0 = Cov.view(args.class_num,-1).unsqueeze(0)
            cov_distance = 1 - torch.sum((Cov1*Cov0),dim=2) / (torch.norm(Cov1, dim=2) * torch.norm(Cov0, dim=2) + 1e-12)
            loss_fd = -torch.sum(cov_distance * cov_weight.cuda().detach()) / 2
            loss_fd_print = loss_fd.clone().detach()
            loss += args.alpha_2 * loss_fd

            optimizer.zero_grad()
            optimizer_c.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_c.step()

            pbar.set_description(f'iter_num:{iter_num}; time:{time()-start:.2f} sec; loss_pos:{loss_pos_print.item():.4f}; loss_neg:{loss_neg_print.item():.4f}; loss_IFA:{loss_ifa_print.item():.4f}; loss_FD:{loss_fd_print.item():.4f}')
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                epoch += 1
                start_inference = time()
                netF.eval()
                netB.eval()
                netC.eval()
                if args.dset == "visda-2017":
                    acc, accc, harmonic = cal_acc(
                        dset_loaders["test"],
                        netF,
                        netB,
                        netC,
                    )
                    log_str = (
                        "Task: {}, Iter:{}/{}; epoch:{}; Arithmetic: {:.2f}".format(
                            args.name, iter_num, max_iter, epoch, acc
                        )
                        + "\n"
                        + "T: "
                        + accc
                    )

                args.out_file.write(log_str + "\n")
                args.out_file.flush()
                print("\n" + log_str + "\n")
                netF.train()
                netB.train()
                netC.train()

                pbar.set_description(f'inference time:{time()-start_inference:.2f} sec')
    if args.issave:
        args.output_dir += f'/{str(args.seed)}/'
        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F.pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B.pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C.pt"))
    # return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def update_CV(features, labels, Amount, Ave, Cov):
    N = features.size(0)
    C = args.class_num
    A = features.size(1)

    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)

    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A) # mask

    features_by_sort = NxCxFeatures.mul(NxCxA_onehot) # masking

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1

    ave_CxA = features_by_sort.sum(0) / Amount_CxA

    var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

    var_temp = torch.bmm(
        var_temp.permute(1, 2, 0),
        var_temp.permute(1, 0, 2)
    ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

    sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

    sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

    weight_CV = sum_weight_CV.div(
        sum_weight_CV + Amount.view(C, 1, 1).expand(C, A, A)
    )
    weight_CV[weight_CV != weight_CV] = 0

    weight_AV = sum_weight_AV.div(
        sum_weight_AV + Amount.view(C, 1).expand(C, A)
    )
    weight_AV[weight_AV != weight_AV] = 0

    additional_CV = weight_CV.mul(1 - weight_CV).mul(
        torch.bmm(
            (Ave - ave_CxA).view(C, A, 1),
            (Ave - ave_CxA).view(C, 1, A)
        )
    )

    Cov = (Cov.mul(1 - weight_CV).detach() + var_temp.mul(weight_CV)) + additional_CV
    Ave = (Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
    Amount = Amount + onehot.sum(0)
    return Amount, Ave, Cov

def IFA(w, features, logit, cv_matrix, ratio):
    N = features.size(0)
    C = args.class_num
    A = features.size(1)
    log_prob_ifa_ = []
    sigma2_ = []
    pseudo_labels = torch.argmax(logit, dim=1).detach()
    for i in range(C):
        labels = (torch.ones(N)*i).cuda().long()
        NxW_ij = w.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        CV_temp = cv_matrix[pseudo_labels]

        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij-NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        with torch.no_grad():
            sigma2_.append(torch.mean(sigma2))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)
        ifa_logit = logit + 0.5 * sigma2
        log_prob_ifa_.append(F.cross_entropy(ifa_logit, labels, reduction='none'))
    log_prob_ifa = torch.stack(log_prob_ifa_)
    loss = torch.sum(2 * log_prob_ifa.T, dim=1)
    return loss, torch.stack(sigma2_)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="7", help="device id to run")
    parser.add_argument("--s", type=int, default=0, help="soure")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=8, help="number of workers")
    parser.add_argument("--dset", type=str, default="visda-2017")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate") #1e-3
    parser.add_argument("--net", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=2021, help="random seed") #2021

    parser.add_argument("--alpha_1", type=float, default=1e-4)
    parser.add_argument("--alpha_2", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--lambda_0", type=float, default=5.0)
    parser.add_argument("--lr_F", type=float, default=0.1)
    parser.add_argument("--lr_B", type=float, default=1.0)
    parser.add_argument("--lr_C", type=float, default=1.0)
    parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="weight/target/")
    parser.add_argument("--output_src", type=str, default="weight/source/")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--issave", type=bool, default=False)
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--nuclear", default=False, action="store_true")
    parser.add_argument("--var", default=False, action="store_true")
    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dset == "visda-2017":
        names = ["train", "validation"]
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = "./data/"
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src, args.da, args.dset, names[args.s][0].upper()
        )
        args.output_dir = osp.join(
            args.output,
            args.da,
            args.dset,
            names[args.s][0].upper() + names[args.t][0].upper()
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(
            osp.join(args.output_dir, "log_{}.txt".format(args.tag)), "w"
        )
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)
