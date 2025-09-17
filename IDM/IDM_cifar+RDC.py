import os
import random
import time
import copy
import argparse
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from dc_utils import get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, number_sign_augment, parser_bool, downscale
from dc_utils import epoch as epoch_
import torchnet
import torch.nn.functional as F
import pickle
from losses import SupConLoss
import logging
from torch.utils.data import DataLoader
import kornia.augmentation as K
from utils import get_dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

best_acc = 0

def setup_logging(save_path):
    # 로그 폴더 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 로그 파일 경로 설정
    log_file = os.path.join(save_path, 'training_log.txt')
    
    # 기존 루트 로거 핸들러 제거 (중복 방지)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 글로벌 로깅 설정 (기본 로거를 사용)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # INFO 및 root 제거
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 루트 로거의 propagate를 False로 설정해 중복 방지
    logging.getLogger().propagate = False

def mixup(data_a, data_b, label_a, label_b, alpha=0.75):
    """
    Perform MixUp between two datasets.
    Args:
        data_a, data_b: Image tensors to mix (N, C, H, W)
        label_a, label_b: Label tensors to mix (N, num_classes)
        alpha: Beta distribution parameter
    Returns:
        mixed_data, mixed_labels: Mixed images and labels
    """
    l = np.random.beta(alpha, alpha)  # Sample lambda from Beta distribution
    l = max(l, 1 - l)  # Ensure lambda is symmetric

    mixed_data = l * data_a + (1 - l) * data_b
    mixed_labels = l * label_a + (1 - l) * label_b
    return mixed_data, mixed_labels

def get_augmented_img_syn(img_syn, label_syn, syn_aug=True):
    if syn_aug:
        # Step 1: Augment synthetic images
        augmented_img_syn = []
        augmented_label_syn = []
        for aug_i in range(4):  # 4번 증강
            seed = int(time.time() * 1000) % 100000
            augmented_img_syn.append(DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=ParamDiffAug()))
            augmented_label_syn.append(label_syn)
        
        augmented_img_syn = torch.cat(augmented_img_syn)
        augmented_label_syn = torch.cat(augmented_label_syn)
    else:
        # Augmentation을 하지 않는 경우, 원본 이미지를 그대로 사용
        augmented_img_syn = img_syn
        augmented_label_syn = label_syn

    return augmented_img_syn, augmented_label_syn

def get_mixup_img_sym(img_syn, label_syn, dataset, lambda_val, syn_aug=True):
    augmented_img_syn, augmented_label_syn = get_augmented_img_syn(img_syn, label_syn, syn_aug=syn_aug)

    # Step 2: Apply Mixup between synthetic and randomly labeled real images
    mixed_images = []
    mixed_labels = []

    for i in range(augmented_img_syn.size(0)):
        # 데이터셋에서 무작위로 이미지를 선택
        real_img, real_label = random.choice(dataset)
        real_img = real_img.unsqueeze(0).to(img_syn.device)
        real_label = torch.tensor(real_label, dtype=torch.long, device=img_syn.device)
        
        # Mixup 적용
        mixed_img = lambda_val * augmented_img_syn[i] + (1 - lambda_val) * real_img
        mixed_images.append(mixed_img)
        mixed_labels.append(real_label)

    mixed_images = torch.cat(mixed_images)
    mixed_labels = torch.stack(mixed_labels)

    # Return both SupCon and CE Mixup images and labels
    return mixed_images, mixed_labels

def get_images_dataset(dataset, c=None, n=0, seed=None):
    """
    Get random n images from the specified class c in a given dataset.
    
    Args:
        dataset: The dataset from which to sample images (e.g., `noisy_sample_dataset` or `dst_train`).
        c: The class label to filter images by. If None, samples are taken across all classes.
        n: The number of images to sample. Must be greater than 0.
    
    Returns:
        (images, labels): A tuple containing a tensor of sampled images and their corresponding labels.
    """
    assert n > 0, 'n must be larger than 0'

    if seed is not None:
        np.random.seed(seed)

    # Split dataset indices by class if `c` is provided
    if c is not None:
        indices_class = [i for i, (_, label) in enumerate(dataset) if label == c]
        if len(indices_class) < n:
            raise ValueError(f"Class {c} has fewer than {n} samples in the dataset.")
        idx_shuffle = np.random.choice(indices_class, n, replace=False)
    else:
        # Sample from all classes if no specific class is given
        idx_shuffle = np.random.choice(len(dataset), n, replace=False)

    # Collect images and labels based on the sampled indices
    images = torch.stack([dataset[i][0] for i in idx_shuffle])
    labels = torch.tensor([dataset[i][1] for i in idx_shuffle])

    return images.to(dataset[0][0].device), labels.to(dataset[0][0].device)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 이미지와 라벨을 가져온 뒤, 이미지가 Tensor가 아니라면 Tensor로 변환합니다.
        image = self.images[idx]
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)  # 이미지가 numpy 또는 PIL 이미지일 경우 Tensor로 변환
        
        label = self.labels[idx]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)  # 라벨을 Tensor로 변환

        # transform이 있을 경우 적용
        if self.transform:
            image = self.transform(image)

        return image, label
    
def sharpen(probs, T):
    """
    Apply sharpening to a probability distribution.
    Args:
        probs: Tensor of shape (batch_size, num_classes) - probability distribution
        T: Temperature for sharpening (T < 1.0 sharpens the distribution)
    Returns:
        Sharpened probability tensor of shape (batch_size, num_classes)
    """
    probs_sharpened = probs ** (1 / T)  # Apply temperature scaling
    probs_sharpened = probs_sharpened / probs_sharpened.sum(dim=1, keepdim=True)  # Re-normalize
    return probs_sharpened

def calculate_accuracy(loader, model, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Loss calculation
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            # Prediction and accuracy calculation
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def main():
    global best_acc

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=2, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=7, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Epoch', type=int, default=150, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.2, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.02, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--eval_interval', type=int, default=150, help='Evaluation interval')
    parser.add_argument('--trained_bs', type=int, default=256, help='Batch size for student training')
    parser_bool(parser, 'net_decay', True)  # weight decay flag
    parser.add_argument('--model_train_steps', type=int, default=1, help='Student training steps')
    parser.add_argument('--net_num', type=int, default=1, help='Number of student nets')
    parser.add_argument('--fetch_net_num', type=int, default=1, help='Fetched nets per cycle')
    parser_bool(parser, 'syn_ce', True)  # synthetic CE flag
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer for image tensors')
    parser.add_argument('--train_net_num', type=int, default=1, help='Nets used for image update')
    parser.add_argument('--aug_num', type=int, default=1, help='Augmentation draws')
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for loss')
    parser.add_argument('--lambda_val', type=float, default=0.75, help='MixUp weight')
    parser.add_argument('--warmup_epoch', type=float, default=5, help='Warmup epochs')
    parser.add_argument('--confidency', type=float, default=0.95, help='Confidence threshold')
    parser.add_argument('--lambda_unlabeled', type=float, default=0, help='Unlabeled loss weight')
    parser_bool(parser, 'zca', True)
    parser_bool(parser, 'aug', True)
    parser_bool(parser, 'syn_aug', True)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    checkpoint_dir = args.save_path + '/IDM+RDC_{}_ipc{}_{}/'.format(args.dataset, args.ipc, args.model)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        os.mkdir(checkpoint_dir)

    setup_logging(checkpoint_dir)

    eval_it_pool = np.arange(0, args.Epoch+1, args.eval_interval).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Epoch] # The list of iterations when we evaluate models and record results.
    logging.info('eval_it_pool: %s', eval_it_pool)
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader, _, _, _ = get_dataset(args.dataset, args.data_path, args.batch_real, 'none', args=args)
    train_loader = DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=4)
    eval_loader = DataLoader(dst_train, batch_size=args.batch_train, shuffle=False, num_workers=4)
    
    # Initialize a list to store losses for each data point
    dataset_size = len(dst_train)  # Number of data points in the dataset
    loss_list = [0] * dataset_size  # Initialize with zeros

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    num_iterations = 0

    for exp in range(args.num_exp):
        logging.info('\n================== Exp %d ==================\n ', exp)
        logging.info('Hyper-parameters: \n', args.__dict__)
        logging.info('Evaluation model pool: %s', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            logging.info('class c = %d: %d real images', c, len(indices_class[c]))

        def get_images(c=None, n=0): # get random n images from class c
            if c is not None:
                idx_shuffle = np.random.permutation(indices_class[c])[:n]
                return images_all[idx_shuffle]
            else:
                assert n > 0, 'n must be larger than 0'
                indices_flat = [_ for sublist in indices_class for _ in sublist]
                idx_shuffle = np.random.permutation(indices_flat)[:n]
                return images_all[idx_shuffle], labels_all[idx_shuffle]

        for ch in range(channel):
            logging.info('real images channel %d, mean = %.4f, std = %.4f', ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch]))

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_array = np.concatenate([np.ones(args.ipc) * i for i in range(num_classes)]).astype(np.int64)
        label_syn = torch.tensor(label_array, device=args.device, requires_grad=False)

        logging.info("Shape of Image_SYN: %s", image_syn.shape)

        if args.init == 'real':
            logging.info('initialize synthetic data from random real images')
            for c in range(num_classes):
                if not args.aug:
                    image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
                else:
                    half_size = im_size[0]//2
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :half_size, :half_size] = downscale(get_images(c, args.ipc), 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, half_size:, :half_size] = downscale(get_images(c, args.ipc), 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :half_size, half_size:] = downscale(get_images(c, args.ipc), 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, half_size:, half_size:] = downscale(get_images(c, args.ipc), 0.5).detach().data
        else:
            logging.info('initialize synthetic data from random noise')

        ''' training '''
        if args.optim == 'sgd':
            optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        elif args.optim == 'adam':
            optimizer_img = torch.optim.Adam([image_syn, ], lr=args.lr_img)
        else:
            raise NotImplemented()
        optimizer_img.zero_grad()
        logging.info('%s training begins', get_time())

        ''' Train synthetic data '''
        net_num = args.net_num
        net_list = list()
        optimizer_list = list()
        acc_meters = list()
        for net_index in range(args.net_num):
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            if args.net_decay:
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
            else:
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            net_list.append(net)
            optimizer_list.append(optimizer_net)
            acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))
        
        criterion = nn.CrossEntropyLoss().to(args.device)
        contrastive_criterion = SupConLoss(temperature=args.temp).to(args.device)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=30, gamma=0.1)
        
        for epoch in range(args.Epoch+1):

            ''' Evaluate synthetic data '''
            if epoch in eval_it_pool[1:]:
                for model_eval in model_eval_pool:
                    logging.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, epoch = %d', args.model, model_eval, epoch)
                    logging.info('DSA augmentation strategy: \n%s', args.dsa_strategy)
                    logging.info('DSA augmentation parameters: \n%s', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        if args.aug:
                            image_syn_eval, label_syn_eval = number_sign_augment(image_syn_eval, label_syn_eval)
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    logging.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------', len(accs), model_eval, np.mean(accs), np.std(accs))

                    if epoch == args.Epoch: # record the final results
                        accs_all_exps[model_eval] += accs

                    # save the checkpoint of synthetic set with best performance\
                    best_synset_filename = checkpoint_dir + 'acc_{}.pkl'.format(np.mean(accs))
                    if best_acc < np.mean(accs):
                        best_acc = np.mean(accs)
                        with open(best_synset_filename, 'wb') as pkl_file:
                            pickle.dump((image_syn.detach(), label_syn.detach()), pkl_file)
                            logging.info("Saving best synset with accuracy: %s", np.mean(accs))

            # reset accuracy meters
            acc_report = "Training Acc: "
            # Training accuracy using eval_loader
            for net_index, train_model in enumerate(net_list):
                train_loss, train_acc = calculate_accuracy(eval_loader, train_model, criterion, args.device)
                acc_report += 'Net {}: {:.2f}% '.format(net_index, train_acc * 100)

            acc_report += ' Testing Acc: '
            # Testing accuracy using testloader
            for net_index, test_model in enumerate(net_list):
                test_loss, test_acc = calculate_accuracy(testloader, test_model, criterion, args.device)
                acc_report += 'Net {}: {:.2f}% '.format(net_index, test_acc * 100)
            logging.info(acc_report)

            for batch_idx, (img_real, lab_real) in enumerate(train_loader):
                img_real, lab_real = img_real.to(args.device), lab_real.to(args.device)

                _ = list(range(len(net_list)))
                random.shuffle(_)
                net_index_list = _[:args.train_net_num]
                train_net_list = [net_list[ind] for ind in net_index_list]
                train_acc_list = [acc_meters[ind] for ind in net_index_list]

                embed_list = [net.module.embed_channel_avg if torch.cuda.device_count() > 1 else net.embed_channel_avg for net in train_net_list]

                augmentation_transform = torch.nn.Sequential(
                    K.RandomHorizontalFlip(p=0.5),
                    K.RandomCrop((32, 32), padding=4)
                ).to(args.device)  # GPU로 이동
                loss_avg = 0
                mtt_loss_avg = 0
                metrics = {'syn': 0, 'real': 0}
                acc_avg = {'syn':torchnet.meter.ClassErrorMeter(accuracy=True)}

                ''' update synthetic data '''
                if 'BN' not in args.model or args.model=='ConvNet_GBN': # for ConvNet
                    for image_sign, image_temp in [['syn', image_syn]]:
                        loss = torch.tensor(0.0).to(args.device)
                        for net_ind in range(len(train_net_list)):
                            net = train_net_list[net_ind]
                            net.eval()
                            embed = embed_list[net_ind]
                            net_acc = train_acc_list[net_ind]
                            for c in range(num_classes):
                                loss_c = torch.tensor(0.0).to(args.device)
                                img_real = get_images(c, args.batch_real)
                                img_syn = image_temp[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                                lab_syn = label_syn[c*args.ipc:(c+1)*args.ipc]

                                assert args.aug_num == 1

                                if args.aug:
                                    img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)

                                if args.dsa:
                                    img_real_list = list()
                                    img_syn_list = list()
                                    for aug_i in range(args.aug_num):
                                        seed = int(time.time() * 1000) % 100000
                                        img_real_list.append(DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param))
                                        img_syn_list.append(DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param))
                                    img_real = torch.cat(img_real_list)
                                    img_syn = torch.cat(img_syn_list)
                                
                                if args.ipc == 1 and not args.aug:
                                    logits_real = net(img_real).detach()
                                    loss_real = F.cross_entropy(logits_real, labels_all[indices_class[c]][:img_real.shape[0]], reduction='none')
                                    indices_topk_loss = torch.topk(loss_real, k=2560, largest=False)[1]
                                    img_real = img_real[indices_topk_loss]
                                    metrics['real'] += loss_real[indices_topk_loss].mean().item()

                                output_real = embed(img_real, last=-1).detach()
                                output_syn = embed(img_syn, last=-1)

                                loss_c += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                                logits_syn = net(img_syn)
                                metrics[image_sign] += F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)).detach().item()
                                acc_avg[image_sign].add(logits_syn.detach(), lab_syn.repeat(args.aug_num))

                                syn_ce_loss = 0
                                weight_i = net_acc.value()[0] if net_acc.n != 0 else 0
                                if args.ipc == 1 and not args.aug:
                                    if logits_syn.argmax() != c:
                                        syn_ce_loss += (F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)) * weight_i)
                                else:
                                    syn_ce_loss += (F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)) * weight_i)

                                if args.ipc == 50:
                                    ce_weight = 0.1
                                elif args.ipc == 1 or 10:
                                    ce_weight = 0.5
                                loss_c += (syn_ce_loss * ce_weight)

                                if epoch >= args.warmup_epoch:
                                    augmented_img_syn, augmented_label_syn = get_augmented_img_syn(img_syn, lab_syn, syn_aug=args.syn_aug)
                                    mixed_images, mixed_labels = get_mixup_img_sym(img_syn, lab_syn, clean_dataset, lambda_val=args.lambda_val, syn_aug=args.syn_aug)
                                    features2 = F.normalize(embed(mixed_images), dim=1)
                                    features = F.normalize(embed(augmented_img_syn), dim=1)
                                    loss_c += (contrastive_criterion(features.unsqueeze(1), augmented_label_syn))
                                    loss_c += ((args.lambda_val) * contrastive_criterion(features2.unsqueeze(1), augmented_label_syn) + (1-args.lambda_val)*contrastive_criterion(features2.unsqueeze(1), mixed_labels))

                                optimizer_img.zero_grad()
                                loss_c.backward()
                                optimizer_img.step()

                                loss += loss_c.item()

                        if image_sign == 'syn':
                            loss_avg += loss.item()
                else:
                    raise NotImplemented()

                loss_avg /= (num_classes)
                mtt_loss_avg /= (num_classes)
                metrics = {k:v/num_classes for k, v in metrics.items()}

                shuffled_net_index = list(range(len(net_list)))
                random.shuffle(shuffled_net_index)
                
                if epoch < args.warmup_epoch:
                    for j in range(min(args.fetch_net_num, len(shuffled_net_index))):
                        training_net_idx = shuffled_net_index[j]
                        net_train = net_list[training_net_idx]
                        net_train.train()
                        optimizer_net_train = optimizer_list[training_net_idx]
                        acc_meter_net_train = acc_meters[training_net_idx]
                        for i in range(args.model_train_steps):
                            img_real_, lab_real_ = get_images(c=None, n=args.trained_bs)
                            real_logit = net_train(img_real_)
                            syn_cls_loss = criterion(real_logit, lab_real_)
                            optimizer_net_train.zero_grad()
                            syn_cls_loss.backward()
                            optimizer_net_train.step()
                            acc_meter_net_train.add(real_logit.detach(), lab_real_)
                
                else:
                    for j in range(min(args.fetch_net_num, len(shuffled_net_index))):
                        training_net_idx = shuffled_net_index[j]
                        net_train = net_list[training_net_idx]
                        net_train.train()
                        optimizer_net_train = optimizer_list[training_net_idx]
                        acc_meter_net_train = acc_meters[training_net_idx]

                        for i in range(args.model_train_steps):
                            # X^hat
                            img_real_, lab_real_ = get_images_dataset(clean_dataset, n=args.trained_bs, seed= None)
                            img_real_, lab_real_ = img_real_.to(args.device), lab_real_.to(args.device)

                            # U^hat
                            img_noisy_, _ = get_images_dataset(noisy_dataset, n=args.trained_bs, seed= None)
                            img_noisy_ = img_noisy_.to(args.device)
                            # Apply augmentation sequentially to img_noisy_
                            img_noisy_augmented = augmentation_transform(img_noisy_)
                    
                            # 모델을 한 번에 전체 augmentation batch로 실행
                            with torch.no_grad():
                                pred_logits = net_train(img_noisy_)
                                pseudo_labels = sharpen(pred_logits, 0.5)
                                pred_logits_aug = net_train(img_noisy_augmented)
                                pseudo_labels_aug = sharpen(pred_logits_aug, 0.5)
                                output_real_ = net_train(img_real_)

                            # MixUp for clean and noisy data
                            mixed_images, mixed_labels = mixup(
                                img_real_, img_noisy_,
                                F.one_hot(lab_real_, num_classes=num_classes).float(),
                                pseudo_labels
                            )

                            logits = net_train(mixed_images)

                            # regularization
                            prior = torch.ones(num_classes)/num_classes
                            prior = prior.cuda()        
                            pred_mean = torch.softmax(logits, dim=1).mean(0)
                            penalty = torch.sum(prior*torch.log(prior/pred_mean))

                            # Update model parameters
                            optimizer_net_train.zero_grad()
                            loss = 0
                            loss += -torch.mean(torch.sum(F.log_softmax(output_real_, dim=1) * F.one_hot(lab_real_, num_classes=num_classes).float(), dim=1))
                            loss += F.cross_entropy(logits, mixed_labels.argmax(dim=1))
                            loss += (torch.mean((pseudo_labels - pseudo_labels_aug)**2)) * args.lambda_unlabeled
                            loss += penalty
                            loss.backward()
                            optimizer_net_train.step()

                            real_logit = net_train(img_real_)
                            acc_meter_net_train.add(real_logit.detach(), lab_real_)

                num_iterations += args.fetch_net_num * args.model_train_steps

                if batch_idx%10 == 0:
                    logging.info('%s epoch = %03d, batch_idx = %04d, loss = syn:%.4f, net_list size = %s, metrics = syn:%.4f/real:%.4f, syn acc = syn:%.4f', get_time(), epoch, batch_idx, \
                        loss_avg, str(len(net_list)), metrics['syn'], metrics['real'], acc_avg['syn'].value()[0] if acc_avg['syn'].n!=0 else 0)
            
            last_step = 0

            if num_iterations // 50 > last_step:  # Check if the next 100-step boundary is crossed
                scheduler.step()
                last_step = num_iterations // 100
            
            if (epoch+1) >= args.warmup_epoch:
                # Set model to evaluation mode for inference
                net_train.eval()

                for batch_idx_, (img_real, lab_real) in enumerate(eval_loader):
                    img_real, lab_real = img_real.to(args.device), lab_real.to(args.device)
                
                    with torch.no_grad():
                        real_logit = net_train(img_real)  # Forward pass
                        syn_cls_loss = F.cross_entropy(real_logit, lab_real, reduction='none')  # Loss for each data point in the batch

                    # Store the loss per data point in the loss_list
                    batch_size = img_real.size(0)
                    for k in range(batch_size):
                        data_index = batch_idx_ * args.batch_train + k  # Ensure batch index is correctly multiplied by batch_train size
                        # Ensure we don't go out of bounds when processing the final smaller batch
                        if data_index < len(loss_list):
                            loss_list[data_index] += (syn_cls_loss[k].item() / min(args.fetch_net_num, len(shuffled_net_index)))
                        else:
                            print(f"Skipping index {data_index} as it's out of bounds!")

                # Apply logarithm to each loss
                log_losses = np.log(np.array(loss_list) + 1e-8)  # Avoid log(0) by adding a small value

                # Fit GMM with 2 components on the log of the losses
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(log_losses.reshape(-1, 1))

                # Get the GMM components: means, covariances, and probabilities
                gmm_probs = gmm.predict_proba(log_losses.reshape(-1, 1))

                # Find the lower mean component (clean set)
                component_means = gmm.means_.flatten()
                lower_mean_component = np.argmin(component_means)  # Index of the lower mean (clean images)
                
                # Select clean images where the confidence in the lower-mean component is > 0.5
                selected_indices = np.where(gmm_probs[:, lower_mean_component] > 0.5)[0]
                unselected_indices = np.where(gmm_probs[:, lower_mean_component] <= 0.5)[0]

                logging.info(f"Selected: {len(selected_indices)} images, Unselected: {len(unselected_indices)} images")

                # net_train 모델을 evaluation 모드로 전환
                acc_test_list = []
                for net_index, test_model in enumerate(net_list):
                    test_model.eval()
                    loss_test, acc_test = epoch_('test', testloader, test_model, None, criterion, args, aug = False)
                    acc_test_list.append(acc_test)
                best_net_index = acc_test_list.index(max(acc_test_list))
                net_train = net_list[best_net_index]
                net_train.eval()

                # relabeled_indices 초기화
                relabeled_indices = []
                # dst_train의 이미지와 라벨을 동시에 추출하여 각각 리스트로 저장
                dst_train_images, predicted_labels = zip(*[(data, label) for data, label in dst_train])

                # 리스트 형태로 변환
                dst_train_images = list(dst_train_images)
                predicted_labels = list(predicted_labels)

                confidency_list = []
                relabeled_labels = []

                batch_size = args.batch_train  # 배치 크기 설정 (GPU 메모리 용량에 맞게 조정)

                batch_images = []
                batch_indices = []

                for idx in unselected_indices:
                    image, _ = dst_train[idx]
                    batch_images.append(image)
                    batch_indices.append(idx)

                    # 배치 크기에 도달하면 처리
                    if len(batch_images) == batch_size:
                        # 배치를 텐서로 변환 및 GPU로 이동
                        batch_images_tensor = torch.stack(batch_images).to(args.device)

                        # 모델에 배치 입력
                        with torch.no_grad():
                            outputs = net_train(batch_images_tensor)  # (batch_size, num_classes)

                        # Softmax와 confidence 계산
                        confidences = F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                        predictions = outputs.argmax(dim=1).cpu().numpy()

                        for i, confidence in enumerate(confidences):
                            idx = batch_indices[i]
                            confidency_list.append(confidence)
                            relabeled_labels.append(predictions[i])

                            # confidence가 기준 이상이면 relabel
                            if confidence >= args.confidency:
                                predicted_labels[idx] = predictions[i]
                                relabeled_indices.append(idx)

                        # 배치 초기화
                        batch_images = []
                        batch_indices = []

                # 남은 이미지를 처리 (배치 크기 미만)
                if len(batch_images) > 0:
                    batch_images_tensor = torch.stack(batch_images).to(args.device)
                    with torch.no_grad():
                        outputs = net_train(batch_images_tensor)

                    confidences = F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                    predictions = outputs.argmax(dim=1).cpu().numpy()

                    for i, confidence in enumerate(confidences):
                        idx = batch_indices[i]
                        confidency_list.append(confidence)
                        relabeled_labels.append(predictions[i])

                        if confidence >= args.confidency:
                            predicted_labels[idx] = predictions[i]
                            relabeled_indices.append(idx)
                    
                # relabeled_indices가 빈 리스트일 때 빈 numpy 배열로 설정
                if len(relabeled_indices) == 0:
                    relabeled_indices = np.array([], dtype=int)
                else:
                    relabeled_indices = np.array(relabeled_indices).astype(int)
                
                # selected_indices와 relabeled_indices를 합쳐 clean_indices 생성
                clean_indices = np.concatenate((selected_indices, relabeled_indices))

                # unselected_indices에서 relabeled_indices를 제외하여 noisy_indices 생성
                noisy_indices = np.setdiff1d(unselected_indices, relabeled_indices)

                # Assuming dst_train is your noisy CIFAR-10N dataset
                noisy_labels = np.array([dst_train[i][1] for i in range(len(dst_train))])  # Noisy CIFAR10N labels

                # clean_dataset과 noisy_dataset을 직접 생성
                clean_images = []
                clean_labels = []
                for idx in clean_indices:
                    clean_images.append(dst_train_images[idx])
                    clean_labels.append(predicted_labels[idx])

                noisy_images = []
                noisy_labels = []
                for idx in noisy_indices:
                    noisy_image, noisy_label = dst_train[idx]  # dst_train에서 원래 라벨과 이미지를 가져옴
                    noisy_images.append(noisy_image)
                    noisy_labels.append(noisy_label)

                # clean_dataset과 noisy_dataset 인스턴스 생성
                clean_dataset = CustomDataset(clean_images, clean_labels)
                noisy_dataset = CustomDataset(noisy_images, noisy_labels)

                logging.info(f"Clean dataset: {len(clean_indices)}, Noisy dataset: {len(noisy_indices)}")

                # 데이터 로더 설정 (배치 크기는 필요에 따라 조정)
                train_loader = DataLoader(clean_dataset, batch_size=args.batch_train, shuffle=True, num_workers=4, collate_fn=torch.utils.data.default_collate)

            logging.info('%s epoch = %03d, loss = syn:%.4f, net_list size = %s, metrics = syn:%.4f/real:%.4f, syn acc = syn:%.4f', get_time(), epoch, \
                loss_avg, str(len(net_list)), metrics['syn'], metrics['real'], acc_avg['syn'].value()[0] if acc_avg['syn'].n!=0 else 0)
            num_epochs_trained = num_iterations / ((50000 / args.batch_train) * args.net_num)
            logging.info("Number of Epoch Trained on Model: {:.2f}".format(num_epochs_trained))

    logging.info('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logging.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%', args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100)

if __name__ == '__main__':
    main()