import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch, train_epoch_semisupervised
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import glob
import random
from weight_perturbation import setup_directions, get_weights, set_weights, set_states, setup_directions_random
import copy
import math
from losses import SupConLoss
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

class Synthesizer():
    """Condensed data class
    """

    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            # print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor ** 2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                        w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor ** 2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor ** 2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        best_acc, last_acc = test_data(args, loader, val_loader, repeat=3, test_resnet=False, logger=logger)
        return best_acc, last_acc

        # if bench and not (args.dataset in ['mnist', 'fashion']):
        #     test_data(args, loader, val_loader, test_resnet=True, logger=logger)


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar10n_asym_20':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        data = torch.load(os.path.join('/data/heeyeon/Dataset Condensation/Acc-DD/torchdata/CIFAR10N_asym_20.pt'), map_location='cpu', weights_only=True)
        train_dataset.targets = data
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        train_dataset.nclass = 10
    elif args.dataset == 'cifar10n_asym_40':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        data = torch.load(os.path.join('/data/heeyeon/Dataset Condensation/Acc-DD/torchdata/CIFAR10N_asym_40.pt'), map_location='cpu', weights_only=True)
        train_dataset.targets = data
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        train_dataset.nclass = 10
    elif args.dataset == 'cifar10n_sym_20':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        data = torch.load(os.path.join('/data/heeyeon/Dataset Condensation/Acc-DD/torchdata/CIFAR10N_sym_20.pt'), map_location='cpu', weights_only=True)
        train_dataset.targets = data
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        train_dataset.nclass = 10
    elif args.dataset == 'cifar10n_sym_40':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        data = torch.load(os.path.join('/data/heeyeon/Dataset Condensation/Acc-DD/torchdata/CIFAR10N_sym_40.pt'), map_location='cpu', weights_only=True)
        train_dataset.targets = data
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        train_dataset.nclass = 10
    elif args.dataset == 'cifar10n_ran1':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        data = torch.load(os.path.join('/data/heeyeon/Dataset Condensation/Acc-DD/torchdata/CIFAR-10_human.pt'), map_location='cpu')
        labels = data['random_label1']
        train_dataset.targets = labels
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        train_dataset.nclass = 10
    elif args.dataset == 'cifar10n_worse':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        data = torch.load(os.path.join('/data/heeyeon/Dataset Condensation/Acc-DD/torchdata/CIFAR-10_human.pt'), map_location='cpu')
        labels = data['worse_label']
        train_dataset.targets = labels
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    if args.dataset[:9]=='cifar10n_':
        normalize = utils.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'], device=device)
    else:
        normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    """Matching losses (feature or gradient)
    """
    loss = None

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))

    elif args.match == 'grad':
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

    return loss

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
        diff_augment = DiffAug(strategy='color_crop_cutout_flip_scale_rotate', batch=True)

        # Step 1: Augment synthetic images (batch-wise)
        augmented_img_syn = [diff_augment(img_syn) for _ in range(4)]  # 4번 augmentation 반복
        augmented_img_syn = torch.cat(augmented_img_syn)  # augmentation 결과를 하나로 합침
        augmented_label_syn = label_syn.repeat(4)  # 레이블을 4번 반복
    else:
        # Augmentation을 하지 않는 경우, 원본 이미지를 그대로 사용
        augmented_img_syn = img_syn
        augmented_label_syn = label_syn

    return augmented_img_syn, augmented_label_syn

def get_mixup_img_sym(img_syn, label_syn, dataset, lambda_val, syn_aug=True):
    """
    Perform Mixup between synthetic images and real images from a data loader.

    Args:
        img_syn: Synthetic images (N, C, H, W).
        label_syn: Synthetic labels (N).
        data_loader: DataLoader object for real dataset.
        lambda_val: Mixup coefficient (0 <= lambda_val <= 1).
        syn_aug: Whether to apply augmentation on synthetic images.

    Returns:
        mixed_images: Mixup images (N, C, H, W).
        mixed_labels: Mixup labels (N, num_classes).
    """
    # Step 1: Augment synthetic images
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

    # Return mixed images and labels
    return mixed_images, mixed_labels

def supcon_loss(img_syn, lab_syn, model, clean_dataset, args):
    """
    Compute the SupCon loss for synthetic and mixed data.

    Args:
        img_syn: Synthetic images (N, C, H, W).
        lab_syn: Synthetic labels (N).
        model: Model used to extract embeddings.
        clean_dataset_loader: DataLoader for the real clean dataset.
        contrastive_criterion: SupConLoss criterion.
        args: Arguments containing temperature, syn_aug, lambda_val, etc.

    Returns:
        loss_c: Total contrastive loss.
    """
    contrastive_criterion = SupConLoss(temperature=args.temp)

    # Step 1: Augment synthetic images
    augmented_img_syn, augmented_label_syn = get_augmented_img_syn(img_syn, lab_syn, syn_aug=True)

    # Step 2: Mixup between synthetic and real images
    mixed_images, mixed_labels = get_mixup_img_sym(
        img_syn, lab_syn, clean_dataset, lambda_val=args.lambda_val, syn_aug=True
    )

    # Step 3: Extract features using the model
    features_augmented = F.normalize(model(augmented_img_syn), dim=1)  # Augmented synthetic features
    features_mixed = F.normalize(model(mixed_images), dim=1)  # Mixed features

    # Step 4: Compute SupCon loss
    loss = 0
    # Loss for augmented synthetic data
    loss += contrastive_criterion(features_augmented.unsqueeze(1), augmented_label_syn)
    # Loss for mixed data
    loss += (args.lambda_val * contrastive_criterion(features_mixed.unsqueeze(1), augmented_label_syn) + \
             (1 - args.lambda_val) * contrastive_criterion(features_mixed.unsqueeze(1), mixed_labels))

    return loss

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.targets = labels
        self.nclass = args.nclass
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

def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def load_state(file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    return checkpoint


def condense(args, logger, device='cuda'):
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
        eval_loader = DataLoader(trainset, batch_size=args.batch_real, shuffle=False, num_workers=4)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
        eval_loader = DataLoader(trainset, batch_size=args.batch_real, shuffle=False, num_workers=4)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    synset = Synthesizer(args, nclass, nch, hs, ws)
    synset.init(loader_real, init_type=args.init)
    save_img(os.path.join(args.save_dir, 'init.png'),
             synset.data,
             unnormalize=False,
             dataname=args.dataset)

    aug, aug_rand = diffaug(args)
    save_img(os.path.join(args.save_dir, f'aug.png'),
             aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
             unnormalize=True,
             dataname=args.dataset)

    if not args.test:
        synset.test(args, val_loader, logger, bench=False)

    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

    ts = utils.TimeStamp(args.time)
    n_iter = args.niter * 100 // args.inner_loop
    it_log = n_iter // 50
    # it_test = [n_iter // 200, n_iter // 100, n_iter // 50, n_iter // 25, n_iter // 10, n_iter // 5, n_iter // 2, n_iter]

    logger(f"\nStart condensing with {args.match} matching for {args.niter} iteration")
    args.fix_iter = max(1, args.fix_iter)

    for it in range(args.niter):
        if it%5==0:
            loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
            file_dir_set = args.model_path                               # path of checkpoints pretrained models 'pth.tar
            if isinstance(file_dir_set, str):
                file_dir_set = [file_dir_set]  
            args.model_num = len(file_dir_set)
            pth_idx = random.randint(0, args.model_num-1)
            current_state_dict = load_state(file_dir_set[pth_idx])
            model = define_model(args, nclass)
            model.load_state_dict(current_state_dict)
            model_param = [device, nclass, file_dir_set[pth_idx]]
            xdirection, ydirection = setup_directions(args, model, model_param)
            w = get_weights(model)
            s = copy.deepcopy(model.state_dict())
            xcoordinates = random.uniform(args.vmax, args.vmin) * math.pow(-1, random.randint(1, 100))
            ycoordinates = random.uniform(args.vmax, args.vmin) * math.pow(-1, random.randint(1, 100))
            if args.dir_type == 'weights':
                set_weights(model, w, [xdirection, ydirection], [xcoordinates, ycoordinates])
            elif args.dir_type == 'states':
                set_states(model, s, [xdirection, ydirection], [xcoordinates, ycoordinates])
            model.train()
            model = model.to(device)
            optimizer_net = optim.SGD(model.parameters(),
                                        args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            criterion = nn.CrossEntropyLoss()
            loss_list = [0] * 50000

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        for ot in range(args.inner_loop):
            ts.set()

            for c in range(nclass):
                img, lab = loader_real.class_sample(c)
                img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                ts.stamp("data")

                n = img.shape[0]
                if n < args.batch_real:
                    deficit = args.batch_real - n  # 부족한 개수 계산

                    # img를 부족한 개수만큼 복제하여 augmentation 수행
                    img_extra = aug(img.repeat((deficit // n) + 1, 1, 1, 1))[:deficit]
                    lab_extra = lab.repeat((deficit // n) + 1)[:deficit]  # 레이블도 동일하게 반복 후 자르기

                    # 기존 데이터와 합쳐서 batch_real 크기로 맞춤
                    img = torch.cat([img, img_extra], dim=0)
                    img = img[:args.batch_real]
                    lab = torch.cat([lab, lab_extra], dim=0)
                    lab = lab[:args.batch_real]
                img_aug = aug(torch.cat([img, img_syn]))
                ts.stamp("aug")

                loss = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                if it%5>=args.warmup_epoch:
                    loss += supcon_loss(img_syn, lab_syn, model, golden_dataset, args)
                loss_total += loss.item()
                ts.stamp("loss")

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()
                ts.stamp("backward")

            if args.n_data > 0:
                for _ in range(args.net_epoch):
                    if it%5 < args.warmup_epoch:
                        train_epoch(args,
                                    loader_real,
                                    model,
                                    criterion,
                                    optimizer_net,
                                    n_data=args.n_data,
                                    aug=aug_rand,
                                    mixup=args.mixup_net)
                    else:
                        train_epoch_semisupervised(args,
                            loader_real,
                            loader_noisy,
                            model,
                            optimizer_net,
                            n_data=args.n_data,
                            aug=aug_rand)

            if (ot + 1) % 10 == 0:
                ts.flush()

        logger(
            f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total / nclass / args.inner_loop:.2f}")

        if (it + 1) % args.val_interval == 0:
            save_img(os.path.join(args.save_dir, f'img{it + 1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)

            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data.pt'))
            print("img and data saved!")

            if not args.test:
                best_acc_e, last_acc_e = synset.test(args, val_loader, logger)
                if best_acc_e <= last_acc_e:
                    torch.save(
                        [synset.data.detach().cpu(), synset.targets.cpu()],
                        os.path.join(args.save_dir, f'data_best.pt'))
                    print("best img and data updated!")

        if (it%5) +1 >= args.warmup_epoch:
            model.eval()
            for batch_idx_, (img_real, lab_real) in enumerate(eval_loader):
                img_real, lab_real = img_real.to(device), lab_real.to(device)
            
                with torch.no_grad():
                    real_logit = model(img_real)  # Forward pass
                    syn_cls_loss = F.cross_entropy(real_logit, lab_real, reduction='none')  # Loss for each data point in the batch

                # Store the loss per data point in the loss_list
                batch_size = img_real.size(0)
                for k in range(batch_size):
                    data_index = batch_idx_ * args.batch_real + k  # Ensure batch index is correctly multiplied by batch_train size
                    # Ensure we don't go out of bounds when processing the final smaller batch
                    if data_index < len(loss_list):
                        loss_list[data_index] += (syn_cls_loss[k].item())
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

            print(f"Selected: {len(selected_indices)} images, Unselected: {len(unselected_indices)} images")

            # relabeled_indices 초기화
            relabeled_indices = []
            # dst_train의 이미지와 라벨을 동시에 추출하여 각각 리스트로 저장
            dst_train_images, predicted_labels = zip(*[(data, label) for data, label in trainset])

            # 리스트 형태로 변환
            dst_train_images = list(dst_train_images)
            predicted_labels = list(predicted_labels)

            confidency_list = []
            relabeled_labels = []

            batch_size = args.batch_real

            batch_images = []
            batch_indices = []

            for idx in unselected_indices:
                image, _ = trainset[idx]
                batch_images.append(image)
                batch_indices.append(idx)

                # 배치 크기에 도달하면 처리
                if len(batch_images) == batch_size:
                    # 배치를 텐서로 변환 및 GPU로 이동
                    batch_images_tensor = torch.stack(batch_images).to(device)

                    # 모델에 배치 입력
                    with torch.no_grad():
                        outputs = model(batch_images_tensor)  # (batch_size, num_classes)

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
                batch_images_tensor = torch.stack(batch_images).to(device)
                with torch.no_grad():
                    outputs = model(batch_images_tensor)

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
            noisy_labels = np.array([trainset[i][1] for i in range(len(trainset))])  # Noisy CIFAR10N labels

            # clean_dataset과 noisy_dataset을 직접 생성
            clean_images = []
            clean_labels = []
            for idx in clean_indices:
                clean_images.append(dst_train_images[idx])
                clean_labels.append(predicted_labels[idx])

            noisy_images = []
            noisy_labels = []
            for idx in noisy_indices:
                noisy_image, noisy_label = trainset[idx]  # dst_train에서 원래 라벨과 이미지를 가져옴
                noisy_images.append(noisy_image)
                noisy_labels.append(noisy_label)

            # clean_dataset과 noisy_dataset 인스턴스 생성
            golden_dataset = CustomDataset(clean_images, clean_labels)
            noisy_dataset = CustomDataset(noisy_images, noisy_labels)

            print(f"Clean dataset: {len(clean_indices)}, Noisy dataset: {len(noisy_indices)}")

            # 데이터 로더 설정 (배치 크기는 필요에 따라 조정)
            loader_real = ClassMemDataLoader(golden_dataset, batch_size=args.batch_real, drop_last=True)
            loader_noisy = ClassMemDataLoader(noisy_dataset, batch_size=args.batch_real, drop_last=True)



if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    assert args.ipc > 0

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.save_dir = args.save_dir + '/RDC'

    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for run_id in range(2):
        logger("-" * 60)
        logger(f"Experiment {run_id + 1}:\n")
        condense(args, logger)