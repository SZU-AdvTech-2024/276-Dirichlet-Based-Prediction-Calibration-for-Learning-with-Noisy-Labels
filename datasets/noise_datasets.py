# -*- coding: utf-8 -*-
# @Author : Jack (thanks for Cheng Tan. Most of the codes afapted from his codebase "Co-training-based_noisy-label-learning-master")
# @Email  : liyifan20g@ict.ac.cn
# @File   : noise_datasets.py
import os
import random
import numpy as np
import torchvision
from PIL import Image
import torch
from torchnet.meter import AUCMeter
from .cifar import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .utils import noisify
from .randaugment import TransformFixMatch_CIFAR10, TransformFixMatch_CIFAR100, Transform_2strong_CIFAR100, TransformGJS, Transform2Weak_CIFAR10, Transform2Weak_CIFAR100
import json
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')


class NoiseDataset(torchvision.datasets.VisionDataset):

    def __init__(
        self,
        noise_type: str = 'none',
        asym_trans: dict = None,
        percent: float = 0.0,
        noise_file = ''
    ) -> None:

        assert percent <= 1.0 and percent >= 0.0
        assert noise_type in ['sym', 'asym', 'ins', 'none']

        self.percent = percent
        self.noise_type = noise_type
        self.asym_trans = asym_trans

        # dataset info
        self.min_target = min(self.targets)
        self.max_target = max(self.targets)
        self.num_classes = len(np.unique(self.targets))
        assert self.num_classes == self.max_target - self.min_target + 1
        self.num_samples = len(self.targets)

        self.trainlabel = self.targets.copy()


        if self.noise_type == 'sym':
            if os.path.exists(noise_file) and noise_file != '':
                self.noiselabel = json.load(open(noise_file, "r"))
                self.targets = self.noiselabel.copy()
            else:
                self.symmetric_noise() #

                self.noiselabel = self.targets.copy()
                self.noiselabel1 = [int(x) for x in self.targets.copy()]
                print("save noisy labels to %s ..."%noise_file)
                with open(noise_file, 'w') as f:
                    json.dump(self.noiselabel1, f)
            print('symmetric_noise loaded')
        elif self.noise_type == 'asym':
            if os.path.exists(noise_file) and noise_file != '':
                self.noiselabel = json.load(open(noise_file, "r"))
                self.targets = self.noiselabel.copy()
            else:
                self.asymmetric_noise() #
                self.noiselabel = self.targets.copy()
                self.noiselabel1 = [int(x) for x in self.targets.copy()]
                print("save noisy labels to %s ..." % noise_file)
                with open(noise_file, 'w') as f:
                    json.dump(self.noiselabel1, f)
            print('asymmetric_noise loaded')
        elif self.noise_type == 'ins':
            if os.path.exists(noise_file) and noise_file != '':
                self.noiselabel = json.load(open(noise_file, "r"))
                self.targets = self.noiselabel.copy()
            else:
                self.instance_noise(tau=self.percent)
                self.noiselabel = self.targets.copy()
                self.noiselabel1 = [int(x) for x in self.targets.copy()]
                print("save noisy labels to %s ..." % noise_file)
                with open(noise_file, 'w') as f:
                    json.dump(self.noiselabel1, f)
            print('instance_noise loaded')

    def symmetric_noise(self):
        type = 1

        if type == 1:
            indices = np.random.permutation(len(self.data))
            for i, idx in enumerate(indices):
                if i < self.percent * len(self.data):
                    self.targets[idx] = np.random.randint(
                        low=self.min_target,
                        high=self.max_target + 1,
                        dtype=np.int32)
            print("sys noise rate is %.4f" % self.percent)
        else:
            random_state = 0
            if self.num_classes == 10:
                dataset = 'cifar10'
            elif self.num_classes == 100:
                dataset = 'cifar100'
            train_noisy_labels, actual_noise_rate = noisify(
                dataset=dataset,
                train_labels=np.array(self.targets).reshape([50000, 1]),
                noise_type=self.noise_type,
                noise_rate=self.percent,
                random_state=random_state,
                nb_classes=self.num_classes)
            # targets 为noise_label
            self.targets = [int(label) for label in train_noisy_labels]
            print("Actual noise rate is %.4f" % actual_noise_rate)

    def asymmetric_noise(self):
        type = 1
        if type == 1:
            target_copy = self.targets.copy()
            if self.asym_trans == None:
                indices = np.arange(self.num_samples)
                np.random.shuffle(indices)
                idx = indices[:int(self.percent * self.num_samples)]
                target_copy = np.array(target_copy)
                target_copy[idx] = (target_copy[idx] + 1) % (
                    self.max_target + 1) + self.min_target
                self.targets = target_copy
            else:
                for i in self.asym_trans.keys():
                    indices = list(np.where(np.array(target_copy) == i)[0])
                    np.random.shuffle(indices)
                    for j, idx in enumerate(indices):
                        if j <= self.percent * len(indices):
                            self.targets[idx] = (self.asym_trans[i] if i
                                                 in self.asym_trans.keys() else
                                                 i)
            del target_copy
        else:
            random_state = 0
            # import ipdb; ipdb.set_trace()
            if self.num_classes == 10:
                dataset = 'cifar10'
            elif self.num_classes == 100:
                dataset = 'cifar100'
                self.noise_type = 'pairflip'
            train_noisy_labels, actual_noise_rate = noisify(
                dataset=dataset,
                train_labels=np.array(self.targets).reshape([50000, 1]),
                noise_type=self.noise_type,
                noise_rate=self.percent,
                random_state=random_state,
                nb_classes=self.num_classes)
            self.targets = [int(label) for label in train_noisy_labels]
            print("Actual noise rate is %.4f" % actual_noise_rate)

    def instance_noise(
        self,
        tau: float = 0.2,
        std: float = 0.1,
        feature_size: int = 3 * 32 * 32,
        # seed: int = 1
    ):
        '''
        Thanks the code from https://github.com/SML-Group/Label-Noise-Learning wrote by SML-Group.
        LabNoise referred much about the generation of instance-dependent label noise from this repo.
        '''
        from scipy import stats
        from math import inf
        import torch.nn.functional as F

        # np.random.seed(int(seed))
        # torch.manual_seed(int(seed))
        # torch.cuda.manual_seed(int(seed))

        # common-used parameters
        num_samples = self.num_samples
        num_classes = self.num_classes

        P = []
        # sample instance flip rates q from the truncated normal distribution N(\tau, {0.1}^2, [0, 1])
        flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std,
                                            loc=tau,
                                            scale=std)
        '''
        The standard form of this distribution is a standard normal truncated to the range [a, b]
        notice that a and b are defined over the domain of the standard normal. 
        To convert clip values for a specific mean and standard deviation, use:

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        truncnorm takes  and  as shape parameters.

        so the above `flip_distribution' give a truncated standard normal distribution with mean = `tau`,
        range = [0, 1], std = `std`
        '''
        # import ipdb; ipdb.set_trace()
        # how many random variates you need to get
        q = flip_distribution.rvs(num_samples)
        # sample W \in \mathcal{R}^{S \times K} from the standard normal distribution N(0, 1^2)
        W = torch.tensor(
            np.random.randn(num_classes, feature_size,
                            num_classes)).float().to(
                                device)  #K*dim*K, dim=3072
        for i in range(num_samples):
            x, y = self.transform(Image.fromarray(self.data[i])), torch.tensor(
                self.targets[i])
            x = x.to(device)
            # step (4). generate instance-dependent flip rates
            # 1 x feature_size  *  feature_size x 10 = 1 x 10, p is a 1 x 10 vector
            p = x.reshape(1, -1).mm(W[y]).squeeze(0)  #classes
            # step (5). control the diagonal entry of the instance-dependent transition matrix
            # As exp^{-inf} = 0, p_{y} will be 0 after softmax function.
            p[y] = -inf
            # step (6). make the sum of the off-diagonal entries of the y_i-th row to be q_i
            p = q[i] * F.softmax(p, dim=0)
            p[y] += 1 - q[i]
            P.append(p)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(self.min_target, self.max_target + 1)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(num_samples)]

        print('noise rate = ', (new_label != np.array(self.targets)).mean())
        self.targets = new_label


class NoiseCIFAR10(CIFAR10, NoiseDataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download=False,
        mode: str = None,
        noise_type: str = 'none',
        percent: float = 0.0,
        pred=[],
        prob=[],
        noise_file='',
        log=''
    ) -> None:

        self.transform_train_weak = transforms.Compose([
            #transforms.RandomResizedCrop(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_train_strong = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        if mode == 'train_index':
            # self.transform_train = Transform2Weak_CIFAR10((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.transform_train = TransformFixMatch_CIFAR10(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif mode == 'train_index_2strong':
            self.transform_train = TransformFixMatch_CIFAR10(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        # elif mode == 'train_dpc':
        #     self.transform_train = transforms.Compose([
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #     ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])


        self.root = root
        self.mode = mode
        self.transform = self.transform_test
        self.target_transform = None
        asym_trans = {
            9: 1,  # truck ->  automobile
            2: 0,  # bird  ->  airplane
            3: 5,  # cat   ->  dog
            5: 3,  # dog   ->  cat
            4: 7,  # deer  ->  horse
        }

        CIFAR10.__init__(self,
                         root=root,
                         train=train,
                         transform=transform,
                         download=download)
        NoiseDataset.__init__(self,
                              noise_type=noise_type,
                              asym_trans=asym_trans,
                              percent=percent,
                              noise_file=noise_file)

        self.train_data = self.data.copy()
        # self.noise_label = self.targets.copy()
        if self.mode == 'labeled':
            pred_idx = pred.nonzero()[0]
            self.probability = [prob[i] for i in pred_idx]
            clean = (np.array(self.noiselabel) == np.array(self.trainlabel))
            auc_meter = AUCMeter()
            auc_meter.reset()
            auc_meter.add(prob, clean)
            auc, _, _ = auc_meter.value()
            if log != '':
                log.write('Numer of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                log.flush()
            print('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
        elif self.mode == 'meta':
            idx = list(range(50000))
            random.shuffle(idx)
            meta_id = idx[0:1000]
            self.train_data = [self.train_data[id] for id in meta_id]
            self.noise_label = [self.trainlabel[id] for id in meta_id]
            print(len(self.train_data), len(self.noise_label))

            data_list_val = {}
            for j in range(10):
                data_list_val[j] = [i for i, label in enumerate(self.noise_label) if label == j]
                print("ratio class", j, ":", len(data_list_val[j]) / 1000 * 100)

        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            self.probability = [prob[i] for i in pred_idx]

        if self.mode == "unlabeled" or self.mode == 'labeled':
            self.train_data = self.data[pred_idx]
            self.noise_label = [self.noiselabel[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.noise_label)))


    def __getitem__(self, index):
        if self.mode == 'train_single':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            img = self.transform_train_weak(image)
            return img, target
        elif self.mode == 'train':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)
            return raw, img1, img2, target
        elif self.mode == 'train_index':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            img = self.transform_train(image)
            return img, target, index
        elif self.mode == 'test':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            img = self.transform_test(image)
            return img, target

        elif self.mode == 'warmup':
            image, target = self.data[index], self.noiselabel[index]
            image = Image.fromarray(image)
            img = self.transform_train_weak(image)
            return img, target, index
        elif self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform_train_weak(img)
            img2 = self.transform_train_weak(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform_train_weak(img)
            img2 = self.transform_train_weak(img)
            prob = self.probability[index]
            return img1, img2, prob
        elif self.mode == 'meta':
            image, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(image)
            img = self.transform(img)
            return img, target
        elif self.mode == 'eval_train':
            image, target = self.train_data[index], self.noiselabel[index]
            image = Image.fromarray(image)
            img = self.transform_train_weak(image)
            return img, target, index
    def __len__(self):
        if self.mode != 'test':
            # print('长度为：', len(self.train_data))
            return len(self.train_data)

        else:
            return len(self.data)


class NoiseCIFAR100(CIFAR100, NoiseDataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        download=True,
        mode: str = None,
        noise_type: str = 'none',
        percent: float = 0.0,
        pred=[],
        prob=[],
        noise_file='',
        log = ''
    ) -> None:

        self.transform_train_weak = transforms.Compose([
            #transforms.RandomResizedCrop(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441),
                                 (0.267, 0.256, 0.276)),
        ])
        self.transform_train_strong = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441),
                                 (0.267, 0.256, 0.276)),
        ])
        # import ipdb; ipdb.set_trace()
        if mode == 'train_index':
            self.transform_train = TransformFixMatch_CIFAR100(
                (0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        if mode == 'train_index_2strong':
            self.transform_train = Transform_2strong_CIFAR100(
                (0.507, 0.487, 0.441), (0.267, 0.256, 0.276))

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441),
                                 (0.267, 0.256, 0.276)),
        ])

        self.root = root
        self.mode = mode
        self.transform = self.transform_test
        self.target_transform = None

        CIFAR100.__init__(self,
                          root=root,
                          train=train,
                          transform=transform,
                          download=download)
        NoiseDataset.__init__(self,
                              noise_type=noise_type,
                              asym_trans=None,
                              percent=percent,
                              noise_file=noise_file)

        self.train_data = self.data.copy()
        # self.noise_label = self.targets.copy()
        if self.mode == 'labeled':
            pred_idx = pred.nonzero()[0]
            self.probability = [prob[i] for i in pred_idx]
            clean = (np.array(self.noiselabel) == np.array(self.trainlabel))
            auc_meter = AUCMeter()
            auc_meter.reset()
            auc_meter.add(prob, clean)
            auc, _, _ = auc_meter.value()
            log.write('Numer of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
            log.flush()
            print('Numer of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            self.probability = [prob[i] for i in pred_idx]
        elif self.mode == 'meta':
            idx = list(range(50000))
            random.shuffle(idx)
            meta_id = idx[0:1000]
            self.train_data = [self.train_data[id] for id in meta_id]
            self.noise_label = [self.trainlabel[id] for id in meta_id]
            print(len(self.train_data), len(self.noise_label))

            data_list_val = {}
            for j in range(10):
                data_list_val[j] = [i for i, label in enumerate(self.noise_label) if label == j]
                print("ratio class", j, ":", len(data_list_val[j]) / 1000 * 100)
        if self.mode == "unlabeled" or self.mode == 'labeled':
            self.train_data = self.data[pred_idx]
            self.noise_label = [self.noiselabel[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == 'train_single':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            img = self.transform_train_weak(image)
            return img, target
        elif self.mode == 'train':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)
            return raw, img1, img2, target
        elif self.mode == 'train_index':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            img = self.transform_train(image)
            return img, target, index
        elif self.mode == 'test':
            image, target = self.data[index], self.targets[index]
            image = Image.fromarray(image)
            img = self.transform_test(image)
            return img, target

        elif self.mode == 'warmup':
            image, target = self.data[index], self.noiselabel[index]
            image = Image.fromarray(image)
            img = self.transform_train_weak(image)
            return img, target, index
        elif self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform_train_weak(img)
            img2 = self.transform_train_weak(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform_train_weak(img)
            img2 = self.transform_train_weak(img)
            prob = self.probability[index]
            return img1, img2, prob
        elif self.mode == 'meta':
            image, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(image)
            img = self.transform(img)
            return img, target
        elif self.mode == 'eval_train':
            image, target = self.train_data[index], self.noiselabel[index]
            image = Image.fromarray(image)
            img = self.transform_train_weak(image)
            return img, target, index

    def __len__(self):
        if self.mode != 'test':
            # print('长度为：', len(self.train_data))
            return len(self.train_data)

        else:
            return len(self.data)


class cifar_dataloader():

    def __init__(self, cifar_type, root, batch_size, num_workers, noise_type,
                 percent, noise_file='', log = ''):
        self.cifar_type = cifar_type
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_type = noise_type
        self.percent = percent
        self.noise_file = noise_file
        self.log = log
    def run(self, mode, pred=[], prob=[], noise_file='',log=''):
        if mode == 'train_single':
            if self.cifar_type == 'cifar-10':
                train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'cifar-100':
                train_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            else:
                raise "incorrect cifar dataset name -> (`cifar-10`, `cifar-100`)"
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True,
                                      num_workers=self.num_workers)
            return train_loader
        elif mode == 'train_index' or mode == 'train_index_2strong':
            if self.cifar_type == 'cifar-10':
                train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'cifar-100':
                train_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            else:
                raise "incorrect cifar dataset name -> (`cifar-10`, `cifar-100`)"
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False,
                                      num_workers=self.num_workers)
            return train_loader
        elif mode == 'tripartite':
            if self.cifar_type == 'cifar-10':
                train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'cifar-100':
                train_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            else:
                raise "incorrect cifar dataset name -> (`cifar-10`, `cifar-100`)"
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False,
                                      num_workers=self.num_workers)
            return train_loader
        elif mode == 'train':
            if self.cifar_type == 'cifar-10':
                train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'cifar-100':
                train_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode=mode)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False,
                                      num_workers=self.num_workers)
            return train_loader
        elif mode == 'test':
            if self.cifar_type == 'cifar-10':
                test_dataset = NoiseCIFAR10(self.root, train=False, transform=None, noise_type='none',
                                                percent=0.0, mode=mode)
            elif self.cifar_type == 'cifar-100':
                test_dataset = NoiseCIFAR100(self.root, train=False, transform=None, noise_type='none',
                                                percent=0.0, mode=mode)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=True,
                                     num_workers=self.num_workers)
            return test_loader
        # dpc
        elif mode == 'warmup':
            if self.cifar_type == 'cifar-10':
                train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                             percent=self.percent, mode='warmup', noise_file=noise_file)
            elif self.cifar_type == 'cifar-100':
                train_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                              percent=self.percent, mode='warmup', noise_file=noise_file)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size*2,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=self.num_workers)
            return train_loader

        elif mode == 'dpc_train':
            if self.cifar_type == 'cifar-10':
                labeled_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                             percent=self.percent, mode='labeled',pred=pred,prob=prob,noise_file=noise_file, log=log)
                unlabeled_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                             percent=self.percent, mode='unlabeled',pred=pred,prob=prob,noise_file=noise_file)
            elif self.cifar_type == 'cifar-100':
                labeled_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                percent=self.percent, mode='labeled',pred=pred,prob=prob,noise_file=noise_file, log=log)
                unlabeled_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                                  percent=self.percent, mode='unlabeled',pred=pred,prob=prob,noise_file=noise_file)

            labeled_trainloader = DataLoader(dataset=labeled_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True,
                                      num_workers=self.num_workers)
            unlabeled_trainloader = DataLoader(dataset=unlabeled_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True,
                                      num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader
        elif mode == 'eval_train':
            if self.cifar_type == 'cifar-10':
                eval_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                             percent=self.percent, mode='eval_train', noise_file=noise_file)
            elif self.cifar_type == 'cifar-100':
                eval_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type,
                                             percent=self.percent, mode='eval_train', noise_file=noise_file)
            eval_loader = DataLoader(dataset=eval_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=self.num_workers)
            return eval_loader

        elif mode == 'meta':
            if self.cifar_type == 'cifar-10':
                meta_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type='none',
                                             percent=self.percent, mode='meta', noise_file=noise_file)
            elif self.cifar_type == 'cifar-100':
                meta_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type='none',
                                             percent=self.percent, mode='meta', noise_file=noise_file)
            meta_loader = DataLoader(dataset=meta_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers= self.num_workers
                                     )
            return meta_loader