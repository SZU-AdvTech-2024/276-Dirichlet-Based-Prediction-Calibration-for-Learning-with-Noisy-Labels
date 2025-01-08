import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets import cifar_dataloader
import numpy as np
from utils import get_model
from losses import EDL_Loss
from torchvision.models import resnet50, vgg19_bn
from tqdm import tqdm
from torch.distributions.beta import Beta
import pickle
import os
from sklearn.mixture import GaussianMixture
from models.model import ResNet18
import higher



class DPC_L2B:
    def __init__(self,
                 config: dict = None,
                 input_channel: int = 3,
                 num_classes: int = 10,
                 ):
        self.num_classes = num_classes
        self.batch_size = config['batch_size']
        device = torch.device('cuda:%s' % config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.noise_mode = config['noise_type']
        self.epochs = config['epochs']
        self.dataset = config['dataset']
        self.noise_type = config['noise_type'] + '_' + str(config['percent'])
        self.prob1 = []
        self.prob2 = []
        self.pred1 = []
        self.pred2 = []
        self.acc = 0
        self.temp_epoch = 0
        self.lambda_u = config['lambda_u']
        self.p_threshold = config['p_threshold']
        self.lr = config['lr']
        self.T = config['T']
        self.alpha = config['alpha']
        self.r = config['percent']
        self.warmup_epoch = config['warmup_epoch']
        self.need_clean = config['need_clean']
        self.single_meta = config['single_meta']

        # Backbones for different datasets
        if 'cifar' in self.dataset or 'tiny_imagenet' in self.dataset:
            if 'ins' in config['noise_type']:
                config['model1_type'] = 'resnet18'
                config['model2_type'] = 'resnet18'
            self.model1 = self.create_model()
            self.model2 = self.create_model()
        # Optimizers
        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # Loss function definition
        self.EDL_loss = EDL_Loss(num_classes=self.num_classes)
        self.noise_file = '%s/%s_%s.json' % (config['root'], self.noise_type,self.dataset)


    def linear_rampup(self, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return self.lambda_u * float(current)

    def SemiLoss(self, outputs_x, l, targets_x1, targets_x2, outputs_u, targets_u, epoch, warm_up):
        probs_u = self.conv_p(outputs_u)

        Lx11, Lx12 = self.EDL_loss(outputs_x, targets_x1)
        Lx21, Lx22 = self.EDL_loss(outputs_x, targets_x2)

        Lx = (l * Lx11 + (1 - l) * Lx21).mean() + (l * Lx12 + (1 - l) * Lx22).mean()
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.linear_rampup(epoch, warm_up)

    #替换softmax
    def conv_p(self, logits):
        # 10/n_class
        alpha_t = torch.exp(logits) + 10. / self.num_classes
        total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)
        expected_p = alpha_t / total_alpha_t
        return expected_p

    def train_one(self, model1, model2, optimizer, epoch, labeled_trainloader,unlabeled_trainloader, meta_loader=None):
        model1.train()
        model2.eval()
        if meta_loader is not None:
            meta_iter = iter(meta_loader)
        num_iter = (len(labeled_trainloader.dataset) // self.batch_size)

        for batch_idx in range(num_iter):
            try:
                inputs_x, inputs_x2, labels_x, w_x = next(labeled_train_iter)
            except:
                # print('try1')
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, inputs_x2, labels_x, w_x = next(labeled_train_iter)
            try:
                inputs_u, inputs_u2, w_u = next(unlabeled_train_iter)
            except:
                # print('try2')
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, w_u = next(unlabeled_train_iter)
            if meta_loader is not None:
                try:
                    val_data, val_labels = next(meta_iter)
                except:
                    meta_iter = iter(meta_loader)
                    val_data, val_labels = next(meta_iter)

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, self.num_classes).scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)
            w_u = w_u.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, w_u = inputs_u.cuda(), inputs_u2.cuda(), w_u.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11,_ = model1(inputs_u)
                outputs_u12,_ = model1(inputs_u2)
                outputs_u21,_ = model2(inputs_u)
                outputs_u22,_ = model2(inputs_u2)

                pu = (self.conv_p(outputs_u11) + self.conv_p(outputs_u12) + self.conv_p(outputs_u21) + self.conv_p(outputs_u22)) / 4
                ptu = pu ** (1 / self.T)  # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x,_ = model1(inputs_x)
                outputs_x2,_ = model2(inputs_x2)

                px = (self.conv_p(outputs_x) + self.conv_p(outputs_x2)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / self.T)  # temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

            # mixmatch
            l = np.random.beta(self.alpha, self.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits, logits2 = model1(mixed_input)
            logits_x = logits[:batch_size * 2]
            logits_u = logits2[batch_size * 2:]

            Lx, Lu, lamb = self.SemiLoss(logits_x, l, target_a[:batch_size * 2].argmax(1),
                                          target_b[:batch_size * 2].argmax(1), logits_u, mixed_target[batch_size * 2:],
                                          epoch + batch_idx / num_iter, self.warmup_epoch)

            # regularization
            prior = torch.ones(self.num_classes) / self.num_classes
            prior = prior.cuda()
            pred_mean = self.conv_p(logits).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + lamb * Lu + penalty
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if meta_loader is not None:
                model1 = self.l2b(model1, optimizer, input_a, target_a, val_data, val_labels)


            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                             % (
                                 self.dataset, self.r, self.noise_mode, epoch, self.epochs, batch_idx + 1, num_iter,
                                 Lx.item(), Lu.item()))
            sys.stdout.flush()

    def train(self,epoch,loader,meta_loader):
        print('Training ...')
        self.temp_epoch = epoch
        lr = self.lr
        if epoch >= 150:
            lr /= 10
        for param_group in self.optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer2.param_groups:
            param_group['lr'] = lr

        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train',noise_file=self.noise_file)

        if epoch < self.warmup_epoch:
            warmup_trainloader = loader.run('warmup',noise_file=self.noise_file)
            print('Warmup Net1')
            self.warmup(epoch, self.model1, self.optimizer1, warmup_trainloader)
            print('\nWarmup Net2')
            self.warmup(epoch, self.model2, self.optimizer2, warmup_trainloader)
        else:
            print('\nEval_train...')
            self.prob1 = self.eval_train(self.model1, eval_loader)
            self.prob2 = self.eval_train(self.model2, eval_loader)

            self.pred1 = (self.prob1 > self.p_threshold)
            self.pred2 = (self.prob2 > self.p_threshold)

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('dpc_train', pred=self.pred2, prob=self.prob2, noise_file = self.noise_file)  # co-divide

            if self.single_meta < 1:
                second_meta = None
                first_meta = None
            elif self.single_meta == 1:
                second_meta = None
                if self.need_clean:
                    first_meta = meta_loader
                else:
                    first_meta = labeled_trainloader

            self.train_one(self.model1, self.model2, self.optimizer1, epoch, labeled_trainloader, unlabeled_trainloader, meta_loader=first_meta)

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('dpc_train', pred=self.pred1, prob=self.prob1, noise_file = self.noise_file)  # co-divide
            self.train_one(self.model2, self.model1, self.optimizer2, epoch, labeled_trainloader, unlabeled_trainloader, meta_loader=second_meta)

        # self.test(epoch, self.model1, self.model2, test_loader)

    def l2b(self, net, optimizer, image, labels, val_data, val_labels):
        criterion = nn.CrossEntropyLoss(reduction="none").cuda()
        with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_optimizer):
            for s in range(1):
                y_f_hat,_ = meta_net(image)
                # print(y_f_hat)
                _, pesudo_labels = y_f_hat.max(1)
                cost1 = criterion(y_f_hat, labels)
                cost2 = criterion(y_f_hat, pesudo_labels)

                cost = torch.cat((cost1.unsqueeze(0), cost2.unsqueeze(0)), dim=0)

                eps = torch.zeros(cost.size()).cuda()
                eps = eps.requires_grad_()

                l_f_meta = torch.sum(cost * eps)

                meta_optimizer.step(l_f_meta)
            # if self.need_clean:
            #     val_data, val_labels = next(iter(metaloader))
            # else:
            #     val_data, _, val_labels, _ = next(iter(metaloader))
            val_data = val_data.cuda()
            val_labels = val_labels.cuda()

            y_g_hat,_ = meta_net(val_data)
            l_g_meta = torch.mean(criterion(y_g_hat, val_labels))

            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0].detach()

            w_tilde = -grad_eps
            w_tilde = w_tilde.view(-1)

            w_tilde = torch.clamp(w_tilde, min=0)

            norm_c = torch.sum(w_tilde) + 1e-10

            if norm_c != 0:
                weight = w_tilde / norm_c
            else:
                weight = w_tilde

            weight = weight.view(2, -1)

        y_f_hat,_ = net(image)
        _, pesudo_labels = y_f_hat.max(1)

        cost1 = criterion(y_f_hat, labels)
        cost2 = criterion(y_f_hat, pesudo_labels)

        cost = torch.cat((cost1.unsqueeze(0), cost2.unsqueeze(0)), dim=0)

        l_f = torch.sum(cost * weight)

        optimizer.zero_grad()
        l_f.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.80, norm_type=2)
        optimizer.step()

        return net


    def warmup(self,epoch,model,optimizer,dataloader):
        model.train()
        num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1

        # pbar = tqdm(dataloader)
        # for (inputs, labels, indexes) in pbar:
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            # 将列表转换为Tensor
            inputs_tensor = inputs.clone().detach()
            labels_tensor = labels.clone().detach()     #torch.tensor(labels)

            # 创建Variable或直接操作Tensor
            inputs = inputs_tensor.cuda()
            labels = labels_tensor.cuda()
            #inputs, labels = Variable(inputs).to(self.device, non_blocking=True),Variable(labels).to(self.device, non_blocking=True)
            optimizer.zero_grad()
            outputs,_ = model(inputs)
            loss1, loss2 = self.EDL_loss(outputs, labels)
            loss = loss1.mean() + loss2.mean()

            if self.noise_mode == 'asym':
                penalty = self.NegEntropy(outputs)
                L = loss + penalty
            elif self.noise_mode == 'sym' or self.noise_mode == 'ins':
                L = loss
            L.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                             % (self.dataset, self.r, self.noise_mode, epoch, self.epochs, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()


    def NegEntropy(self, outputs):
        probs = self.conv_p(outputs)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

    def test(self, test_loader):
        print('\nDPC Evaluating ...')
        self.model1.eval()
        self.model2.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, targets) in (test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1, _ = self.model1(inputs)
                outputs2, _ = self.model2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        self.acc = acc
        self.save_results()
        return acc

    def eval_train(self, model, eval_loader):
        model.eval()
        margins = torch.zeros(50000)
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = model(inputs)

                for b in range(inputs.size(0)):
                    evidence_pos = outputs[b, targets[b]]
                    copy_outputs = outputs[b].clone()
                    copy_outputs[targets[b]] = -1e5
                    evidence_neg = copy_outputs.max()
                    margins[index[b]] = evidence_pos - evidence_neg

        margins = (margins - margins.min()) / (margins.max() - margins.min())

        input_margin = margins.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_margin)
        prob = gmm.predict_proba(input_margin)
        prob1 = prob[:, gmm.means_.argmax()]

        return prob1

    def evaluate(self, test_loader):
        print('\nDPC Evaluating ...')
        self.model1.eval()
        self.model2.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, targets) in (test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1,_ = self.model1(inputs)
                outputs2,_ = self.model2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        self.acc = acc
        self.save_results()
        return acc

    def save_results(self, name='dpc+l2b'):
        save_root = 'result_root/%s/'%self.dataset
        filename = save_root + self.noise_type + '_save_' + name + '.txt'

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        if 'cifar' in self.dataset:
            results = {'epoch:': self.temp_epoch, 'acc:': self.acc}
            # 检查文件是否存在
        if os.path.exists(filename):
            # 如果文件存在，读取已有的数据
            with open(filename, 'r') as file:
                data = json.load(file)
                # 追加新的数据
                data['results'].append(results)
        else:
            # 如果文件不存在，创建新文件并保存新数据
            data = {'results': [results]}

        # 保存更新后的数据
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def create_model(self):
        model = ResNet18(num_classes=self.num_classes)
        model = model.cuda()
        return model