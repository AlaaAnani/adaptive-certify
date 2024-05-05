import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
from tqdm import tqdm
from torch.nn import functional as F
from utils.utils import get_confusion_matrix
from twosample import binom_test
import json 
from itertools import permutations
import pickle
from utils.utils import get_confusion_matrix, find_boundaries_torch
from collections import defaultdict
        
class Certifier:
    def __init__(self):
        self.args = self.parse_args()
        self.model = self.init_model()
        self.init_dataset()
        #self.experiment_h_distribution(self.model)
        #self.exp_fluctuations(self.model)
        #self.experiment_table(self.model)
        #self.sigma_inference(self.model)
        self.exp_images(self.model)
        #self.experiment_find_best_threshold(self.model)
        
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Train segmentation network')
        
        parser.add_argument('--cfg',
                            #default='configs/cocostuff10k/cocostuff10k.yaml',
                            #default='configs/pascal_ctx/pascal_ctx.yaml',
                            default='configs/acdc/acdc.yaml',
                            #default='configs/cityscapes/cityscapes.yaml',

                            help='experiment configure file name',
                            type=str)
        parser.add_argument('opts',
                            help="Modify config options using the command-line",
                            default=None,
                            nargs=argparse.REMAINDER)

        args = parser.parse_args()
        update_config(config, args)
        self.config = config
        return args

    def init_model(self):
        # cudnn related setting
        # cudnn.benchmark = config.CUDNN.BENCHMARK
        # cudnn.deterministic = config.CUDNN.DETERMINISTIC
        # cudnn.enabled = config.CUDNN.ENABLED

        # build model
        if torch.__version__.startswith('1'):
            module = eval('models.'+config.MODEL.NAME)
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)

        model_state_file = config.TEST.MODEL_FILE
        
        print('=> loading model from {}'.format(model_state_file))
            
        pretrained_dict = torch.load(model_state_file)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        n_gpus = torch.cuda.device_count()
        gpus = [i for i in range(n_gpus)]
        print(f"Number of GPUs available: {n_gpus}")        
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        return model
    
    def init_dataset(self):
        # prepare data
        test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
        if config.DATASET.DATASET == 'cityscapes' or config.DATASET.DATASET == 'acdc':
            self.temperature = 1.335
        else:
            self.temperature = 1.0
        test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                            root=config.DATASET.ROOT,
                            list_path=config.DATASET.TEST_SET,
                            num_samples=config.TEST.NUM_SAMPLES,
                            num_classes=config.DATASET.NUM_CLASSES,
                            multi_scale=False,
                            flip=False,
                            normalize=False,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            base_size=config.TEST.BASE_SIZE,
                            crop_size=test_size,
                            downsample_rate=1)
        self.test_dataset = test_dataset
        
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True)
        self.test_loader = testloader
        # load the hierarchy
        self.hierarchy = json.load(open(os.path.join(config.DATASET.ROOT, config.DATASET.HIERARCHY), 'r'))
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        self.logdir = os.path.join(self.config.LOG_DIR, self.config.DATASET.DATASET)
        os.makedirs(self.logdir, exist_ok=True)
        
    def sample(self, model, image_np01, size, N, sigma, border_padding, 
               logits_only=False, do_tqdm=False, cuda_id=None):
        BS = self.config.TEST.BATCH_SIZE_PER_GPU
        print('Sampling @ batch size =', BS)
        out, out_logits = [], []
        remaining = N
        if do_tqdm: pbar = tqdm(total=N)
        with torch.no_grad():
            while (remaining) > 0:
                cnt = min(remaining, BS)
                pred = self.test_dataset.multi_scale_inference_noisybatch(config,
                            model, 
                            cnt, sigma,
                            image_np01,
                            normalize=True,
                            scales=self.config.TEST.SCALE_LIST, 
                            flip=self.config.TEST.FLIP_TEST,
                            unscaled=False,
                            cuda_id=cuda_id,
                            border_padding=border_padding,
                            size=size) # torch.Size([bs, num_classes, w, h])
                if (pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]):
                    pred = F.interpolate(pred, (size[-2], size[-1]), 
                                    mode='bilinear')
                if logits_only:
                    out_logits.append(pred.cpu().numpy())

                pred = pred.argmax(dim=1).cpu().numpy() # torch.Size([bs, w, h])
                out.append(pred)
                remaining -= cnt
                if do_tqdm: pbar.update(cnt)
        if do_tqdm: pbar.close()
        if logits_only:
            out_logits = np.concatenate(out_logits)/self.temperature
            return None, out_logits
        return np.concatenate(out), out_logits/self.temperature
    
    def fast_certify(self, samples_flattened, n0, n, tau=0.75, alpha=0.001, stats=None, non_ignore_idx=None):
        if not isinstance(samples_flattened, torch.Tensor):
            samples_flattened = torch.from_numpy(samples_flattened)
        modes, _ = torch.mode(samples_flattened[:n0], 0)
        counts_at_modes = (samples_flattened[n0:] == modes.unsqueeze(0)).sum(0)
        pvals_ = binom_test(np.array(counts_at_modes), np.array([n-n0]*len(samples_flattened[0])), np.array([tau]*len(samples_flattened[0])), alt='greater')
        abstain = pvals_ > alpha/len(samples_flattened[0])
        modes = modes.cpu().numpy()
        modes[np.array(abstain)] = self.config.DATASET.ABSTAIN_LABEL
        if stats is not None:
            d = {}
            d['fluctuations'] = samples_flattened[:, abstain & non_ignore_idx].cpu().numpy()
            return modes, d
        return modes
    
    def segcertify(self, model, image_np01, n0, n, sigma, tau, alpha, 
                   size, border_padding,
                   do_tqdm=False, cuda_id=0, samples_logits=None, stats=None, non_ignore_idx=None):
        if samples_logits is None:
            _, samples_logits = self.sample(model, image_np01, size, 
                                            border_padding=border_padding, 
                                            N=n+n0, sigma=sigma, 
                                            logits_only=True, 
                                            do_tqdm=do_tqdm, cuda_id=cuda_id)
        samples_flattened = torch.from_numpy(samples_logits.argmax(1).reshape(n+n0, -1))
        if stats:
            certified_pred, d = self.fast_certify(samples_flattened, n0, n, tau, alpha, stats, non_ignore_idx)
            certified_pred = certified_pred.reshape(size)
            return certified_pred, d


        certified_pred = self.fast_certify(samples_flattened, n0, n, tau, alpha, stats)
        certified_pred = certified_pred.reshape(size)
        return certified_pred
    
    def get_difference(self, posteriors):
        # N, num_classes, w, h = posteriors.shape
        if not isinstance(posteriors, torch.Tensor):
            posteriors = torch.tensor(posteriors)
        posteriors_m = torch.mean(posteriors, dim=0)
        posteriors_std = torch.std(posteriors, dim=0)
        sorted_posteriors, _ = torch.sort(posteriors_m, dim=0)
        diff = sorted_posteriors[-1, :, :] - sorted_posteriors[-2, :, :]
        posteriors_std = (posteriors_std - torch.min(posteriors_std))/(torch.max(posteriors_std)-torch.min(posteriors_std))
        posteriors_m_std = torch.mean(posteriors_std, dim=0) # the average standard deviation at each pixel (w, h)

        return diff, posteriors_m_std
    
    def adapt_to_hierarchy(self, samples_flattened, diff, f=None):
        if isinstance(samples_flattened, torch.Tensor):
            samples_flattened = samples_flattened.cpu().numpy()
        diff = diff.flatten()
        h_map = np.zeros_like(diff)
        for h_i, i in enumerate(reversed(range(len(f)))):
            h_map[diff <= f[i]] = h_i + 1
        samples_flattened_adaptive = samples_flattened.copy()
        hs = np.unique(h_map)
        for i in hs:
            # extend LUT to include the ignore label
            LUT_ls = self.hierarchy['lookup_tables'][f'{int(i)+1}']
            LUT = np.array(LUT_ls + [config.TRAIN.IGNORE_LABEL]*(config.TRAIN.IGNORE_LABEL + 1 - len(LUT_ls)))
            idx = h_map == i+1 
            samples_flattened_adaptive[:, idx] = LUT[samples_flattened[:, idx]]
        return samples_flattened_adaptive, np.array(h_map)

    def info_gain_adaptive(self, classes_certify, gt_adaptive_label, label, stats=False, boundary=True, h_map=None, images=False):
        confusion_matrix = get_confusion_matrix(
            gt_adaptive_label.reshape(label.size()),
            classes_certify.reshape(label.size()),
            label.size(),
            len(self.hierarchy['nodes']),
            config.TRAIN.IGNORE_LABEL,
            abstain=config.DATASET.ABSTAIN_LABEL)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        print('pixel_acc', pixel_acc, 'mean_acc', mean_acc, 'mean_IoU', mean_IoU)
        if boundary:
            boundary_map = find_boundaries_torch(label[0], 10)
            boundary_map = boundary_map.cpu().numpy().flatten()
            boundary_idx = boundary_map == 1
            non_boundary_idx = boundary_map != 1
        if isinstance(gt_adaptive_label, torch.Tensor):
            gt_adaptive_label = gt_adaptive_label.cpu().numpy()
        if isinstance(classes_certify, torch.Tensor):
            classes_certify = classes_certify.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        
        gt_adaptive_label = gt_adaptive_label.flatten()
        classes_certify = classes_certify.flatten()
        label = label.flatten()
        

        c_ig_map = np.zeros(classes_certify.shape)
        
        l = [0]*len(self.hierarchy['node_info_gain_lookup'].keys())
        for idx, v in self.hierarchy['node_info_gain_lookup'].items():
            l[int(idx)] = v
        LUT_infogain = np.array(l)

        non_ignore_idx = label != config.TRAIN.IGNORE_LABEL
        c_idx = classes_certify != config.DATASET.ABSTAIN_LABEL
        abstain_idx = classes_certify == config.DATASET.ABSTAIN_LABEL
        correct_pred = classes_certify == gt_adaptive_label
        wrong_pred = classes_certify != gt_adaptive_label
        filter_idx = non_ignore_idx & c_idx & correct_pred
        
        # cig sum
        c_ig = (np.log(config.DATASET.NUM_CLASSES) - np.log(LUT_infogain[classes_certify[filter_idx]])).sum()
        # cig map
        c_ig_map[filter_idx] = np.log(config.DATASET.NUM_CLASSES) - np.log(LUT_infogain[classes_certify[filter_idx]])
        
        # calculate cig per class
        class_ids = np.unique(label[non_ignore_idx])



        ig_per_class_dict = defaultdict(lambda: np.zeros(self.config.DATASET.NUM_CLASSES))
        for i in class_ids:
            ig_per_class_dict['cig_per_cls'][i] += c_ig_map[filter_idx & (label == i)].sum()
            ig_per_class_dict['num_pixels_per_cls'][i] += (non_ignore_idx & (label == i)).sum()
            ig_per_class_dict['abstain_per_cls'][i] += (non_ignore_idx & (label == i) & abstain_idx).sum()
            ig_per_class_dict['certified_per_cls'][i] += (non_ignore_idx & (label == i) & c_idx).sum()

            if boundary:
                for b, idx in [('boundary', boundary_idx), ('non_boundary', non_boundary_idx)]:
                    ig_per_class_dict[f'cig_per_cls_{b}'][i] += c_ig_map[filter_idx & (label == i) & idx].sum()
                    ig_per_class_dict[f'num_pixels_per_cls_{b}'][i] += (non_ignore_idx & (label == i) & idx).sum()
                    ig_per_class_dict[f'abstain_per_cls_{b}'][i] += (non_ignore_idx & (label == i) & abstain_idx & idx).sum()
                    ig_per_class_dict[f'certified_per_cls_{b}'][i] += (non_ignore_idx & (label == i) & c_idx & idx).sum()

            
        if stats:
            d = {}
            d['abstain_count'] = (classes_certify[non_ignore_idx] == config.DATASET.ABSTAIN_LABEL).sum()
            d['certified_count'] = (classes_certify[non_ignore_idx] != config.DATASET.ABSTAIN_LABEL).sum()
            d['num_pixels'] = non_ignore_idx.sum()
            d['ig_per_class_dict'] = dict(ig_per_class_dict)
            d['confusion_matrix'] = confusion_matrix
            if h_map is not None:
                h_map = np.array(h_map).flatten()
                ls = np.unique(h_map)
                h_distribution_dict = {}
                for l in ls:
                    if l not in h_distribution_dict:
                        h_distribution_dict[l] = {}
                    print(l, (non_ignore_idx & (h_map == l)).sum()/non_ignore_idx.sum())
                    h_distribution_dict[l]['abstain_per_level'] = (non_ignore_idx & abstain_idx & (h_map == l)).sum()
                    h_distribution_dict[l]['certified_per_level'] = (non_ignore_idx & c_idx & (h_map == l)).sum()

                    h_distribution_dict[l]['cig_per_level'] = c_ig_map[filter_idx & (h_map == l)].sum()
                    h_distribution_dict[l]['correctly_certified_per_level'] = (filter_idx & (h_map == l)).sum()
                    h_distribution_dict[l]['wrongly_certified_per_level'] = (non_ignore_idx & wrong_pred & c_idx & (h_map == l)).sum()
                    h_distribution_dict[l]['num_pixels_per_level'] = (non_ignore_idx & (h_map == l)).sum()
                    
                    h_distribution_dict[l]['abstain_per_level_boundary'] = (boundary_idx & non_ignore_idx & abstain_idx & (h_map == l)).sum()
                    h_distribution_dict[l]['abstain_per_level_non_boundary'] = (non_boundary_idx & non_ignore_idx & abstain_idx & (h_map == l)).sum()
                    
                    h_distribution_dict[l]['cig_per_level_boundary'] = c_ig_map[boundary_idx & filter_idx & (h_map == l)].sum()
                    h_distribution_dict[l]['cig_per_level_non_boundary'] = c_ig_map[non_boundary_idx & filter_idx & (h_map == l)].sum()
                    
                    h_distribution_dict[l]['correctly_certified_per_level_boundary'] = (boundary_idx & filter_idx & (h_map == l)).sum()
                    h_distribution_dict[l]['correctly_certified_per_level_non_boundary'] = (non_boundary_idx & filter_idx & (h_map == l)).sum()

                    h_distribution_dict[l]['wrongly_certified_per_level_boundary'] = (boundary_idx & non_ignore_idx & wrong_pred & c_idx & (h_map == l)).sum()
                    h_distribution_dict[l]['wrongly_certified_per_level_non_boundary'] = (non_boundary_idx & non_ignore_idx & wrong_pred & c_idx & (h_map == l)).sum()

                    h_distribution_dict[l]['num_pixels_per_level_boundary'] = (boundary_idx & non_ignore_idx & (h_map == l)).sum()
                    h_distribution_dict[l]['num_pixels_per_level_non_boundary'] = (non_boundary_idx & non_ignore_idx & (h_map == l)).sum()


                d['h_distribution_dict'] = h_distribution_dict
            if boundary:
                for b, idx in [('boundary', boundary_idx), ('non_boundary', non_boundary_idx)]:
                    d[f'abstain_count_{b}'] = (classes_certify[non_ignore_idx & idx] == config.DATASET.ABSTAIN_LABEL).sum()
                    d[f'certified_count_{b}'] = (classes_certify[non_ignore_idx & idx] != config.DATASET.ABSTAIN_LABEL).sum()
                    d[f'num_pixels_{b}'] = (non_ignore_idx & idx).sum()
            if images:
                d['classes_certify'] = classes_certify
                d['boundary_map'] = boundary_map
                d['label'] = label
                d['gt_adaptive_label'] = gt_adaptive_label
            return ig_per_class_dict, d
        return ig_per_class_dict
    
    def adaptivecertify(self, model, image_np01, label, n0, n, sigma, tau, alpha, 
                   size, border_padding, f,
                   do_tqdm=False, cuda_id=0, 
                   samples_logits=None,):
        if samples_logits is None:
            _, samples_logits = self.sample(model, image_np01, size, 
                                            border_padding=border_padding, 
                                            N=n+n0, sigma=sigma, 
                                            logits_only=True, 
                                            do_tqdm=do_tqdm, cuda_id=cuda_id)
        samples_flattened = torch.from_numpy(samples_logits.argmax(1).reshape(n+n0, -1))
        posteriors = F.softmax(torch.tensor(samples_logits[:n0]), dim=1)
        diff, _ = self.get_difference(posteriors[:n0])

        samples_flattened_adaptive, h_map_pred = self.adapt_to_hierarchy(samples_flattened, diff, f=f)
        gt_adaptive_label, h_map = self.adapt_to_hierarchy(label.reshape(label.shape[0], -1), diff, f=f)
        certified_pred = self.fast_certify(samples_flattened_adaptive, n0, n, tau, alpha)
        certified_pred = certified_pred.reshape(size)
        return certified_pred, gt_adaptive_label, h_map_pred
    
    def experiment_find_best_threshold(self, model):
        logdir = os.path.join(self.logdir, 'best_thresh')
        os.makedirs(logdir, exist_ok=True)
        def get_threshold_permutations(l, t=3):
            x_sorted = []
            x = list(permutations(l, t))
            for f in x:
                if np.all(np.diff(f) >= 0):
                    x_sorted.append(f)
            return x_sorted
        th_functions = get_threshold_permutations([0, 0, 0.05, 0.25, 0.3, 0.4, 0.5], t=len(config.DATASET.THRESHOLD_FUNCTION))
        print(f'Using {len(th_functions)} threshold functions.')
        model.eval()
        i = 0
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader)):
                image, label, _, name, *border_padding = batch
                i +=1
                name = name[0]
                log_path = os.path.join(logdir, f'{name}.pkl')

                stats = {}
                stats[name] = {}
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32)/255.0 # (1024, 2048, 3)
                # uncertified baseline
                
                _, baseline_logits = self.sample(model, image_np01, 
                                                label.size(), 
                                                border_padding=border_padding, 
                                                N=1, sigma=0, 
                                                logits_only=True, 
                                                do_tqdm=False, cuda_id=0)
                confusion_matrix = get_confusion_matrix(
                    label,
                    baseline_logits,
                    label.size(),
                    len(self.hierarchy['nodes']),
                    config.TRAIN.IGNORE_LABEL)
                
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                pixel_acc = tp.sum()/pos.sum()
    
                print('Baseline accuracy =', pixel_acc)
                stats[name]['baseline'] =  {'confusion_matrix': confusion_matrix}
                # certification 
                n, n0, sigma, tau, alpha = 100, 10, 0.25, 0.75, 0.001
                samples_logits = None
                for n in tqdm(reversed([100]), desc='n samples'):
                    if samples_logits is None:
                        _, samples_logits = self.sample(model, image_np01, label.size(), 
                                                        border_padding=border_padding, 
                                                        N=n+n0, sigma=sigma, 
                                                        logits_only=True, 
                                                        do_tqdm=True, cuda_id=0)
                    else:
                        samples_logits = samples_logits[:n+n0]
                
                    certified_pred = self.segcertify(model, image_np01, n0, n,
                                                    sigma, tau, alpha, label.size(), 
                                                    border_padding=border_padding,
                                                    do_tqdm=True,
                                                    samples_logits=samples_logits)
                    k = (n, n0, None, 0, sigma, tau)
                    cig_per_class, d = self.info_gain_adaptive(certified_pred, label, label, stats=True)
                    print(sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), None, n)

                    stats[name][k] = d

                    for f in tqdm(th_functions, desc='threshold funcs'):
                        certified_adaptive_pred, gt_adaptive_label = self.adaptivecertify(model, image_np01, label, n0, n,
                                                        sigma, tau, alpha, label.size(), 
                                                        border_padding=border_padding,
                                                        f=f,
                                                        do_tqdm=True,
                                                        samples_logits=samples_logits)
                        cig_per_class, d = self.info_gain_adaptive(certified_adaptive_pred, gt_adaptive_label, label, stats=True)
                        print('\n', sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), f, n)
                        k = (n, n0, str(f), 4, sigma, tau)
                        stats[name][k] = d
                pickle.dump(stats, open(log_path, 'wb'))
                print('Dumped', log_path)

    def experiment_h_distribution(self, model):

        model.eval()
        i = 0
        logdir = os.path.join(self.logdir, 'distribution')
        os.makedirs(logdir, exist_ok=True)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader)):
                image, label, _, name, *border_padding = batch
                i +=1
                name = name[0]
                log_path = os.path.join(logdir, f'{name}.pkl')
                if os.path.exists(log_path): 
                    print('skipping')
                    continue
                stats = {name: {}}
                
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32)/255.0 # (1024, 2048, 3)
                # uncertified baseline
                _, baseline_logits = self.sample(model, image_np01, 
                                                label.size(), 
                                                border_padding=border_padding, 
                                                N=1, sigma=0, 
                                                logits_only=True, 
                                                do_tqdm=False, cuda_id=0)
                confusion_matrix = get_confusion_matrix(
                    label,
                    baseline_logits,
                    label.size(),
                    len(self.hierarchy['nodes']),
                    config.TRAIN.IGNORE_LABEL)
                
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                pixel_acc = tp.sum()/pos.sum()
    
                print('Baseline accuracy =', pixel_acc)
                stats[name]['baseline'] =  {'confusion_matrix': confusion_matrix}
                # certification 
                samples_logits = None
                n0, alpha = 10, 0.001
                for sigma in [0.25]:
                    for n in list(reversed(sorted([100]))):
                        if n==100: tau=0.75
                        if n==500: tau=0.95
                        if samples_logits is None:
                            _, samples_logits = self.sample(model, image_np01, label.size(), 
                                                            border_padding=border_padding, 
                                                            N=n+n0, sigma=sigma, 
                                                            logits_only=True, 
                                                            do_tqdm=True, cuda_id=0)
                        else:
                            samples_logits = samples_logits[:n+n0]
                    
                        certified_pred = self.segcertify(model, image_np01, n0, n,
                                                        sigma, tau, alpha, label.size(), 
                                                        border_padding=border_padding,
                                                        do_tqdm=True,
                                                        samples_logits=samples_logits)
                        k = (n, n0, None, 0, sigma, tau)
                        cig_per_class, d = self.info_gain_adaptive(certified_pred, label, label, stats=True)
                        print(sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), None, sigma, n)

                        stats[name][k] = d
                        f = tuple(config.DATASET.THRESHOLD_FUNCTION)
                        certified_adaptive_pred, gt_adaptive_label, h_map = self.adaptivecertify(model, image_np01, label, n0, n,
                                                        sigma, tau, alpha, label.size(), 
                                                        border_padding=border_padding,
                                                        f=f,
                                                        do_tqdm=True,
                                                        samples_logits=samples_logits)
                        print('h_map', len(h_map), np.array(h_map).shape)
                        cig_per_class, d = self.info_gain_adaptive(certified_adaptive_pred, gt_adaptive_label, label, stats=True, h_map=h_map)
                        print('\n', sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), f, sigma, n)
                        k = (n, n0, str(f), 4, sigma, tau)
                        stats[name][k] = d
                    samples_logits = None
                pickle.dump(stats, open(log_path, 'wb'))
                print('Dumped', log_path)
    
    def experiment_table(self, model):
        model.eval()
        i = 0
        logdir = os.path.join(self.logdir, 'table')
        os.makedirs(logdir, exist_ok=True)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader)):
                image, label, _, name, *border_padding = batch
                i +=1
                name = name[0]
                log_path = os.path.join(logdir, f'{name}.pkl')
                if os.path.exists(log_path): 
                    print('skipping')
                    continue
                stats = {name: {}}
                
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32)/255.0 # (1024, 2048, 3)
                # uncertified baseline
                _, baseline_logits = self.sample(model, image_np01, 
                                                label.size(), 
                                                border_padding=border_padding, 
                                                N=1, sigma=0, 
                                                logits_only=True, 
                                                do_tqdm=False, cuda_id=0)
                confusion_matrix = get_confusion_matrix(
                    label,
                    baseline_logits,
                    label.size(),
                    len(self.hierarchy['nodes']),
                    config.TRAIN.IGNORE_LABEL)
                
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                pixel_acc = tp.sum()/pos.sum()
    
                print('Baseline accuracy =', pixel_acc)
                stats[name]['baseline'] =  {'confusion_matrix': confusion_matrix}
                # certification 
                samples_logits = None
                n0, alpha = 10, 0.001
                for sigma in [0.25, 0.33, 0.50]:
                    for n in list(reversed(sorted([100, 500]))):
                        if n==100: tau=0.75
                        if n==500: tau=0.95
                        if samples_logits is None:
                            _, samples_logits = self.sample(model, image_np01, label.size(), 
                                                            border_padding=border_padding, 
                                                            N=n+n0, sigma=sigma, 
                                                            logits_only=True, 
                                                            do_tqdm=True, cuda_id=0)
                        else:
                            samples_logits = samples_logits[:n+n0]
                    
                        certified_pred = self.segcertify(model, image_np01, n0, n,
                                                        sigma, tau, alpha, label.size(), 
                                                        border_padding=border_padding,
                                                        do_tqdm=True,
                                                        samples_logits=samples_logits)
                        k = (n, n0, None, 0, sigma, tau)
                        cig_per_class, d = self.info_gain_adaptive(certified_pred, label, label, stats=True)
                        print(sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), None, sigma, n)

                        stats[name][k] = d
                        f = tuple(config.DATASET.THRESHOLD_FUNCTION)
                        certified_adaptive_pred, gt_adaptive_label, h_map = self.adaptivecertify(model, image_np01, label, n0, n,
                                                        sigma, tau, alpha, label.size(), 
                                                        border_padding=border_padding,
                                                        f=f,
                                                        do_tqdm=True,
                                                        samples_logits=samples_logits)
                        cig_per_class, d = self.info_gain_adaptive(certified_adaptive_pred, gt_adaptive_label, label, stats=True, h_map=h_map)
                        print('\n', sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), f, sigma, n)
                        k = (n, n0, str(f), 4, sigma, tau)
                        stats[name][k] = d
                    samples_logits = None
                pickle.dump(stats, open(log_path, 'wb'))
                print('Dumped', log_path)
    
    def exp_fluctuations(self, model):
        model.eval()
        i = 0
        logdir = os.path.join(self.logdir, 'fluctuations')
        os.makedirs(logdir, exist_ok=True)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader)):
                image, label, _, name, *border_padding = batch
                i +=1
                name = name[0]
                log_path = os.path.join(logdir, f'{name}.pkl')
                if os.path.exists(log_path): continue
                stats = {name: {}}
                
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32)/255.0 # (1024, 2048, 3)
                # uncertified baseline
                _, baseline_logits = self.sample(model, image_np01, 
                                                label.size(), 
                                                border_padding=border_padding, 
                                                N=1, sigma=0, 
                                                logits_only=True, 
                                                do_tqdm=False, cuda_id=0)
                confusion_matrix = get_confusion_matrix(
                    label,
                    baseline_logits,
                    label.size(),
                    len(self.hierarchy['nodes']),
                    config.TRAIN.IGNORE_LABEL)
                
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                pixel_acc = tp.sum()/pos.sum()
    
                print('Baseline accuracy =', pixel_acc)
                stats[name]['baseline'] =  {'confusion_matrix': confusion_matrix}
                # certification 
                samples_logits = None
                n0, alpha, sigma, tau = 10, 0.001, 0.25, 0.75
                n = 100
                _, samples_logits = self.sample(model, image_np01, label.size(), 
                                                border_padding=border_padding, 
                                                N=n+n0, sigma=sigma, 
                                                logits_only=True, 
                                                do_tqdm=True, cuda_id=0,)

                non_ignore_idx = label.flatten().numpy() != config.DATASET.IGNORE_LABEL
                certified_pred, d_fluct = self.segcertify(model, image_np01, n0, n,
                                                sigma, tau, alpha, label.size(), 
                                                border_padding=border_padding,
                                                do_tqdm=True,
                                                samples_logits=samples_logits, stats=True, non_ignore_idx=non_ignore_idx)
                k = (n, n0, None, 0, sigma, tau)
                stats[name][k] = d_fluct
                cig_per_class, d = self.info_gain_adaptive(certified_pred, label, label, stats=True)
                print(sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), None, sigma, n)
                for k_, v_ in d.items():
                    stats[name][k][k_] = v_
                pickle.dump(stats, open(log_path, 'wb'))
                print('Dumped', log_path)        

    def exp_images(self, model):
        model.eval()
        i = 0
        logdir = os.path.join(self.logdir, 'images')
        os.makedirs(logdir, exist_ok=True)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader)):
                image, label, _, name, *border_padding = batch
                i +=1
                name = name[0]
                log_path = os.path.join(logdir, f'{name}.pkl')
                if os.path.exists(log_path) and False: 
                    print('skipping')
                    continue
                stats = {name: {}}
                
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32)/255.0 # (1024, 2048, 3)
                # uncertified baseline
                _, baseline_logits = self.sample(model, image_np01, 
                                                label.size(), 
                                                border_padding=border_padding, 
                                                N=1, sigma=0, 
                                                logits_only=True, 
                                                do_tqdm=False, cuda_id=0)
                confusion_matrix = get_confusion_matrix(
                    label,
                    baseline_logits,
                    label.size(),
                    len(self.hierarchy['nodes']),
                    config.TRAIN.IGNORE_LABEL)
                
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                pixel_acc = tp.sum()/pos.sum()
    
                print('Baseline accuracy =', pixel_acc)
                stats[name]['baseline'] =  {'confusion_matrix': confusion_matrix}
                # certification 
                samples_logits = None
                n0, alpha = 10, 0.001
                for sigma in [0.25]:
                    for n in list(reversed(sorted([100]))):
                        if n==100: tau=0.75
                        if n==500: tau=0.95
                        if samples_logits is None:
                            _, samples_logits = self.sample(model, image_np01, label.size(), 
                                                            border_padding=border_padding, 
                                                            N=n+n0, sigma=sigma, 
                                                            logits_only=True, 
                                                            do_tqdm=True, cuda_id=0)
                        else:
                            samples_logits = samples_logits[:n+n0]
                    
                        certified_pred = self.segcertify(model, image_np01, n0, n,
                                                        sigma, tau, alpha, label.size(), 
                                                        border_padding=border_padding,
                                                        do_tqdm=True,
                                                        samples_logits=samples_logits)
                        #stats[name]['images'] = {'certified_seg': certified_pred}

                        k = (n, n0, None, 0, sigma, tau)
                        cig_per_class, d = self.info_gain_adaptive(certified_pred, label, label, stats=True, images=True)
                        d['image_np01'] = image_np01
                        print(sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), None, sigma, n)

                        stats[name][k] = d
                        f = tuple(config.DATASET.THRESHOLD_FUNCTION)
                        certified_adaptive_pred, gt_adaptive_label, h_map = self.adaptivecertify(model, image_np01, label, n0, n,
                                                        sigma, tau, alpha, label.size(), 
                                                        border_padding=border_padding,
                                                        f=f,
                                                        do_tqdm=True,
                                                        samples_logits=samples_logits)
                        print('h_map', len(h_map), np.array(h_map).shape)
                        cig_per_class, d = self.info_gain_adaptive(certified_adaptive_pred, gt_adaptive_label, label, stats=True, h_map=h_map, images=True)
                        d['image_np01'] = image_np01
                        print('\n', sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES), f, sigma, n)
                        k = (n, n0, str(f), 4, sigma, tau)
                        stats[name][k] = d
                    samples_logits = None
                pickle.dump(stats, open(log_path, 'wb'))
                print('Dumped', log_path)
    def sigma_inference(self, model):
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader)):
                image, label, _, name, *border_padding = batch
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32)/255.0 # (1024, 2048, 3)
                
                # uncertified baseline
                _, baseline_logits = self.sample(model, image_np01, 
                                                label.size(), 
                                                border_padding=border_padding, 
                                                N=1, sigma=0, 
                                                logits_only=True, 
                                                do_tqdm=False, cuda_id=0)
                confusion_matrix = get_confusion_matrix(
                    label,
                    baseline_logits,
                    label.size(),
                    len(self.hierarchy['nodes']),
                    config.TRAIN.IGNORE_LABEL)


                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                pixel_acc = tp.sum()/pos.sum()
    
                print('Baseline accuracy =', pixel_acc)
                n, n0, sigma, tau, alpha = 100, 10, 0.25, 0.75, 0.001
                # certification
                _, samples_logits = self.sample(model, image_np01, label.size(), 
                                                border_padding=border_padding, 
                                                N=n+n0, sigma=sigma, 
                                                logits_only=True, 
                                                do_tqdm=True, cuda_id=0)
            
                certified_pred = self.segcertify(model, image_np01, n0, n,
                                                sigma, tau, alpha, label.size(), 
                                                border_padding=border_padding,
                                                do_tqdm=True,
                                                samples_logits=samples_logits)
                cig_per_class = self.info_gain_adaptive(certified_pred, label, label)
                print('SegCertify CIG', sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES))
                certified_adaptive_pred, gt_adaptive_label = self.adaptivecertify(model, image_np01, label, n0, n,
                                                sigma, tau, alpha, label.size(), 
                                                border_padding=border_padding,
                                                do_tqdm=True,
                                                samples_logits=samples_logits, f=tuple(config.DATASET.THRESHOLD_FUNCTION))
                cig_per_class = self.info_gain_adaptive(certified_adaptive_pred, gt_adaptive_label, label)
                print('AdaptiveCertify CIG', sum(cig_per_class['cig_per_cls'])/sum(cig_per_class['num_pixels_per_cls'])/np.log(self.config.DATASET.NUM_CLASSES))


                


if __name__ == '__main__':
    Certifier()   
