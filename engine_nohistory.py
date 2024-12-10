"""
Train and eval functions used in train_main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
from models import postprocessors
import os
import sys
from typing import Iterable
import numpy as np
from PIL import Image
from torch.nn import functional as nnf
import cv2

import torch
import torch.distributed as dist

import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from datasets.coco_eval import CocoEvaluator
from datasets.refexp_eval import RefExpEvaluator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.a2d_eval import calculate_precision_at_k_and_iou_metrics, calculate_bbox_precision_at_k_and_iou_metrics
from datasets.ytvos import YTVOSDataset
import copy

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, num_writer, args, cfg, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_encoder', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_decoder', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_pretrained_module', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}], Num_Frames: [{}]'.format(epoch, cfg.DATALOADER.TRAIN.NUM_FRAMES)
    print_freq = 10
    cnt = 10
    i = 0
    write_interval = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, writer, args, epoch, header):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs, model_need_to_write = model(samples, captions, targets)     #模型得到输出？
        loss_dict, loss_need_to_write = criterion(outputs, targets, args, cfg)     #计算loss函数项

        weight_dict = criterion.weight_dict     #loss的权重系数
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)       #total loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)        #对所有进程中的loss值sum/avg，然后替换每个进程中的loss，使每个进程中的loss统一
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v        #不乘权重系数的loss
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]       #乘权重系数的loss
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())  #对所有loss项求和

        loss_value = losses_reduced_scaled.item()   #精度更高

        if not math.isfinite(loss_value):   #判断是否有限
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()   #清空梯度
        losses.backward()   #反向传播梯度

        '''for name, param in model.named_parameters():
            if param.grad is None:
                print(name)'''

        if max_norm > 0:    #gradient clipping
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_encoder=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_decoder=optimizer.param_groups[2]["lr"])
        metric_logger.update(lr_pretrained_module=optimizer.param_groups[3]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        if args.rank == 0:
            num_writer += 1
        '''cnt -= 1
        if cnt == 0:
            break'''

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, num_writer


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, evaluator_list, device, args, cfg):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    predictions = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header=header):
        dataset_name = targets[0]["dataset_name"]
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs, _ = model(samples, captions, targets)
        loss_dict, _ = criterion(outputs, targets, args, cfg)     #计算loss函数项
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        for evaluator in evaluator_list:
            evaluator.update(res)

        # REC & RES predictions
        for p, target in zip(results, targets):
            for s, b, m in zip(p['scores'], p['boxes'], p['rle_masks']):
                predictions.append({'image_id': target['image_id'].item(),
                                    'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                    'bbox': b.tolist(),
                                    'segmentation': m,
                                    'score': s.item()})

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    refexp_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()
        elif isinstance(evaluator, RefExpEvaluator):
            refexp_res = evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # update stats
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()
    if refexp_res is not None:
        stats.update(refexp_res)

    # evaluate RES
    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]

    eval_metrics = {}
    if utils.is_main_process():
        if dataset_name == 'refcoco':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco/instances_refcoco_val.json'))
        elif dataset_name == 'refcoco+':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco+/instances_refcoco+_val.json'))
        elif dataset_name == 'refcocog':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcocog/instances_refcocog_val.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        # ap_metrics = coco_eval.stats[:6]
        # eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        # bbox
        precision_at_k, overall_iou, mean_iou = calculate_bbox_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'bbox P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'bbox overall_iou': overall_iou, 'bbox mean_iou': mean_iou})
        # mask
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'segm P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'segm overall_iou': overall_iou, 'segm mean_iou': mean_iou})
        print(eval_metrics)
        stats.update(eval_metrics)

    return stats


'''@torch.no_grad()
def evaluate_a2d(model, data_loader, postprocessor, device, args, cfg):
    model.eval()
    predictions = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        image_ids = [t['image_id'] for t in targets]

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs, _ = model(samples, captions, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        processed_outputs = postprocessor(outputs, orig_target_sizes, target_sizes)

        for p, image_id in zip(processed_outputs, image_ids):
            for s, m in zip(p['scores'], p['rle_masks']):
                predictions.append({'image_id': image_id,
                                    'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                    'segmentation': m,
                                    'score': s.item()})

    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]
    # evaluation
    eval_metrics = {}
    if utils.is_main_process():
        if args.dataset_file == 'a2d':
            coco_gt = COCO(os.path.join(args.a2d_path, 'a2d_sentences_test_annotations_in_coco_format.json'))
        elif args.dataset_file == 'jhmdb':
            coco_gt = COCO(os.path.join(args.jhmdb_path, 'jhmdb_sentences_gt_annotations_in_coco_format.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
        print(eval_metrics)

    # sync all processes before starting a new epoch or exiting
    dist.barrier()
    return eval_metrics'''


@torch.no_grad()
def evaluate_a2d(model, data_loader, postprocessor, device, args, cfg):
    model.eval()
    predictions = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        image_ids = [t['image_id'] for t in targets]

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs, _ = model(samples, captions, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        processed_outputs = postprocessor(outputs, orig_target_sizes, target_sizes, targets=targets)
        for p, image_id in zip(processed_outputs, image_ids):
            for m in p['rle_masks']:
                predictions.append({'image_id': image_id,
                                    'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                    'segmentation': m,
                                    'score': 1})

    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]
    # evaluation
    eval_metrics = {}
    if utils.is_main_process():
        if args.dataset_file == 'a2d':
            coco_gt = COCO(os.path.join(args.a2d_path, 'a2d_sentences_test_annotations_in_coco_format.json'))
        elif args.dataset_file == 'jhmdb':
            coco_gt = COCO(os.path.join(args.jhmdb_path, 'jhmdb_sentences_gt_annotations_in_coco_format.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
        print(eval_metrics)

    # sync all processes before starting a new epoch or exiting
    dist.barrier()
    return eval_metrics



