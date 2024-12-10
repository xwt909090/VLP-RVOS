"""
Training script of ReferFormer
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
import datasets.samplers as samplers
from datasets.coco_eval import CocoEvaluator
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, evaluate_a2d
from models import our_build_model_vlmo as our_build_model
from models.postprocessors import build_postprocessors

from tools.load_pretrained_weights import pre_trained_model_to_finetune
import args_setting



def main(args, cfg):
    args.masks = True
    args.dataset_file = 'joint' # joint training of ytvos and refcoco
    #args.binary = 1             # only run on binary referred

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    
    print(f'\n Run on {args.dataset_file} dataset.')
    print('\n')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()  # 固定种子 + 进程rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessor = our_build_model(args, cfg)  # 构建模型结构     返回模型、loss函数、后处理器
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.OPTIM.LR_ENCODER_NAMES) and not match_name_keywords(n,
                                                                                                       cfg.OPTIM.LR_DECODER_NAMES)
                 and not match_name_keywords(n, cfg.OPTIM.LR_LINEAR_PROJ_NAMES) and not match_name_keywords(n,
                                                                                                            cfg.OPTIM.LR_PRETRAINED_MODULE_NAMES) and p.requires_grad],
            "lr": cfg.OPTIM.LR,  # 没用
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, cfg.OPTIM.LR_ENCODER_NAMES) and p.requires_grad],
            "lr": cfg.OPTIM.LR_ENCODER,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, cfg.OPTIM.LR_DECODER_NAMES) and p.requires_grad],
            "lr": cfg.OPTIM.LR_DECODER,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, cfg.OPTIM.LR_PRETRAINED_MODULE_NAMES) and p.requires_grad],
            "lr": cfg.OPTIM.LR_PRETRAINED_MODULE,  # 没用
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, cfg.OPTIM.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.OPTIM.LR * cfg.OPTIM.LR_LINEAR_PROJ_MULT,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.OPTIM.LR,
                                  weight_decay=args.weight_decay)  # adam优化器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.OPTIM.LR_DROP)  # 学习率

    # no validation ground truth for ytvos dataset
    dataset_train = build_dataset(args.dataset_file, image_set='train', args=args, cfg=cfg)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(  # 按batch_size的大小采样   训练集的采样器，需要在dataloader中用到
        sampler_train, cfg.DATALOADER.TRAIN.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    repeat = ""
    if args.exp:
        repeat = "_exp" + args.exp
    output_dir = os.path.join(args.output_dir, args.experiment_name + repeat)
    output_dir = Path(output_dir)  # 保存输出的文件夹路径
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            resume_path = os.path.join(output_dir, args.resume)
            checkpoint = torch.load(resume_path, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    writer = None
    if args.rank == 0:
        log_dir = "./output/ours/refcoco_dirs/logs/"
        writer = SummaryWriter(log_dir=log_dir)  # , comment='_scalars', filename_suffix="loss")
    num_writer = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, cfg.OPTIM.EPOCHS):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats, last_num_writer = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, writer, num_writer, args, cfg,
            args.clip_max_norm)
        num_writer = last_num_writer
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}


        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer training and evaluation script', parents=[args_setting.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

