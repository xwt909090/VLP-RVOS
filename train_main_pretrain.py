import argparse
import datetime
import json
import random
import time
from pathlib import Path
from collections import namedtuple
from functools import partial

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import util.misc as utils
import datasets.samplers as samplers
from datasets.coco_eval import CocoEvaluator
from datasets import build_dataset, get_coco_api_from_dataset
from engine_nohistory import evaluate, train_one_epoch
from models import build_model_clip
from models.postprocessors import build_postprocessors

import args_setting


def main(args, cfg):
    # set environ
    os.environ["MDETR_CPU_REDUCE"] = "1"

    args.masks = True
    assert args.dataset_file in ["refcoco", "refcoco+", "refcocog", "all"]

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

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

    model, criterion, postprocessor = build_model_clip(args, cfg, batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE)  # 构建模型结构     返回模型、loss函数、后处理器
    model.to(device)

    model_without_ddp = model  # 目前是没有分布式设置的模型
    if args.distributed:  # 对模型分布式设置
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # numel()返回数组中元素的个数，用来得到网络中参数的总数目
    print('number of params:', n_parameters)

    for n, p in model_without_ddp.named_parameters():
        if p.requires_grad:
            print(n)

    # lr_backbone_names = ["backbone.0", "text_encoder"]
    def match_name_keywords(n, name_keywords):  # 只要name_keywords中有一个元素存在于n中就返回true
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #    print(n)

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
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.OPTIM.LR_DROP)    #学习率

    # build train  dataset
    if args.dataset_file != "all":
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args, cfg=cfg)
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]
        dataset_train = torch.utils.data.ConcatDataset(
            [build_dataset(name, image_set="train", args=args, cfg=cfg) for name in dataset_names]
        )

    print("\nTrain dataset sample number: ", len(dataset_train))
    print("\n")

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
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    # build val datasetssss
    Val_all = namedtuple(typename="val_data",
                         field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])  # 返回一个带名字的元组
    if args.dataset_file != "all":
        dataset_names = [args.dataset_file]
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]

    val_tuples = []
    for name in dataset_names:
        dataset_val = build_dataset(name, image_set="val", args=args, cfg=cfg)
        sampler_val = (
            samplers.DistributedSampler(dataset_val,
                                        shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(
                dataset_val)
        )
        data_loader_val = DataLoader(
            dataset_val,
            cfg.DATALOADER.TRAIN.BATCH_SIZE,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dataset_val)

        val_tuples.append(Val_all(dataset_name=name, dataloader=data_loader_val, base_ds=base_ds, evaluator_list=None))

    # build evaluator list for dataset_val
    def build_evaluator_list(base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        iou_types = ["bbox"]
        if args.masks:
            iou_types.append("segm")

        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        # TODO: currently ont support RefExpEvaluator (memory error)
        return evaluator_list

    repeat = ""
    if args.exp:
        repeat = "_exp" + args.exp
    output_dir = os.path.join(args.output_dir, args.experiment_name + repeat)
    output_dir = Path(output_dir)  # 保存输出的文件夹路径
    if args.resume:
        if args.resume.startswith('https'):  # 字符串是否以https开头，是的话得下载checkpoint
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
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = cfg.OPTIM.LR_DROP
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        '''if not args.eval:
            test_stats = {}
            for i, item in enumerate(val_tuples):
                evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
                postprocessors = build_postprocessors(args, item.dataset_name)
                item = item._replace(evaluator_list=evaluator_list)
                print(f"Evaluating {item.dataset_name}")
                curr_test_stats = evaluate(
                    model=model,
                    criterion=criterion,
                    postprocessors=postprocessors,
                    data_loader=item.dataloader,
                    evaluator_list=item.evaluator_list,
                    device=device,
                    args=args,
                    cfg=cfg,
                )
                test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})

            log_stats = {
                **{f"test_{k}": v for k, v in test_stats.items()},
                "n_parameters": n_parameters,
            }
            print(log_stats)'''

    if args.eval:
        print("Evaluating......")
        test_stats = {}
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
            postprocessors = build_postprocessors(args, item.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=model,
                criterion=criterion,
                postprocessors=postprocessors,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
                cfg=cfg,
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        print(log_stats)

        return
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

        '''test_stats = {}
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
            postprocessors = build_postprocessors(args, item.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=model,
                criterion=criterion,
                postprocessors=postprocessors,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
                cfg=cfg,
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})'''

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     # **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet pretrain training and evaluation script',
                                     parents=[args_setting.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

