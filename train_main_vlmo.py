"""
Training script of ReferFormer
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
from datetime import datetime as timelog
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate, evaluate_a2d
from models import build_model_vlmo


import args_setting
from tools.load_pretrained_weights import pre_trained_model_to_finetune
from util.WarmupMultiStepLR import WarmupMultiStepLR





def main(args, cfg):
    os.environ["USE_TF"] = 'None'

    args.masks = True       #是否训练分割头

    utils.init_distributed_mode(args)       #配置一些关于进程的环境变量吧，用于分布式操作/训练
    print("git:\n  {}\n".format(utils.get_sha()))       #给sha密码
    #print(args)
    
    print(f'\n Run on {args.dataset_file} dataset.')    #dataset name
    print('\n')

    device = torch.device(args.device)      #cuda/cpu

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()     #固定种子 + 进程rank
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False

    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
    g = torch.Generator()
    g.manual_seed(seed)

    model, criterion, postprocessor = build_model_vlmo(args, cfg) #构建模型结构     返回模型、loss函数、后处理器
    model.to(device)

    model_without_ddp = model       #没有distributedDataParallel
    if args.distributed:    #是否进行分布式训练
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    for n, p in model_without_ddp.named_parameters():
        if p.requires_grad:
            print(n)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)        #numel()返回数组中元素的个数，用来得到网络中参数的总数目
    print('number of params:', n_parameters)
    #exit()
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
                 if not match_name_keywords(n, cfg.OPTIM.LR_ENCODER_NAMES) and not match_name_keywords(n, cfg.OPTIM.LR_DECODER_NAMES)
                 and not match_name_keywords(n, cfg.OPTIM.LR_LINEAR_PROJ_NAMES) and not match_name_keywords(n, cfg.OPTIM.LR_PRETRAINED_MODULE_NAMES) and p.requires_grad],
            "lr": cfg.OPTIM.LR,      #没用
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
            "lr": cfg.OPTIM.LR_PRETRAINED_MODULE,       #没用
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, cfg.OPTIM.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.OPTIM.LR * cfg.OPTIM.LR_LINEAR_PROJ_MULT,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.OPTIM.LR,
                                  weight_decay=args.weight_decay)   #adam优化器
    if args.lr_warmup:
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=args.lr_milestones,
                                         gamma=args.lr_gamma, warmup_iters=args.lr_warmup_iters)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.OPTIM.LR_DROP)    #学习率

    # no validation ground truth for ytvos dataset
    dataset_train = build_dataset(args.dataset_file, image_set='train', args=args, cfg=cfg)      #image_set表示加载训练集还是验证集,arg.dataset_file表示用的数据集的名称

    if args.distributed:
        if args.cache_mode:     #是否缓存图片到内存中
            sampler_train = samplers.NodeDistributedSampler(dataset_train)      #分布式dataloader 采样器
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(    #按batch_size的大小采样   训练集的采样器，需要在dataloader中用到
        sampler_train, cfg.DATALOADER.TRAIN.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,            #训练集的dataloader
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, worker_init_fn=worker_init_fn,
    generator=g)
    
    # A2D-Sentences
    if args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
        dataset_val = build_dataset(args.dataset_file, image_set='val', args=args, cfg=cfg)
        if args.distributed:        #获取验证集的采样器，这两个数据集需要val
            if args.cache_mode:
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, cfg.DATALOADER.TRAIN.BATCH_SIZE, sampler=sampler_val,     #验证集的dataloader
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True, worker_init_fn=worker_init_fn,
    generator=g)

    
    if args.dataset_file == "davis":
        assert args.pretrained_weights is not None, "Please provide the pretrained weight to finetune for Ref-DAVIS17"
        print("============================================>")
        print("Ref-DAVIS17 are finetuned using the checkpoint trained on Ref-Youtube-VOS")
        print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)       #预训练的模型用别的数据集微调需要经过一个什么处理？
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")

    if args.dataset_file == "jhmdb":
        assert args.resume is not None, "Please provide the checkpoint to resume for JHMDB-Sentences"
        print("============================================>")
        print("JHMDB-Sentences are directly evaluated using the checkpoint trained on A2D-Sentences")
        print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
        # load checkpoint in the args.resume
        print("============================================>")

    # for Ref-Youtube-VOS and A2D-Sentences
    # finetune using the pretrained weights on Ref-COCO
    if args.dataset_file != "davis" and args.dataset_file != "jhmdb" and args.pretrained_weights is not None:
        print("============================================>")
        print("Load pretrained weights from {} ...".format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")

    repeat = ""
    if args.exp:
        repeat = "_exp" + args.exp
    output_dir = os.path.join(args.output_dir, args.experiment_name + repeat)
    output_dir = Path(output_dir)      #保存输出的文件夹路径
    if args.resume:     #从checkpoint恢复/继续
        if args.resume.startswith('https'):     #字符串是否以https开头，是的话得下载checkpoint
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            resume_path = os.path.join(output_dir, args.resume)
            checkpoint = torch.load(resume_path, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)  #加载的checkpoint和现在的模型结构不对应的地方，missing_keys表示缺失的weights,unexpected_keys表示多出来的weights
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)    #深复制函数，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = cfg.OPTIM.LR_DROP
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
                    'Only A2D-Sentences and JHMDB-Sentences datasetssss support evaluation'
        test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args, cfg)
        return
    #tensorboard
    writer = None
    if args.rank == 0:
        nowtime = str(timelog.now())
        nowtime = "_".join(nowtime.split())
        log_dir = args.output_dir + "logs/"+nowtime+"/"
        writer = SummaryWriter(log_dir=log_dir)#, comment='_scalars', filename_suffix="loss")
    num_writer = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, cfg.OPTIM.EPOCHS):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if writer is not None:
            writer.add_scalar("lr/lr_default", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            writer.add_scalar("lr/lr_vision_language_decoder", optimizer.state_dict()['param_groups'][1]['lr'], epoch)
            writer.add_scalar("lr/lr_segmentor_decoder", optimizer.state_dict()['param_groups'][2]['lr'], epoch)
            writer.add_scalar("lr/lr_reduce_resizer_linearProj", optimizer.state_dict()['param_groups'][3]['lr'], epoch)
            writer.add_scalar("lr/lr_visuak_backbone", optimizer.state_dict()['param_groups'][4]['lr'], epoch)
        train_stats, last_num_writer = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, writer, num_writer, args, cfg,
            args.clip_max_norm)
        num_writer = last_num_writer
        lr_scheduler.step() #更新学习率？
        if args.output_dir:     #保存checkpoints
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

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},   #日志
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.dataset_file == 'a2d':
            test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args, cfg)
            log_stats.update({**{f'{k}': v for k, v in test_stats.items()}})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[args_setting.get_args_parser()])    #初始化命令行解析器
    args = parser.parse_args()      #解析添加的参数
    repeat = ""
    if args.exp:
        repeat = "_exp" + args.exp
    output_dir = os.path.join(args.output_dir, args.experiment_name + repeat)
    if args.output_dir:     #创建输出路径
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    main(args)



