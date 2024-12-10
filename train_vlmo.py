import argparse
import torch

from train_main_vlmo import main
from third_party.dassl.utils import setup_logger, set_random_seed, collect_env_info
from third_party.dassl.config import get_cfg_default
import args_setting
import os
from pathlib import Path
# custom


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE = 1
    cfg.DATALOADER.TRAIN.NUM_FRAMES = 6
    cfg.DATALOADER.TEST.NUM_FRAMES = 6
    cfg.DATALOADER.TEST.TESTING = False

    cfg.INPUT.IMG_SIZE = 352

    cfg.OPTIM.LR = 1e-4
    cfg.OPTIM.LR_PRETRAINED_MODULE = 5e-5
    cfg.OPTIM.LR_PRETRAINED_MODULE_NAMES = ['temporal_encoder', 'clip_resnet_model.visual']
    cfg.OPTIM.LR_ENCODER = 5e-5
    cfg.OPTIM.LR_ENCODER_NAMES = ['uprompt_encoder', 'context_encoder']
    cfg.OPTIM.LR_DECODER = 5e-5
    cfg.OPTIM.LR_DECODER_NAMES = ['segDecoder']
    cfg.OPTIM.LR_LINEAR_PROJ_NAMES = ['reduce', 'resizer']
    cfg.OPTIM.LR_LINEAR_PROJ_MULT = 1.0
    cfg.OPTIM.EPOCHS = 24
    cfg.OPTIM.LR_DROP = [12]

    cfg.MODEL.BACKBONE.HIDDEN_DIM = 512
    cfg.MODEL.BACKBONE.N_HEADS = 8
    cfg.MODEL.BACKBONE.T_DIM = 768
    cfg.MODEL.BACKBONE.V_DIM = 1024
    cfg.MODEL.BACKBONE.DOWN_SCALE = 14

    cfg.MODEL.TEMPORAL = CN()
    cfg.MODEL.TEMPORAL.TEMP_DIM = 384

    cfg.FEATURE = CN()
    cfg.FEATURE.EXTRACT_LAYERS = (3, 7, 9, 11)
    cfg.FEATURE.INSERT_LAYERS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    cfg.TRAIN.BINARY = True
    cfg.TRAIN.NUM_VPROMPT = 0
    cfg.TRAIN.NUM_TPROMPT = 0
    cfg.TRAIN.ONLY_USE_TEXT_PROMPT = False
    cfg.TRAIN.ONLY_USE_VISION_PROMPT = False
    cfg.TRAIN.USE_UNIFY_PROMPT = False
    cfg.TRAIN.USE_TEMPORAL = False
    cfg.TRAIN.USE_MULTISCALE = False
    cfg.TRAIN.USE_TEXT_CONTEXT = False
    cfg.TRAIN.TEXT_CONTEXT_CAT_VISION = False
    cfg.TRAIN.TEXT_CONTEXT_LEN = 0

    cfg.VLMO = CN()
    cfg.VLMO.PRETRAIN_FILE = 'vlmo_base_patch16_384_coco.pt'
    cfg.VLMO.VLMO_ARCH = 'vlmo_base_patch16'
    cfg.VLMO.TRAINING = True
    cfg.VLMO.HIDDEN_DIM = 768
    cfg.VLMO.PATCH_SIZE = 16
    cfg.VLMO.IMG_SIZE = 384
    cfg.VLMO.DEPTH = 12
    cfg.VLMO.NUM_HEADS = 12
    cfg.VLMO.DROP_PATH_RATE = 0.15
    cfg.VLMO.MODE = 'separate'
    cfg.VLMO.LANGUAGE = CN()
    cfg.VLMO.LANGUAGE.MASK_LANGUAGE = False

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 2. From the method config file
    if args.experiment_name:
        config_file = args.experiment_name + '.yaml'
        config_file = os.path.join(args.experiment_path + args.date, config_file)
        print("load cfg file from {}".format(config_file))
        cfg.merge_from_file(config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script',
                                     parents=[args_setting.get_args_parser()])  # 初始化命令行解析器
    args = parser.parse_args()  # 解析添加的参数
    repeat = ""
    if args.exp:
        repeat = "_exp" + args.exp
    output_dir = os.path.join(args.output_dir, args.experiment_name + repeat)
    if args.output_dir:  # 创建输出路径
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    cfg = setup_cfg(args)

    #print_args(args, cfg)
    #print("Collecting env info ...")
    #print("** System info **\n{}\n".format(collect_env_info()))

    main(args, cfg)
