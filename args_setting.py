import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference train_test_scripts.', add_help=False)
    # parser.add_argument('--lr', default=1e-4, type=float)
    #
    # parser.add_argument('--lr_pretrained_module', default=5e-5, type=float)  # clip的visual backbone
    # parser.add_argument('--lr_pretrained_module_names', default=['temporal_encoder', 'clip_resnet_model.visual'],
    #                     type=str, nargs='+')
    # parser.add_argument('--lr_encoder', default=5e-5, type=float)  # 中间vision-language transformer模块
    # parser.add_argument('--lr_encoder_names', default=['uprompt_encoder', 'context_encoder'],
    #                     type=str, nargs='+')
    # parser.add_argument('--lr_decoder', default=5e-5, type=float)  # 分割头segmentor decoder
    # parser.add_argument('--lr_decoder_names', default=['segDecoder'], type=str, nargs='+')
    #
    # parser.add_argument('--lr_linear_proj_names', default=['reduce', 'resizer'], type=str, nargs='+')  # 其他的线性层
    # parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    # parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument('--lr_drop', default=[6, 8], type=int, nargs='+')
    parser.add_argument('--lr_warmup', default=False, action='store_true')
    parser.add_argument('--lr_milestones', default=[8, 10], type=int, nargs='+')
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--lr_warmup_iters', default=5, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--amp', default=False, action='store_true')

    parser.add_argument('--experiment_path', type=str, default='./experiments/ytvos/',
                        help="Path to the experiment.")
    parser.add_argument('--date', type=str, default='',
                        help="date to the experiment.")
    parser.add_argument('--experiment_name', type=str, default='uprompt_t37-v10_b4_lr5e-5_f8_s336_tem_prenorm',
                        help="Name of the experiment.")
    parser.add_argument('--exp', type=str, default='',
                        help="Num of the experiment repeat.")
    parser.add_argument('--test_start', type=int, default=11,
                        help="start checkpoint to test")
    parser.add_argument('--test_end', type=int, default=11,
                        help="end checkpoint to test")
    # Model parameters
    # load the pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.")

    # clip visual encoder version
    parser.add_argument('--version', default='ViT-B/16', type=str,
                        help="version of the visual encoder of clip to use")
    parser.add_argument('--reduce_dim', default=64, type=int,
                        help="reduce_dim of the decoder to use")

    # Variants of Deformable DETR
    # parser.add_argument('--with_box_refine', default=False, action='store_true')
    # parser.add_argument('--two_stage', default=False, action='store_true')  # NOTE: must be false

    # * Backbone
    # ["resnet50", "resnet101", "swin_t_p4w7", "swin_s_p4w7", "swin_b_p4w7", "swin_l_p4w7"]
    # ["video_swin_t_p4w7", "video_swin_s_p4w7", "video_swin_b_p4w7"]

    parser.add_argument('--use_checkpoint', action='store_true',
                        help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true',  # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    # parser.add_argument('--hidden_dim', default=256, type=int,
    #                     help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    # parser.add_argument('--nheads', default=8, type=int,
    #                     help="Number of attention heads inside the transformer's attentions")
    # parser.add_argument('--num_frames', default=3, type=int,
    #                     help="Number of clip frames for training")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    # for text
    parser.add_argument('--freeze_text_encoder', action='store_true')  # default: False

    # * Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                     help="Train segmentation head if the flag is provided")
    # parser.add_argument('--mask_dim', default=256, type=int,
    #                     help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    # parser.add_argument('--controller_layers', default=3, type=int,
    #                     help="Dynamic conv layer number")
    # parser.add_argument('--dynamic_mask_channels', default=8, type=int,
    #                     help="Dynamic conv final channel number")
    # parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
    #                     help="Disables relative coordinates")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--ce_mask_loss_coef', default=5, type=float)
    parser.add_argument('--pixel_ce_mask_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--mask_pixel_text_loss_coef', default=2, type=float)
    parser.add_argument('--dice_pixel_text_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--loss_alpha', default=1, type=float)

    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasetssss for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name')
    parser.add_argument('--coco_path', type=str, default='/workspace/datasets/refcoco/refcoco_rvos')
    parser.add_argument('--ytvos_path', type=str,
                        default='/workspace/datasets/ref-youtube-vos')  # /workspace/datasets/ref-youtube-vos    data/ref-youtube-vos
    parser.add_argument('--davis_path', type=str, default='/workspace/datasets/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='/workspace/datasets/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='/workspace/datasets/jhmdb_sentences')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    parser.add_argument('--img_size', default=352, type=int, help="size for the per frame")
    # parser.add_argument('--binary', action='store_true')  # binary是啥意思啊？
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./output/ours/ytvos_dirs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float)  # binary threshold for mask
    parser.add_argument('--ngpu', default=4, type=int, help='gpu number when inference for ref-ytvos and ref-davis')
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether = the masks during inference')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # parser.add_argument('--only_use_vprompt', default=False, action='store_true', help='only_use_vision_prompt')
    # parser.add_argument('--only_use_tprompt', default=False, action='store_true', help='only_use_text_prompt')
    # parser.add_argument('--use_uprompt', default=False, action='store_true', help='use_unify_prompt')
    # parser.add_argument('--use_temporal', default=False, action='store_true', help='use_temporal')
    # parser.add_argument('--num_tprompt', default=37, type=int, help='num_tprompt')
    # parser.add_argument('--num_vprompt', default=10, type=int, help="num_vprompt")

    return parser


