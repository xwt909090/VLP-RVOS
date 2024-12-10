'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model_clip as build_model_clip
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json

import args_setting
from tqdm import tqdm

import multiprocessing as mp
import threading

from tools.colormap import colormap


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def main(args, cfg, epoch):
    args.masks = True
    cfg.DATALOADER.TEST.BATCH_SIZE == 1
    print("Inference only supports for batch size = 1")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    split = args.split
    # save path
    output_dir = args.output_dir
    repeat = ""
    if args.exp:
        repeat = "_exp" + args.exp
    output_dir = os.path.join(output_dir, args.experiment_name + repeat, 'results')
    save_path_prefix = os.path.join(output_dir, split+str(epoch).zfill(4))
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split+str(epoch).zfill(4) + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path(args.ytvos_path) # data/ref-youtube-vos
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # for some reasons the competition's validation expressions dict contains both the validation (202) &
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    # build transform
    img_size = cfg.INPUT.IMG_SIZE
    to_size = (img_size, img_size)
    transform = T.Compose([
        T.Resize(to_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_visualize = T.Compose([
        T.Resize(to_size)
    ])

    # create subprocess
    thread_num = args.ngpu
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, cfg, data, transform, transform_visualize,
                                                   save_path_prefix, save_visualize_path_prefix,
                                                   img_folder, sub_video_list, epoch))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))

def sub_processor(lock, pid, args, cfg, data, transform, transform_visualize, save_path_prefix, save_visualize_path_prefix, img_folder, video_list, epoch):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    # model
    model, criterion, _ = build_model_clip(args, cfg, batch_size=1)
    # model, criterion, _ = build_model_clip(args, cfg)
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(model)
    '''for k, v in model.named_parameters():
        if 'clip_model.visual.transformer.resblocks.0' in k:
            print(v)
    exit(0)'''
    if pid == 0:
        print('number of params:', n_parameters)
    repeat = ""
    if args.exp:
        repeat = "_exp" + args.exp
    resume_path = os.path.join(args.output_dir, args.experiment_name + repeat, "checkpoint"+str(epoch).zfill(4)+".pth")
    if os.path.exists(resume_path):
        print('resume checkpoint from: {}'.format(resume_path))
    else:
        raise ValueError('Please specify the checkpoint for inference.')
    checkpoint = torch.load(resume_path, map_location='cpu')
    state_dict = checkpoint['model']
    # for key in list(state_dict.keys()):
            # if "temporal_pe" in key:
                # del state_dict[key]
            # if "relative_position_index" in key:
                # del state_dict[key]
            # if "relative_position_bias_table" in key:
                # del state_dict[key]
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
    # print(missing_keys)
    # print(unexpected_keys)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))


    # start inference
    num_all_frames = 0
    model.eval()

    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, args.experiment_name + repeat, 'results')
    simi_save_visualize_path_dir_prefix = os.path.join(output_dir, 'valid' + str(epoch).zfill(4) + '_simi')
    if not os.path.exists(simi_save_visualize_path_dir_prefix):
        os.makedirs(simi_save_visualize_path_dir_prefix)

    all_time = 0
    all_frame = 0

    # 1. For each video
    for video in video_list:
        metas = [] # list[dict], length is number of expressions

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas
        num_frames = cfg.DATALOADER.TEST.NUM_FRAMES
        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            #print("frames=", frames)

            video_len = len(frames)

            all_frame += video_len

            # store images
            imgs = []
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                img = Image.open(img_path).convert('RGB')
                origin_w, origin_h = img.size
                imgs.append(transform(img)) # list[img]

            imgs = torch.stack(imgs, dim=0).to(args.device) # [video_len, 3, h, w]
            img_h, img_w = imgs.shape[-2:]
            size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
            target = {"size": size}

            # print("num_frames=", num_frames)
            b = 1
            prompt_token = None
            #print("imgs.len = {}, exp.len = {}, target.len = {}".format(len(imgs), len(exp), len(target)))
            for frame_start in range(0, video_len, num_frames):
                frame_end = frame_start + num_frames if frame_start + num_frames <= video_len else video_len
                frames_tmp = frames[frame_start:frame_end]
                num_frames_tmp = frame_end - frame_start
                # print("num_frames_tmp=", num_frames_tmp)
                t = num_frames_tmp
                with torch.no_grad():
                    time1 = time.time()
                    outputs, _ = model([imgs[frame_start:frame_end, :, :, :]], [exp], [target], video_prompt=prompt_token)
                    time2 = time.time()
                    all_time = all_time + time2 - time1

                    print("mid  FPS:  ", (frame_end - frame_start) / (time2-time1))
                    
                    prompt_token = outputs['prompt_token']
                    pred_masks = outputs["pred_masks"][0]
                    pred_masks = pred_masks[range(num_frames_tmp), ...] # [t, h, w]
                    pred_masks = pred_masks.unsqueeze(0)
                    pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()

                    all_pred_masks = pred_masks


                    # similarity = outputs['similarity']
                    # grid_size = outputs['grid_size']
                    # #             decoder_t_attn_map = outputs['decoder_t_attn_map']
                    # [grid_h, grid_w] = grid_size
                    # similarity = similarity.squeeze(2).view(b * t, grid_h, grid_w)
                    # similarity = similarity.view(b, t, grid_h, grid_w)[0]
                    # scale_factor = \
                    # {'ViT-B/32': 32, 'ViT-B/16': 16, 'RN50': 32, 'RN50x4': 32, 'RN50x16': 32, 'RN50x64': 32}[
                    #     args.version]

                    # similarity_upsampled = F.interpolate(similarity.unsqueeze(1).float(), scale_factor=scale_factor,
                    #                                      mode='bilinear', align_corners=True)
                    # similarity_upsampled = similarity_upsampled.squeeze(1)
                    # # similarity_upsampled = similarity_upsampled.unsqueeze(0)
                    # # similarity_upsampled = (similarity_upsampled.sigmoid() > args.threshold).squeeze(0)
                    # similarity_upsampled = similarity_upsampled.detach().cpu().numpy()
                    # all_simi_upsampled = similarity_upsampled


                if args.visualize:
                    for t, frame in enumerate(frames_tmp):
                        # original
                        img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                        source_img = Image.open(img_path).convert('RGBA') # PIL image
                        source_img = transform_visualize(source_img)
                        #draw = ImageDraw.Draw(source_img)
                        #draw_boxes = all_pred_boxes[t].unsqueeze(0)
                        #draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()

                        # draw boxes
                        #xmin, ymin, xmax, ymax = draw_boxes[0]
                        #draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[i%len(color_list)]), width=2)

                        # draw reference point
                        #ref_points = all_pred_ref_points[t].unsqueeze(0).detach().cpu().tolist()
                        #draw_reference_points(draw, ref_points, source_img.size, color=color_list[i%len(color_list)])

                        # draw mask
                        normed_source_img = np.array(source_img.convert('RGB'))
                        source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[i%len(color_list)])

                        # save
                        save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                        if not os.path.exists(save_visualize_path_dir):
                            os.makedirs(save_visualize_path_dir)
                        save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                        source_img.save(save_visualize_path)

                        normed_source_img = normed_source_img / (np.max(np.max(normed_source_img)) + 1e-8)
                        normed_source_img = np.uint8(255 * normed_source_img)
                        normed_source_img = np.ascontiguousarray(normed_source_img, dtype=np.uint8)
                        # simi_upsampled = all_simi_upsampled[t]
                        # # simi_upsampled = vis_add_mask(source_img, simi_upsampled, color_list[0])
                        # # simi_upsampled = Image.fromarray(simi_upsampled * 255).convert('L')
                        # normed_simi_upsampled = (simi_upsampled - np.min(simi_upsampled)) / (
                        #             np.max(simi_upsampled) - np.min(simi_upsampled) + 1e-8)
                        # normed_simi_upsampled = np.uint8(255 * normed_simi_upsampled)
                        # normed_simi_upsampled = cv2.applyColorMap(normed_simi_upsampled, cv2.COLORMAP_JET)
                        # normed_simi_upsampled = normed_simi_upsampled[:, :, ::-1]
                        # normed_simi_upsampled = cv2.addWeighted(normed_source_img, 0.5, normed_simi_upsampled, 0.5, 0)


                        # simi_save_visualize_path_dir = os.path.join(simi_save_visualize_path_dir_prefix, video, str(i))
                        # if not os.path.exists(simi_save_visualize_path_dir):
                        #     os.makedirs(simi_save_visualize_path_dir)
                        # simi_save_visualize_path_dir = os.path.join(simi_save_visualize_path_dir, frame + '.png')

                        # normed_simi_upsampled = cv2.resize(normed_simi_upsampled, dsize=(640, 360),
                        #                                    interpolation=cv2.INTER_CUBIC)
                        # normed_simi_upsampled = Image.fromarray(normed_simi_upsampled)
                        # normed_simi_upsampled.save(simi_save_visualize_path_dir)


                # save binary image
                save_path = os.path.join(save_path_prefix, video_name, exp_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for j in range(num_frames_tmp):
                    frame_name = frames_tmp[j]
                    #print("all_pred_masks.shape=", all_pred_masks.shape)        #(3, 368, 640)
                    mask = all_pred_masks[j].astype(np.float32)
                    #print("mask.shape=", mask)        #(368, 640)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frame_name + ".png")
                    mask.save(save_file)

            print("FPS:  ", all_frame / all_time)

        with lock:
            progress.update(1)
    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x-2, y-2, x+2, y+2),
                            fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.3 + color * 0.7
    origin_img = cv2.resize(origin_img, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)
    origin_img = Image.fromarray(origin_img)
    return origin_img



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer inference script', parents=[ouropts.get_args_parser()])
    args = parser.parse_args()
    test_start = args.test_start
    test_end = args.test_end
    for epoch in range(test_start, test_end+1):
        main(args, epoch)
