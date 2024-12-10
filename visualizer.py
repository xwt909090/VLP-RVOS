'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
from pathlib import Path

import numpy as np

import torchvision.transforms as T
import os
from PIL import Image, ImageDraw
import json


from tools.colormap import colormap


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize((352, 352)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# torch.multiprocessing.set_start_method('spawn')

def main():

    split = 'valid'
    # save path
    output_dir = "./output/64.9-visualization/"

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if not os.path.exists(save_visualize_path_prefix):
        os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path('/workspace/datasets/ref-youtube-vos/') # data/ref-youtube-vos
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

    video_num = len(video_list)
    print('Start Visualize')
    mask_root = Path('./output/Annotations/')  # data/ref-youtube-vos
    mask_folder = mask_root
    sub_processor(data, save_visualize_path_prefix, img_folder, mask_folder, video_list)




def sub_processor(data, save_visualize_path_prefix, img_folder, mask_folder, video_list):

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

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)
            # store images
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                img = Image.open(img_path).convert('RGBA')  # PIL image
                # img = transform(img)

                mask_path = os.path.join(mask_folder, video_name, str(i), frame + ".png")
                mask = Image.open(mask_path)#.convert('RGBA')
                # mask = transform(mask)

                source_img = vis_add_mask(img, mask, color_list[i % len(color_list)])
                # save
                save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                if not os.path.exists(save_visualize_path_dir):
                    os.makedirs(save_visualize_path_dir)
                save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                source_img.save(save_visualize_path)


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    origin_mask = np.asarray(mask).copy()
    color = np.array(color)
    origin_mask = origin_mask.reshape(origin_mask.shape[0], origin_mask.shape[1]).astype('uint8') # np
    origin_mask = origin_mask > 0

    origin_img[origin_mask] = origin_img[origin_mask] * 0.3 + color * 0.7
    origin_img = Image.fromarray(origin_img)
    return origin_img

  

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
