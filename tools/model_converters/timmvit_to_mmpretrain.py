# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_timmvit(ckpt):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('patch_embed'):
            new_k = k.replace('proj', 'projection')
        elif k.startswith('blocks'):
            layer_num=eval(k.split('.')[1])
            new_k='layers'+k[len("blocks"):]
            new_k=new_k.replace('.norm','.ln').replace('.mlp.fc1.','.ffn.layers.0.0.').replace('.mlp.fc2','.ffn.layers.1')
            # print("hold")
        elif k.startswith('norm'):
            new_k=k.replace('norm','ln1')
        else:
            new_k = k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained timm ViT '
        'models to mmpretrain style.')
    parser.add_argument('--src', type=str,default="../preTrain/vit_base_p16_224_timm.pth",help='src model path or url')

    parser.add_argument('--dst',type=str, default="../preTrain/vit_base_p16_224_timmlab.pth",help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_timmvit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()
