_base_ = [
    'mae_vit-base-p16_1xb64-amp-coslr-800e_reflacx.py'
]

model = dict(
    type='MAE',
    backbone=dict(
        # init_cfg=dict(type='Pretrained',
        #               checkpoint='../preTrain/vit_base_p16_224_timmlab.pth'),
        type='MAEViT', arch='b', patch_size=16, mask_ratio=0.85)
    )