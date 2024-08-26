_base_ = [
    'mae_vit-base-p16_1xb64-amp-coslr-800e_mimic-eye.py'
]

model = dict(
    type='MAE',
    backbone=dict(
        type='MAEViT', arch='b', patch_size=16, mask_ratio=0.85)
    )