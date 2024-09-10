_base_ = [
    # '../_base_/models/mae_vit-base-p16.py',
    # '../_base_/datasets/reflacx_bs64_mae.py',
    '../_base_/default_runtime.py',
]

max_epochs=800


# dataset settings
dataset_type = 'Reflacx'
#修改为绝对路径
data_root = '/public_bme/data/reflacx-1.0.0/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=False,
         with_label=False,
         with_seg=True),
    dict(type='Resize', scale=(224,224), backend="pillow"),
    dict(type='ToTensor', keys=["gt_seg_map"]),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs',
         algorithm_keys=["gt_seg_map"])
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline))



# model settings
model = dict(
    type='GazeMAE',
    backbone=dict(
        init_cfg=dict(type='Pretrained',
                      checkpoint='../preTrain/vit_base_p16_224_timmlab.pth'),
        type='GazeMAEViT', arch='b', patch_size=16, mask_ratio=0.85, clinical_ratio=0.5,preload=False), # 0.9 0.85
         neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])




# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=360,
        by_epoch=True,
        begin=40,
        end=max_epochs,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=100, max_keep_ckpts=8),
    logger=dict(type='LoggerHook', interval=10),
    )

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

