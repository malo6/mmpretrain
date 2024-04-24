_base_ = [
    '../_base_/datasets/reflacx_bs64_mocov2.py',
    # '../_base_/schedules/imagenet_sgd_coslr_200e.py',
    '../_base_/default_runtime.py',
]
max_epochs=400
# model settings
model = dict(
    type='MoCo',
    queue_len=1280,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        init_cfg=dict(
                type='Pretrained',
                checkpoint='../preTrain/resnet50-19c8e357.pth',
                ),
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='BN'),
        zero_init_residual=False),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)


# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-3, weight_decay=1e-4, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=200, by_epoch=True, begin=0, end=max_epochs)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)



default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=100),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=10),
                     )

