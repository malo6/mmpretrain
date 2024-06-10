_base_ = [
    '../_base_/datasets/reflacx_bs64_selfsup.py',
    # '../_base_/schedules/imagenet_sgd_coslr_200e.py',
    '../_base_/default_runtime.py',
]
max_epochs=400

# model settings
model = dict(
    type='BarlowTwins',
    backbone=dict(
        init_cfg=dict(
                type='Pretrained',
                checkpoint='../preTrain/resnet50-19c8e357.pth',
                ),
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=8192,
        out_channels=8192,
        num_layers=3,
        with_last_bn=False,
        with_last_bn_affine=False,
        with_avg_pool=True,
        init_cfg=dict(
            type='Kaiming', distribution='uniform', layer=['Linear'])),
    head=dict(
        type='LatentCrossCorrelationHead',
        in_channels=8192,
        loss=dict(type='CrossCorrelationLoss')))




# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=4))



# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='SGD', lr=1e-3, weight_decay=1e-4, momentum=0.9),
    optimizer=dict(type='AdamW', lr=2e-5, betas=(0.9, 0.95), weight_decay=1e-6)
    )
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

