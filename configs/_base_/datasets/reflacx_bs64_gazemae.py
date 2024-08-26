# dataset settings
dataset_type = 'Reflacx'
data_root = '/public_bme/data/reflacx-1.0.0/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile',
         algorithm_keys=["attention_path"],
         ),
    dict(type='Resize', size=256, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs',
         algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
         )
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
