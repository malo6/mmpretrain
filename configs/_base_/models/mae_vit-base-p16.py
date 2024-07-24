# model settings
model = dict(
    type='MAE',
    backbone=dict(
        init_cfg=dict(type='Pretrained',
                      checkpoint='../preTrain/vit_base_p16_224_timm.pth'),
                    #   checkpoint='../preTrain/vit_b_16-c867db91.pth'),
        # type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75), # 0.9 0.85
        type='GazeMAEViT', arch='b', patch_size=16, mask_ratio=0.80,clinical_ratio=0.5), # 0.9 0.85
        # type='MAEViT', arch='b', patch_size=16, mask_ratio=0.80), # 0.9 0.85
        neck=dict(
        type='',
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
