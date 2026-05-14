num_things_classes = 0
num_stuff_classes = 2
num_classes = 2
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderMask2Former',
    pretrained='pretrained/hibou_l_vit_large_patch16_for_vitadapter.npz',
    backbone=dict(
        type='ViTAdapter',
        pretrain_size=224,
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_type='swiglufused',
        ffn_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        window_attn=[
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False
        ],
        window_size=[
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None
        ],
        with_cp=True,
        pretrained='pretrained/hibou_l_vit_large_patch16_for_vitadapter.npz'),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[1024, 1024, 1024, 1024],
        feat_channels=256,
        out_channels=256,
        in_index=[0, 1, 2, 3],
        num_things_classes=0,
        num_stuff_classes=5,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(
            panoptic_on=True,
            semantic_on=False,
            instance_on=True,
            max_per_image=100,
            iou_thr=0.8,
            filter_low_score=True,
            mode='whole')),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True,
        mode='whole'),
    init_cfg=None)
dataset_type = 'HistoPathTestsDataset'
data_root = 'data/pathtests_ds2_fold1'
img_norm_cfg = dict(
    mean=[180.341865, 147.576273, 179.422335],
    std=[54.029147, 58.67915, 45.266428],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(
        type='MapIgnoreToBackground',
        ignore_index=255,
        bg_index=0,
        max_valid_label=4),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[180.341865, 147.576273, 179.422335],
        std=[54.029147, 58.67915, 45.266428],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='MapIgnoreToBackground',
                ignore_index=255,
                bg_index=0,
                max_valid_label=4),
            dict(
                type='Normalize',
                mean=[180.341865, 147.576273, 179.422335],
                std=[54.029147, 58.67915, 45.266428],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='HistoPathTestsDataset',
        data_root=
        '/leonardo_work/AIFAC_F02_042/vit_adapter_data/pathtests_1mpp_fold1_fgonly',
        img_dir='images/train',
        ann_dir='masks/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(
                type='MapIgnoreToBackground',
                ignore_index=255,
                bg_index=0,
                max_valid_label=4),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[180.341865, 147.576273, 179.422335],
                std=[54.029147, 58.67915, 45.266428],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='ToMask'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
        ]),
    val=dict(
        type='HistoPathTestsDataset',
        data_root=
        '/leonardo_work/AIFAC_F02_042/vit_adapter_data/pathtests_1mpp_fold1_fgonly',
        img_dir='images/val',
        ann_dir='masks/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='MapIgnoreToBackground',
                        ignore_index=255,
                        bg_index=0,
                        max_valid_label=4),
                    dict(
                        type='Normalize',
                        mean=[180.341865, 147.576273, 179.422335],
                        std=[54.029147, 58.67915, 45.266428],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='HistoPathTestsDataset',
        data_root=
        '/leonardo_work/AIFAC_F02_042/vit_adapter_data/pathtests_1mpp_fold1_fgonly',
        img_dir='images/val',
        ann_dir='masks/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='MapIgnoreToBackground',
                        ignore_index=255,
                        bg_index=0,
                        max_valid_label=4),
                    dict(
                        type='Normalize',
                        mean=[180.341865, 147.576273, 179.422335],
                        std=[54.029147, 58.67915, 45.266428],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)
evaluation = dict(
    interval=2000,
    metric=['mIoU', 'mDice'],
    pre_eval=True,
    save_best='mean_micro_dice_1_4',
    rule='greater')
pretrained = 'pretrained/hibou_l_vit_large_patch16_for_vitadapter.npz'
work_dir = '/leonardo_work/AIFAC_F02_042/work_dirs/vit_adapter_hibou_mask2former_1mpp_fold1_v1'
gpu_ids = range(0, 1)
auto_resume = False
device = 'cuda'
seed = 1333213631
