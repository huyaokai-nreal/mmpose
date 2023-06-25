# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']

# runtime
max_epochs = 50
stage2_num_epochs = 10
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# codec settings
codec = dict(
    type='SimCCLabel3D',
    input_size=(256, 256, 256),
    sigma=(5.66, 5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=True)

# model settings
model = dict(
    type='TopdownPose3DEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead3D',
        in_channels=768,
        out_channels=21,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss3D',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False, ))

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]

visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_hooks = dict(
    visualization=dict(
        type='PoseVisualizationHook', enable=True, draw_3d=True),
    checkpoint=dict(save_best='AUC', rule='greater', max_keep_ckpts=1))
# base dataset settings
dataset_type = 'InterHand3DDataset'
data_mode = 'topdown'
data_root = '/data/hand_group/data/data_hand/hand_keypoint/public_data/InterHand2.6M/InterHand2.6M_5fps_batch1/'
backend_args = dict(backend='local')
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/',
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/'
#     }))

# pipelines
train_pipeline = [
    dict(type='LoadImageFromMultiLMDB', color_type='color'),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=45,
        rotate_prob=0.6,
        shift_prob=0.3,
        shift_factor=0.16),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImageFromMultiLMDB', color_type='color'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromMultiLMDB', color_type='color'),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=180),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        ann_file=f'{data_root}/annotations/all/InterHand2.6M_train_data.json',
        camera_file=
        f'{data_root}/annotations/all/InterHand2.6M_train_camera.json',
        joint_file=
        f'{data_root}/annotations/all/InterHand2.6M_train_joint_3d.json',
        img_prefix=f'{data_root}/lmdb_data/train_lmdb',
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        ann_file=
        f'{data_root}/annotations/machine_annot/InterHand2.6M_val_data.json',
        camera_file=
        f'{data_root}/annotations/machine_annot/InterHand2.6M_val_camera.json',
        joint_file=
        f'{data_root}/annotations/machine_annot/InterHand2.6M_val_joint_3d.json',
        img_prefix=f'{data_root}/lmdb_data/val_lmdb',
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        ann_file=f'{data_root}/annotations/all/InterHand2.6M_test_data.json',
        camera_file=
        f'{data_root}/annotations/all/InterHand2.6M_test_camera.json',
        joint_file=
        f'{data_root}/annotations/all/InterHand2.6M_test_joint_3d.json',
        img_prefix=f'{data_root}/lmdb_data/test_lmdb',
        test_mode=True,
        pipeline=val_pipeline,
    ))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = [
    dict(type='MPJPEV2'),
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE')
]
test_evaluator = val_evaluator
