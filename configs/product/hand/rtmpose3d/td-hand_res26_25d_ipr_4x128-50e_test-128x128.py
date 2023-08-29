# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']
# runtime
train_cfg = dict(max_epochs=100, val_interval=5)

data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'

base_lr = 1e-4
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=10,
        start_factor=0.001,
        end_factor=1.0,
        by_epoch=True,
        convert_to_iter_based=True),  # warm-up
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=train_cfg['max_epochs'],
        convert_to_iter_based=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=128)

codec = dict(
    type='RegressionLabel',
    input_size=(128, 128, 128),
    with_depth=True,
    depth_bound=0.4)

# model settings
backbone_out_channels = [64, 96, 128, 160]
model = dict(
    type='TopdownPose3DEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='ResNet',
        depth=26,
        in_channels=1,
        stem_channels=64,
        base_channels=32,
        expansion=1,
        out_indices=(3, ),
        zero_init_residual=False,
        bias_in_conv=False,
        out_channels=backbone_out_channels),
    head=dict(
        type='RTMCCIPRHead3D',
        in_channels=160,
        out_channels=21,
        input_size=codec['input_size'],
        in_featuremap_size=(4, 4),
        simcc_split_ratio=2,
        final_layer_kernel_size=7,
        output_sigma=False,
        gau_cfg=dict(
            hidden_dims=128,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        #loss=dict(
        #    type='RLELoss',
        #    use_target_weight=False,
        #    dim=3,
        #),
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', use_target_weight=False),
                dict(type='L1Loss', use_target_weight=False)
            ]),
        decoder=codec),
    test_cfg=dict(flip_test=False, ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'work_dirs/td-hand_res26_25d_4x128-50e_test-128x128/epoch_100.pth'),
    root_mode='optimize',
)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]

visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_hooks = dict(
    #visualization=dict(
    #    type='PoseVisualizationHook', enable=True, draw_3d=True),
    checkpoint=dict(save_best='1/all_p-mpjpe', rule='less'))
# base dataset settings
backend_args = dict(backend='local')
train_pipeline = [
    dict(type='KeypointTo25DLabel'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='RandomBrightnessContrast', p=0.2),
        ]),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=15,
        rotate_prob=0.3,
        shift_prob=0.5,
        shift_factor=0.2,
        enable_epoch_num=80),
    dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='KeypointTo25DLabel'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

dataset_type = 'PairHand3DDataset'
data_mode = 'topdown'

import os

# lmdb root dir, maybe different between beijing and wuxi

# test only
#data_root = '/data/hand_group/data/data_hand/lmdb_data/'
train_data_list = [
    'data_hand/hand_keypoint/annotations3d/Flora8/XS__all__normal__left__1000__0006__20230810_081421__undistort_tar__Flora8.json',
    'data_hand/hand_keypoint/annotations3d/Flora8/XS__all__normal__right__1000__0006__20230810_081258__undistort_tar__Flora8.json',
    'data_hand/hand_keypoint/annotations3d/Flora8/XS__all__normal__right__1000__0010__20230810_080650__undistort_tar__Flora8.json',
    'data_hand/hand_keypoint/annotations3d/Flora8/XS__all__normal__left__1000__0007__20230810_081633__undistort_tar__Flora8.json',
    'data_hand/hand_keypoint/annotations3d/Flora8/XS__all__normal__right__1000__0006__20230810_082257__undistort_tar__Flora8.json',
    'data_hand/hand_keypoint/annotations3d/Flora8/XS__all__normal__right__1000__0011__20230810_081042__undistort_tar__Flora8.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0006__20230809_055248__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0007__20230809_090503__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0006__20230809_055716__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0007__20230809_101132__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0006__20230809_054849__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0006__20230809_060122__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0007__20230809_060812__undistort_tar__Flora301.json',
]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0007__20230809_100619__undistort_tar__Flora301.json',
]
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_ratio=2,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        flip_left_to_right=False,
        point_type='2.5D'
        # indices=1000,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_file_list=val_data_list,
        data_mode=data_mode,
        test_mode=True,
        pipeline=val_pipeline,
        flip_left_to_right=False,
        point_type='2.5D',
        data_root=data_root))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='MPJPEV2', mode='mpjpe'),
    dict(type='MPJPEV2', mode='p-mpjpe', prefix='1'),
    dict(type='EPE'),
    dict(type='NrealKeypointAP')
]
test_evaluator = val_evaluator
