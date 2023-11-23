# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']
train_cfg = dict(max_epochs=100, val_interval=100)
from mmpose.configs._base_.datasets.xs3d import \
    datasets_info as kpt3d_datasets_info

data_root = '/data/AI_DATA_WX'
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4, weight_decay=1e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'head': dict(lr_mult=0.0, decay_mult=0.0),
            'neck': dict(lr_mult=0.0, decay_mult=0.0),
        }),
)

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=5,
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

# codec settings
codec = dict(
    type='IntegralRegressionLabel',
    input_size=(128, 128),
    heatmap_size=(32, 32),
    sigma=1,
    normalize=False,
    blur_kernel_size=5,
)

pinch_thre = [20, 40]  # pinch双阈值，单位：mm
# model settings
backbone_out_channels = [64, 96, 128, 160]
model = dict(
    type='TopdownPoseLiftEstimatorSeq',
    seq_len=32,
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='ResNet',
        depth=26,
        in_channels=1,
        stem_channels=64,
        base_channels=32,
        expansion=1,
        out_indices=(0, 1, 2, 3),
        zero_init_residual=False,
        bias_in_conv=False,
        out_channels=backbone_out_channels),
    neck=dict(
        type='FPN',
        in_channels=backbone_out_channels,
        out_channels=192,
        num_outs=4,
        upsample_cfg=dict(mode='bilinear', align_corners=True),
        upsample_style='rsn',
        norm_cfg=dict(type='BN'),
        reverse_output=True,
        apply_fpn_conv=False),
    head=dict(
        type='DSNTHead',
        in_channels=192,
        deconv_out_channels=(),
        feat_norm_type='softmax',
        in_featuremap_size=(32, 32),
        num_joints=21,
        decoder=codec,
        deploy=False,
        output_sigma=True),
    kpt3d_lift=dict(
        type='LiftHeadSeqTest',
        lift_loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss'),  # 3d kpts
                dict(type='L1Loss'),  # 3d kpts leftcam
                dict(type='L1Loss'),  # 3d kpts rightcam
                dict(type='MSELoss', loss_weight=0),  # 2d reprojection left
                dict(type='MSELoss', loss_weight=0),  # 2d reprojection right
                # dict(
                #     type='PinchLoss',
                #     enter_thre=pinch_thre[0] / 1000,
                #     exit_thre=pinch_thre[1] / 1000,
                #     loss_weight=0.09),
                # dict(type='L1Loss', loss_weight=0),  # major kpt
            ]),
        seq_len=1,
        channel_num=55,
        output_num=42,
        undistort=True),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ))

# base dataset settings
dataset_type = 'PairHand3DDatasetSeq'
data_mode = 'topdown'

import os

# lmdb root dir, maybe different between beijing and wuxi

# test only
# data_root = '/data/hand_group/data/data_hand/lmdb_data/'
train_data_list = [
    # 0824
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora302.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora304.json',

    #0828
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_062006__point__normal__right__1000__0005__quest__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_062640__all__normal__right__1000__0005__quest__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_063719__pinch__normal__right__1000__0005__quest__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora302.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora303.json',
    'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora304.json',
]

train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)
val_data_list = []
val_date_list = ['20230830']
val_glasses_list = ['Flora301']
val_person_list = ['0005']

for data_date in val_date_list:
    for glasses in val_glasses_list:
        val_data_list += kpt3d_datasets_info['test_data'][data_date].get(
            glasses, [])
val_data_list = [
    item for item in val_data_list if item.split('__')[-3] in val_person_list
]
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
# pipelines
train_pipeline = [
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
        enable_epoch_num=40),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='GenerateTarget',
        target_type='heatmap+keypoint_label',
        encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        seq_len=4,
    ),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(
        type='DistributedRangeSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='PairHand3DDataset',
        data_file_list=val_data_list,
        data_mode=data_mode,
        test_mode=True,
        pipeline=val_pipeline,
        flip_left_to_right=True,
        data_root=data_root,
    ),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='all_mpjpe', rule='less'),
    run_time_info=dict(type='RuntimeInfoHookV2'),
    visualization=dict(
        type='PoseVisualizationHook', enable=True, draw_3d=False),
)

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]
# val_evaluator = dict(type='MPJPEMetricLifting', gesture_list=gesture_list)
val_evaluator = [
    dict(
        type='MPJPEV2',
        mode=['mpjpe', 'p-mpjpe'],
        gesture_list=gesture_list,
        rearrange_result=True,
        result_dir='.'),
    dict(type='EPE'),
    dict(type='NrealKeypointAP'),
]
test_evaluator = val_evaluator

# fp16 settings
fp16 = dict(loss_scale='dynamic')
# model wrapper
find_unused_parameters = True

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # this will slow the training process ???
    # dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
