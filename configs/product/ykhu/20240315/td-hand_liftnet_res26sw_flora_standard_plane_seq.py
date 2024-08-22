# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']
from mmpose.configs._base_.datasets.xs3d import \
    datasets_info as kpt3d_datasets_info

import os
train_cfg = dict(max_epochs=70, val_interval=70)

data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'
seq_length = 4

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
    # clip_grad=dict(max_norm=10, norm_type=2),
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
backbone_out_channels = [48, 96, 192, 384]
model = dict(
    type='TopdownPoseLiftEstimatorSeq',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='ResNet',
        depth='26s',
        in_channels=1,
        stem_channels=32,
        base_channels=48,
        expansion=1,
        out_indices=(3, ),
        strides=(1, 2, 2, 1),
        zero_init_residual=False,
        bias_in_conv=False,
        out_channels=backbone_out_channels),
    head=dict(
        type='RTMCCIPRHead3D',
        in_channels=384,
        out_channels=21,
        input_size=(128, 128, 128),
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2,
        final_layer_kernel_size=3,
        deploy_output='feat',
        output_sigma=False,
        with_gau=False,
        mlp_with_conv=False,
        gau_cfg=dict(
            hidden_dims=128,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', use_target_weight=False),
                dict(type='L1Loss', use_target_weight=False),
            ]),
        decoder=codec),
    kpt3d_lift=dict(
        type='TemporalLiftHeadStandardOri',
        lift_loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', loss_weight=1),  # 3d kpts
                dict(type='L1Loss', loss_weight=1),  # 3d kpts leftcam
                dict(type='L1Loss', loss_weight=1),  # 3d kpts rightcam
                dict(
                    type='PinchLoss',
                    enter_thre=pinch_thre[0] / 1000,
                    exit_thre=pinch_thre[1] / 1000,
                    loss_weight=1),
                dict(
                    type='MPJPAELoss',
                    seq_length=seq_length,
                    loss_weight=1
                )
            ]),
        seq_len=4,
        enhance_static=False,
        baseline=0.135,
    ),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'work_dirs/lift3d_res26sw/td-hand_liftnet_res26sw_flora_standard_plane/best_all_mpjpe_epoch_120.pth'
    ),
)

# base dataset settings
dataset_type = 'PairHand3DDatasetSeq'
data_mode = 'topdown'

train_data_list = []
train_date_list = [
    '20230824', '20230828', '20230906', '20230907', '20231031', '20231227',
    '20240220', '20240229', '20240401', '20240425', '20240517', '20240522',
    '20240801'
]
train_glasses_list = ['Flora301', 'Flora302', 'Flora303', 'Flora304']
for data_date in train_date_list:
    for glasses in train_glasses_list:
        train_data_list += kpt3d_datasets_info['train_data'][data_date].get(
            glasses, [])
train_data_list = [os.path.join(data_root, item) for item in train_data_list]

# train_data_list = [
#   'data_hand/hand_keypoint/annotations3d/simulate_binocular_coco_hand/XS__20230907_033735__pinch__normal__right__1111__0011__undistort_tar__Flora301__15F_office.json',
# ]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',  # xujian 33684 images, 16842 pair instances
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',

    # 边缘半手
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20240508_032300__pinch__normal__right__1101__0006__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20240508_032919__pinch__normal__right__1101__0006__undistort_tar__Flora301.json',

    # 静止
    # 'data_hand/hand_keypoint/annotations3d/fit3d_seqsmooth_auto__binocular_coco/XS__20240516_070512__all__normal__left__1101__0007__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit3d_seqsmooth_auto__binocular_coco/XS__20240516_065421__all__normal__right__1101__0007__undistort_tar__Flora301.json'

    # 黑人
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_054705__all__normal__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_055912__all__bright__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_060453__pinch__bright__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_061312__pinch__bright__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_061915__pinch__normal__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_062634__pinch__dark__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_062946__pinch__dark__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_063405__pinch__normal__right__1111__0023__undistort_tar__Flora301.json'

    # 反向pinch
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083647__pinch__normal__right__1110__0025__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083322__pinch__normal__left__1110__0025__undistort_tar__Flora301.json',
            
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
        seq_len=seq_length,
        data_ratio=1./3,
        serialize_data=True
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
        standard_stereo=True,
        flip_left_to_right=True,
        data_root=data_root,
    ),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='all_mpjpe', rule='less'),
    run_time_info=dict(type='RuntimeInfoHookV2'))

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]

# val_evaluator = dict(type='MPJPEMetricLifting', gesture_list=gesture_list)
val_evaluator = [
    dict(
        type='MPJPEV2',
        mode=['mpjpe', 'p-mpjpe'],
        # mode=['mpjpe'],
        # gesture_list=gesture_list,
        rearrange_result=True,
        pinch_thre=pinch_thre,
        pinch_hard_metric=False,
        # bmk_save_root='/home/ykhu/workspace/mmpose/work_dirs/bad_case_liftnet/pinch0315',
        result_dir='.'),
    # dict(type='EPE'),
    # dict(type='NrealKeypointAP'),
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
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
