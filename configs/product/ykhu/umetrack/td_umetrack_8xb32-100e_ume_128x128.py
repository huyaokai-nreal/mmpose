# flake8: noqa
import os

from mmpose.configs._base_.datasets.xs3d_nimble import \
    datasets_info as kpt3d_datasets_info

_base_ = ['../../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=100, val_interval=5)

data_root = '/data/AI_DATA_WX'

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4), )

# 梯度裁剪
clip_grad = dict(max_norm=50, norm_type=2)
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

pinch_thre = [20, 40]  # pinch双阈值，单位：mm
# model settings
backbone_out_channels = [32, 64, 128, 256]
model = dict(
    type='TopdownPoseUmeNimbleEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='ResNet',
        depth='26s',
        in_channels=1,
        stem_channels=32,
        base_channels=32,
        expansion=1,
        attention=None,  # None, SEBlock, CBAM, ECA, BAM, NonLocalBlock
        out_indices=(3, ),
        strides=(1, 2, 2, 1),
        zero_init_residual=False,
        bias_in_conv=False,
        out_channels=backbone_out_channels),
    head=dict(
        type='UmeHead',
        ume_loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # root pred
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # local pred
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # all pred
                dict(
                    type='PinchLoss',
                    enter_thre=pinch_thre[0] / 1000,
                    exit_thre=pinch_thre[1] / 1000,
                    loss_weight=1,
                    enable_start_epoch=train_cfg['max_epochs'] // 2),
                dict(type='MSELoss', loss_weight=1),  # nimble trans直接监督
                dict(
                    type='RLELoss',
                    dim=3,
                    enable_start_epoch=train_cfg['max_epochs']),
                dict(
                    type='L1Loss',
                    use_target_weight=True,
                    enable_start_epoch=train_cfg['max_epochs'] // 2,
                    loss_weight=0),
            ]),
        use_svd=True,
        use_gmlp=False,
        pose_ncomp=30,
        baseline=0.135,
        reg_shape_type=0,
        enhance_static=False,
        enhance_lefthand=False,
        use_6d_pose_reg=False,
        use_9d_pose_reg=True),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/data/AI_DATA/data_hand/model/mmpose/all_decouple_pca_standard_total_res26s_aug2d/epoch_150.pth'
    ),
)

# base dataset settings
data_mode = 'topdown'

train_data_list = []
train_date_list = [
    '20230824', '20230828', '20230906', '20230907', '20231227', '20240220',
    '20240229', '20240401', '20240517', '20240425', '20240522'
]
train_glasses_list = ['Flora301', 'Flora302', 'Flora303', 'Flora304']
for data_date in train_date_list:
    for glasses in train_glasses_list:
        train_data_list += kpt3d_datasets_info['train_data'][data_date].get(
            glasses, [])
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
ume_data_root = '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/ume_data/training/'
ume_user_list = os.listdir(ume_data_root)
for user in ume_user_list:
    user_path = os.path.join(ume_data_root, user)
    file_list = os.listdir(user_path)
    for file in file_list:
        train_data_list.append(os.path.join(user_path, file))

# train_data_list = [
#     '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/ume_data/training/user_15/recording_00.json',
#     '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240229_085520__pinch__normal__left__1110__0029__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240229_085836__pinch__normal__right__1110__0029__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_062729__pinch__normal__right__1110__0006__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_063058__pinch__normal__left__1110__0006__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_063800__pinch__normal__left__1110__0007__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_064131__pinch__normal__right__1110__0007__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_064839__pinch__normal__left__1110__0014__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_065926__pinch__normal__right__1110__0014__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_070606__pinch__normal__right__1110__0008__undistort_tar__Flora301.json',
# #     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_070606__pinch__normal__right__1110__0008__undistort_tar__Flora301.json',
# ]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    # '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/ume_data/training/user_15/recording_00.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',  #
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
]
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
_input_size = (128, 128)
# pipelines
train_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            # dict(
            #     type='RandomBBoxTransform',
            #     scale_factor=[0.75, 1.25],
            #     rotate_factor=15,
            #     rotate_prob=0.3,
            #     shift_prob=0.5,
            #     shift_factor=0.2),
            dict(type='UmePCL', input_size=_input_size),
            # dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.2),
            # dict(type='MixTwoHands', prob=0.),
            # dict(
            #     type='Albumentation',
            #     transforms=[
            #         dict(
            #             type='CoarseDropout',
            #             p=0.2,
            #             max_holes=2,
            #             max_height=16,
            #             max_width=16,
            #         ),
            #     ]),
            dict(
                type='GenerateNoiseDarkImage',
                prob=0.65,
                gamma_limit=(0.85, 0.95),
                alpha_limit=(0.2, 0.5),
                concat_image=False),
        ],
        enable_epoch_num=int(train_cfg['max_epochs'])),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='UmePCL', input_size=_input_size),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='PairHand3DDataset',
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        flip_left_to_right=True,
        data_root=data_root,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='PairHand3DDataset',
        data_file_list=val_data_list,
        data_mode=data_mode,
        test_mode=True,
        pipeline=val_pipeline,
        flip_left_to_right=True,
        data_root=data_root),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='all_mpjpe', rule='less'),
    run_time_info=dict(type='RuntimeInfoHookV2'))

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]
# val_evaluator = dict(type='MPJPEMetricLifting', gesture_list=gesture_list)
filter_exceed = False
val_evaluator = [
    dict(
        type='MPJPEV2',
        mode='mpjpe',
        # gesture_list=gesture_list,
        scale_metric=False,
        fit_metric=False,
        openhand_metric=False,
        pinch_hard_metric=False,
        category_metric=False,
        score_metric=False,
        # bmk_save_root='/data/stliu/mmpose_new/mmpose/work_dirs/result_20231203/bad_case',
        # show_bmk_thr=(50, 10000000),
        filter_exceed=filter_exceed),  #bad case mpjpe thr (mm)
    # dict(type='MPJPEV2', mode='p-mpjpe', prefix='1'),
    # dict(type='EPE', filter_exceed=filter_exceed),
    # dict(type='NrealKeypointAP', filter_exceed=filter_exceed)
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
