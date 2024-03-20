# flake8: noqa
import os

# from configs._base_.datasets.xs3d import datasets_info as kpt3d_datasets_info

_base_ = ['../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=100, val_interval=5)

data_root = '/data/AI_DATA'
# data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4, weight_decay=1e-4),
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
# learning policy
# param_scheduler = [
#     dict(  # scheduler
#         type='MultiStepLR',
#         begin=0,
#         end=100,
#         milestones=[50, 90],
#         gamma=0.1,
#         by_epoch=True)
# ]

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

pinch_thre = [15, 35]  # pinch双阈值，单位：mm
kpt2d_with_depth = False  # liftnet 是否使用2.5d的深度信息
standard_stereo = True  # 是否转换标准双目
# model settings
backbone_out_channels = [64, 96, 128, 160]
model = dict(
    type='TopdownPoseLiftNimbleEstimator',
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
        output_depth=True,
        decoder=codec,
        deploy=False,
        output_sigma=False),
    kpt3d_lift=dict(
        type='LiftNimbleHeadStandard',
        lift_loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', loss_weight=1),  # 3d kpts
                dict(type='L1Loss', loss_weight=1),  # 3d kpts leftcam
                dict(type='L1Loss', loss_weight=1),  # 3d kpts rightcam
                dict(
                    type='MSELoss',
                    loss_weight=0,
                    enable_start_epoch=train_cfg['max_epochs'] -
                    40),  # 2d reprojection left
                dict(
                    type='MSELoss',
                    loss_weight=0,
                    enable_start_epoch=train_cfg['max_epochs'] -
                    40),  # 2d reprojection right
                dict(
                    type='PinchLoss',
                    enter_thre=pinch_thre[0] / 1000,
                    exit_thre=pinch_thre[1] / 1000,
                    loss_weight=2,
                    enable_start_epoch=train_cfg['max_epochs'] -
                    20),  # 后20 epoch打开pinch loss
                dict(
                    type='L1Loss',
                    loss_weight=0,
                    enable_start_epoch=train_cfg['max_epochs'] -
                    40),  # xyz比例损失
                dict(
                    type='MSELoss',
                    loss_weight=0,
                    enable_start_epoch=train_cfg['max_epochs'] -
                    40),  # nimble pose直接监督
                dict(type='MSELoss', loss_weight=5),  # nimble trans直接监督
            ]),
        all_use_kp2d_gt=False,
        kpt2d_with_depth=kpt2d_with_depth,
        undistort=True,
        use_svd=True,
        lambda_t=train_cfg['max_epochs'],
        pose_ncomp=30,
        euler_or_quaternion='euler',
        disparity_input=False,
        use_plane_coord=True,
        baseline=0.135,
        use_6d_pose_reg=False,
        reproj_thre=440,
        iou_thre=0.5,
        pad_2d=False,
        edge_to_center=False),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        # '/home/zx_li/workspace/mmpose/work_dirs/td-hand_res26_fpn_sk_weightdata_4xb64-50e_0919data-128x128/epoch_50.pth'
        # '/data/AI_DATA/data_hand/model/mmpose/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128/epoch_50.pth'  # ella kp2d
        # '/home/jrchen/git-project/mmpose/work_dirs/pair_hand3d/003_td-stage_two_train_55dim_l1/epoch_60_new.pth'
        # '/home/jrchen/git-project/mmpose/work_dirs/hand_2d_keypoint/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128_pretrainmodel/epoch_30_FT.pth'
        #f'/data/AI_DATA/jrchen/git-project/mmpose/work_dirs/pair_hand3d/006_td-stage_two_train_55dim_RLE_head_train_flora_finetune/FT_kp2d_add_ella_pretrain_lift.pth'  # flora kp2d + ella kp3d
        # '/home/zx_li/workspace/mmpose/work_dirs/td-hand_rsn26_fpn_25d_ipr_right_pcl_2d3d_4x128-100e-128x128/epoch_100.pth',
        # 'work_dirs/keypoint25d/01_td-hand_rsn26_fpn_25d_ipr_right_pcl_2d3d_4x128-100e-128x128_Affine/best_all_p-mpjpe_epoch_90.pth'
        # '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn26_fpn_25d_scale_ipr_right_2d3d_1031data_simu_4x128-100e-128x128/epoch_100.pth'
        '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn26_fpn_25d_scale_ipr_right_2d3d_handmix_aug_240229data_4x128-100e-128x128/epoch_100.pth'
    ),
)

# base dataset settings
dataset_type = 'PairHand3DDataset'
data_mode = 'topdown'

train_data_list = []
annotations3d_base_path = '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/'
annotations3d_fixed_base_path = '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/'

annotations3d_file_list = os.listdir(annotations3d_base_path)
annotations3d_fixed_file_list = os.listdir(annotations3d_fixed_base_path)
train_data_list += [
    os.path.join(annotations3d_base_path, annotations3d_file_sin)
    for annotations3d_file_sin in annotations3d_file_list
]
train_data_list += [
    os.path.join(annotations3d_fixed_base_path, annotations3d_fixed_file_sin)
    for annotations3d_fixed_file_sin in annotations3d_fixed_file_list
]

# train_data_list = [train_data_sin for train_data_sin in train_data_list if '__20240220_' in train_data_sin and ('Flora302' in train_data_sin or 'Flora303' in train_data_sin)]

dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',  #
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301_update.json',  #
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301_update.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301_update.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301_update.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301_update.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301_update.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301_update.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix_with_nimble/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301_update.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_093420__all__dark__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_094228__pinch__dark__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_094637__pinch__bright__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_094851__pinch__bright__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_095839__all__bright__right__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_100545__pinch__normal__right__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_100822__pinch__normal__right__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_101030__pinch__bright__right__1111__0021__undistort_tar__Flora301.json'
]
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
# pipelines
train_pipeline = [
    # dict(
    #     type='RandomStereoParamAug',
    #     prob=0.5,
    #     baseline_range=[-0.005, 0.005],
    #     y_angle_range=[-3, 3]),
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
    # dict(
    #     type='RandomStereoParamAug',
    #     prob=0.5,
    #     baseline_range=[-0.005, 0.005],
    #     y_angle_range=[-3, 3]),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
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
        type=dataset_type,
        # data_ratio=1.0/30,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        flip_left_to_right=True,
        filter_kpt_exceed=True,
        standard_stereo=standard_stereo
        # point_type='leftcam',
        # indices=1000,
    ),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=8,
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
        flip_left_to_right=True,
        data_root=data_root,
        standard_stereo=standard_stereo
        # point_type='leftcam'
    ),
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
filter_exceed = True
val_evaluator = [
    dict(
        type='MPJPEV2',
        mode='mpjpe',
        # gesture_list=gesture_list,
        scale_metric=False,
        fit_metric=False,
        openhand_metric=True,
        pinch_hard_metric=True,
        category_metric=True,
        # bmk_save_root='/data/stliu/mmpose_new/mmpose/work_dirs/result_20231203/bad_case',
        # show_bmk_thr=(50, 10000000),
        filter_exceed=filter_exceed),  #bad case mpjpe thr (mm)
    # dict(type='MPJPEV2', mode='p-mpjpe', prefix='1'),
    # dict(type='EPE',filter_exceed=filter_exceed),
    # dict(type='NrealKeypointAP',filter_exceed=filter_exceed)
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
