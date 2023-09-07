# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=100, val_interval=5)

data_root = '/data/AI_DATA'
# data_root = '/data/AI_DATA_LOCAL'

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

pinch_thre = [20, 40]  # pinch双阈值，单位：mm
# model settings
backbone_out_channels = [64, 96, 128, 160]
model = dict(
    type='TopdownPoseLiftEstimator',
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
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(
                    type='RLELoss',
                    use_target_weight=False,
                    flow_model_pretrain_path=
                    f'{data_root}/data_hand/model/mmpose/td-hand_rsn50_pre_ipr_rle_lscale_wholedata_4xb64-100e-128x128/epoch_100.pth'
                ),
                dict(type='KeypointMSELoss', use_target_weight=True)
            ]),
        decoder=codec,
        deploy=False,
        output_sigma=True),
    kpt3d_lift=dict(
        type='LiftHead',
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
        channel_num=55,
        output_num=42,
        use_kp2d_gt=True,
        undistort=True),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        # '/home/zx_li/workspace/mmpose/work_dirs/td-hand_res26_fpn_sk_weightdata_4xb64-50e_0919data-128x128/epoch_50.pth'
        # f'{data_root}/data_hand/model/mmpose/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128/epoch_50.pth'  # ella kp2d
        # '/home/jrchen/git-project/mmpose/work_dirs/pair_hand3d/003_td-stage_two_train_55dim_l1/epoch_60_new.pth'
        # '/home/jrchen/git-project/mmpose/work_dirs/hand_2d_keypoint/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128_pretrainmodel/epoch_30_FT.pth'
        # f'{data_root}/jrchen/git-project/mmpose/work_dirs/pair_hand3d/006_td-stage_two_train_55dim_RLE_head_train_flora_finetune/FT_kp2d_add_ella_pretrain_lift.pth'  # flora kp2d + ella kp3d
        'work_dirs/jupyter/FT_kp2d_add_ella_pretrain_lift_0730.pth'),
)

# base dataset settings
dataset_type = 'PairHand3DDataset'
data_mode = 'topdown'

import os

# lmdb root dir, maybe different between beijing and wuxi

# test only
#data_root = '/data/hand_group/data/data_hand/lmdb_data/'
train_data_list = [

    #0817
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_055201__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_060208__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_060545__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_061011__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_061409__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_061729__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_062007__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_062844__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_063433__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_063709__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_064712__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_070806__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_071036__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_071343__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_071825__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_072101__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_072508__pinch__normal__left__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_073423__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_075808__pinch__normal__right__1111__0009__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_080625__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_081217__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_081932__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_082713__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_083423__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_084126__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_084826__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_085426__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_090004__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_090959__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_091632__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_092215__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_093630__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_094135__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_095004__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_095817__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_100411__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_100931__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_101522__pinch__normal__left__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230817_101936__pinch__normal__right__1111__0013__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_055201__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_060208__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_060545__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_061011__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_061409__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_061729__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_062007__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_062844__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_063433__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_063709__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_064712__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_070806__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_071036__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_071343__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_071825__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_072101__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_072508__pinch__normal__left__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_073423__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_075808__pinch__normal__right__1111__0009__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_080625__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_081217__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_081932__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_082713__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_083423__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_084126__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_084826__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_085426__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_090004__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_090959__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_091632__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_092215__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_093630__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_094135__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_095004__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_095817__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_100411__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_100931__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_101522__pinch__normal__left__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230817_101936__pinch__normal__right__1111__0013__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_055201__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_060208__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_060545__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_061011__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_061409__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_061729__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_062007__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_062844__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_063433__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_063709__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_064712__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_070806__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_071036__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_071343__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_071825__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_072101__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_072508__pinch__normal__left__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_073423__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_075808__pinch__normal__right__1111__0009__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_080625__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_081217__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_081932__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_082713__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_083423__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_084126__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_084826__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_085426__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_090004__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_090959__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_091632__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_092215__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_093630__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_094135__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_095004__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_095817__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_100411__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_100931__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_101522__pinch__normal__left__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230817_101936__pinch__normal__right__1111__0013__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_055201__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_060208__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_060545__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_061011__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_061409__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_061729__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_062007__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_062844__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_063433__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_063709__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_064712__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_070806__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_071036__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_071343__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_071825__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_072101__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_072508__pinch__normal__left__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_073423__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_075808__pinch__normal__right__1111__0009__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_080625__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_081217__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_081932__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_082713__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_083423__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_084126__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_084826__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_085426__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_090004__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_090959__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_091632__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_092215__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_093630__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_094135__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_095004__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_095817__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_100411__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_100931__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_101522__pinch__normal__left__1111__0013__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230817_101936__pinch__normal__right__1111__0013__undistort_tar__Flora304.json',

    #0822
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_060828__all__normal__left__1111__0014__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_061518__all__normal__right__1111__0014__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_064239__all__normal__right__1111__0014__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_064727__all__normal__left__1111__0014__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_065520__all__normal__left__1111__0015__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_070140__all__normal__right__1111__0015__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_070647__all__normal__left__1111__0015__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_071139__all__normal__right__1111__0015__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_071725__all__normal__left__1111__0011__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230822_072236__all__normal__right__1111__0011__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_060828__all__normal__left__1111__0014__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_061518__all__normal__right__1111__0014__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_064239__all__normal__right__1111__0014__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_064727__all__normal__left__1111__0014__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_065520__all__normal__left__1111__0015__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_070140__all__normal__right__1111__0015__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_070647__all__normal__left__1111__0015__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_071139__all__normal__right__1111__0015__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_071725__all__normal__left__1111__0011__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230822_072236__all__normal__right__1111__0011__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_060828__all__normal__left__1111__0014__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_061518__all__normal__right__1111__0014__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_064239__all__normal__right__1111__0014__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_064727__all__normal__left__1111__0014__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_065520__all__normal__left__1111__0015__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_070140__all__normal__right__1111__0015__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_070647__all__normal__left__1111__0015__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_071139__all__normal__right__1111__0015__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_071725__all__normal__left__1111__0011__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230822_072236__all__normal__right__1111__0011__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_060828__all__normal__left__1111__0014__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_061518__all__normal__right__1111__0014__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_064239__all__normal__right__1111__0014__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_064727__all__normal__left__1111__0014__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_065520__all__normal__left__1111__0015__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_070140__all__normal__right__1111__0015__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_070647__all__normal__left__1111__0015__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_071139__all__normal__right__1111__0015__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_071725__all__normal__left__1111__0011__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230822_072236__all__normal__right__1111__0011__undistort_tar__Flora304.json',

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
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora303.json',
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
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora304.json',
]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0007__20230809_100619__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora8/XS__all__normal__right__1000__0011__20230810_081042__undistort_tar__Flora8.json'

    #0809  眼镜离手比较近
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0007__20230809_090503__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0006__20230809_055248__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0006__20230809_055716__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0007__20230809_101132__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0006__20230809_054849__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0006__20230809_060122__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0007__20230809_060812__undistort_tar__Flora301.json',

    #0815
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_080834__pinch__normal__left__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_081340__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_081732__pinch__normal__left__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_083351__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_083727__pinch__normal__left__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_084129__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_085606__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_091548__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_094552__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_095319__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_095734__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_080834__pinch__normal__left__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_081340__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_081732__pinch__normal__left__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_083351__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_083727__pinch__normal__left__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_084129__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_085606__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_091548__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_094552__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_095319__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230815_095734__pinch__normal__right__1111__0008__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_080834__pinch__normal__left__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_081340__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_081732__pinch__normal__left__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_083351__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_083727__pinch__normal__left__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_084129__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_085606__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_091548__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_094552__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_095319__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora303/XS__20230815_095734__pinch__normal__right__1111__0008__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_080834__pinch__normal__left__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_081340__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_081732__pinch__normal__left__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_083351__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_083727__pinch__normal__left__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_084129__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_085606__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_091548__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_094552__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_095319__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230815_095734__pinch__normal__right__1111__0008__undistort_tar__Flora304.json',
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
    batch_size=128,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        # point_type='leftcam',
        # indices=200,
        # data_ratio=0.3,
        # pinch_random=True,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
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
val_evaluator = [
    dict(type='MPJPEV2', gesture_list=gesture_list, result_dir='.'),
    # dict(type='MPJPEV2', mode='p-mpjpe', prefix='1'),
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
