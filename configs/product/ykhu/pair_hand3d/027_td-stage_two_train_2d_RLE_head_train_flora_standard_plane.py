# flake8: noqa
import os

_base_ = ['../../../_base_/default_runtime.py']

from mmpose.configs._base_.datasets.xs3d_nimble import \
    datasets_info as kpt3d_datasets_info

train_cfg = dict(max_epochs=100, val_interval=5)

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
standard_stereo = True  # 是否转换标准双目
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
        output_depth=False,
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(
                    type='RLELoss',
                    use_target_weight=False,
                    flow_model_pretrain_path=
                    '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn50_pre_ipr_rle_lscale_wholedata_4xb64-100e-128x128/epoch_100.pth'
                ),
                dict(type='KeypointMSELoss', use_target_weight=True)
            ]),
        decoder=codec,
        deploy=False,
        output_sigma=False),
    kpt3d_lift=dict(
        type='LiftHeadStandard',
        lift_loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss'),  # 3d kpts
                dict(type='L1Loss'),  # 3d kpts leftcam
                dict(type='L1Loss'),  # 3d kpts rightcam
                dict(type='MSELoss', loss_weight=0),  # 2d reprojection left
                dict(type='MSELoss', loss_weight=0),  # 2d reprojection right
                dict(
                    type='PinchLoss',
                    enter_thre=pinch_thre[0] / 1000,
                    exit_thre=pinch_thre[1] / 1000,
                    loss_weight=1,
                    enable_start_epoch=train_cfg['max_epochs'] -
                    20),  # 后20 epoch打开pinch loss
            ]),
        all_use_kp2d_gt=False,
        baseline=0.135,
        reproj_thre=440,
        iou_thre=0.5),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn26_fpn_25d_scale_ipr_right_2d3d_handmix_aug_240229data_4x128-100e-128x128/epoch_100.pth'  # 20240315 T 2D model
    ))

# base dataset settings
dataset_type = 'PairHand3DDataset'
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

# train_data_list = [
#     'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora302.json',
#     'data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20240517_030629__all__normal__right__1101__0006__undistort_tar__Flora301.json'
# ]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora304.json',  # xujian 33684 images, 16842 pair instances
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora304.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora304.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora303.json',  # xujian 33684 images, 16842 pair instances
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora303.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora302.json',  # xujian 33684 images, 16842 pair instances
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora302.json',
    # # flora301
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',  # xujian 33684 images, 16842 pair instances
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora301.json',   # lizuoxin
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_075728__all__dark__left__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_080158__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_080536__pinch__bright__left__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_081151__pinch__dark__right__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_055835__all__dark__left__1111__0002__undistort_tar__Flora301.json',    # haoruitao
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_060445__all__dark__left__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_060954__all__normal__right__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_061650__pinch__normal__left__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_061830__pinch__bright__right__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_062256__pinch__normal__left__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_062538__pinch__dark__right__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_062851__pinch__normal__left__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_063209__pinch__bright__right__1111__0002__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_081413__all__normal__right__1111__0020__undistort_tar__Flora301.json',   # maleiyuan
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_082101__all__bright__left__1111__0020__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_082659__pinch__bright__right__1111__0020__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_083215__pinch__dark__left__1111__0020__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_083550__pinch__normal__right__1111__0020__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_083958__pinch__normal__right__1111__0020__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_084301__pinch__bright__left__1111__0020__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_093420__all__dark__left__1111__0021__undistort_tar__Flora301.json',     # panyuehui       #  small_mpjpe: 12.7574  middle_mpjpe: 9.1662  large_mpjpe: 8.0888
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_094228__pinch__dark__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_094637__pinch__bright__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_094851__pinch__bright__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_095839__all__bright__right__1111__0021__undistort_tar__Flora301.json',  # small_mpjpe: 11.0016  middle_mpjpe: 9.8589  large_mpjpe: 13.5482
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_100545__pinch__normal__right__1111__0021__undistort_tar__Flora301.json',  # small_mpjpe: 11.5316  middle_mpjpe: 12.8182  large_mpjpe: 18.1924
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_100822__pinch__normal__right__1111__0021__undistort_tar__Flora301.json',  #  small_mpjpe: 23.1960  middle_mpjpe: 25.5704  large_mpjpe: 15.8753
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_101030__pinch__bright__right__1111__0021__undistort_tar__Flora301.json'  # small_mpjpe: 8.0687  middle_mpjpe: 8.1140  large_mpjpe: 8.5837

    # flora303
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora303.json',    # xujian 33684 images, 16842 pair instances
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora303.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora303.json',   # lizuoxin
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_075728__all__dark__left__1111__0019__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_080158__pinch__normal__right__1111__0019__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_080536__pinch__bright__left__1111__0019__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_081151__pinch__dark__right__1111__0019__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora303.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_055835__all__dark__left__1111__0002__undistort_tar__Flora303.json',    # haoruitao
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_060445__all__dark__left__1111__0002__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_060954__all__normal__right__1111__0002__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_061650__pinch__normal__left__1111__0002__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_061830__pinch__bright__right__1111__0002__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_062256__pinch__normal__left__1111__0002__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_062538__pinch__dark__right__1111__0002__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_062851__pinch__normal__left__1111__0002__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_063209__pinch__bright__right__1111__0002__undistort_tar__Flora303.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_081413__all__normal__right__1111__0020__undistort_tar__Flora303.json',   # maleiyuan
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_082101__all__bright__left__1111__0020__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_082659__pinch__bright__right__1111__0020__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_083215__pinch__dark__left__1111__0020__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_083550__pinch__normal__right__1111__0020__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_083958__pinch__normal__right__1111__0020__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230831_084301__pinch__bright__left__1111__0020__undistort_tar__Flora303.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_093420__all__dark__left__1111__0021__undistort_tar__Flora303.json',     # panyuehui
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_094228__pinch__dark__left__1111__0021__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_094637__pinch__bright__left__1111__0021__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_094851__pinch__bright__left__1111__0021__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_095839__all__bright__right__1111__0021__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_100545__pinch__normal__right__1111__0021__undistort_tar__Flora303.json',  # small_mpjpe: 11.5316  middle_mpjpe: 12.8182  large_mpjpe: 18.1924
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_100822__pinch__normal__right__1111__0021__undistort_tar__Flora303.json',  #  small_mpjpe: 23.1960  middle_mpjpe: 25.5704  large_mpjpe: 15.8753
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_gesture/XS__20230904_101030__pinch__bright__right__1111__0021__undistort_tar__Flora303.json'  # small_mpjpe: 8.0687  middle_mpjpe: 8.1140  large_mpjpe: 8.5837

    #黑人3d
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_054705__all__normal__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_055912__all__bright__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_060453__pinch__bright__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_061312__pinch__bright__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_061915__pinch__normal__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_062634__pinch__dark__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_062946__pinch__dark__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_063405__pinch__normal__right__1111__0023__undistort_tar__Flora301.json'

    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_054705__all__normal__left__1111__0023__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_055912__all__bright__right__1111__0023__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_060453__pinch__bright__left__1111__0023__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_061312__pinch__bright__right__1111__0023__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_061915__pinch__normal__left__1111__0023__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_062634__pinch__dark__right__1111__0023__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_062946__pinch__dark__left__1111__0023__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230911_063405__pinch__normal__right__1111__0023__undistort_tar__Flora302.json'
    # quest系统手势
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083647__pinch__normal__right__1110__0025__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083322__pinch__normal__left__1110__0025__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083322__pinch__normal__left__1110__0025__undistort_tar__Flora302.json',
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083647__pinch__normal__right__1110__0025__undistort_tar__Flora302.json',

    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083322__pinch__normal__left__1110__0025__undistort_tar__Flora303.json',
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083647__pinch__normal__right__1110__0025__undistort_tar__Flora303.json',
]
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
# pipelines
train_pipeline = [
    # dict(
    #     type='RandomStereoParamAugV2',
    #     prob=0.1,
    #     baseline_range=[-0.033, -0.032],
    #     right_shift_range=[-0.1, 0.1],
    #     ella_right_aug=True # if False, normal ella aug
    # ),
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
    #     type='RandomStereoParamAugV2',
    #     prob=1,
    #     baseline_range=[-0.033, -0.032],
    #     right_shift_range=[-0.1, 0.1],
    #     ella_right_aug=True
    # ),
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
        # data_ratio=1.0/5,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        flip_left_to_right=True,
        filter_kpt_exceed=False,
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
        pinch_thre=pinch_thre,
        # bmk_save_root='/home/ykhu/workspace/mmpose/work_dirs/bad_case_liftnet/pinch0315',
        # show_bmk_thr=(20, 10000000),
        filter_exceed=filter_exceed),  #bad case mpjpe thr (mm)
    # dict(type='MPJPEV2', mode='p-mpjpe', prefix='1',filter_exceed=filter_exceed),
    # dict(type='EPE',filter_exceed=filter_exceed),
    # dict(type='NrealKeypointAP',filter_exceed=filter_exceed, category_metric=True)
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
