# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']
# runtime
train_cfg = dict(max_epochs=100, val_interval=10)

data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'
test_type = '3d'
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

codec = dict(
    type='RegressionLabel',
    input_size=(128, 128, 128),
    with_depth=True,
    depth_bound=0.4)
codec2d = dict(
    type='RegressionLabel',
    input_size=(128, 128),
    with_depth=False,
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
        consistency_loss=False,
        heatmap_loss=False,
        output_depth=True,
        deploy_output=['feat', 'depth'],
        input_size=128,
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                #dict(
                #    type='RLELoss',
                #    use_target_weight=False,
                #    flow_model_pretrain_path=
                #    '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn50_pre_ipr_rle_lscale_wholedata_4xb64-100e-128x128/epoch_100.pth'
                #),
                dict(type='L1Loss', use_target_weight=False, loss_weight=1),
                dict(type='L1Loss', use_target_weight=False, loss_weight=1)
            ]),
        decoder=codec,
        deploy=False,
        output_sigma=False),
    test_cfg=dict(flip_test=False, ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/data/AI_DATA/data_hand/model/mmpose/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128/epoch_50.pth'
    ),
    root_mode='optimize' if test_type == '3d' else 'gt',
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
    checkpoint=dict(save_best='all_p-mpjpe', rule='less'))
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
train_2d_pipeline = [
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
    dict(type='TopdownAffine', input_size=codec2d['input_size']),
    dict(type='GenerateTarget', encoder=codec2d),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='KeypointTo25DLabel'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_2d_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec2d['input_size']),
    dict(type='GenerateTarget', encoder=codec2d),
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
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora302.json',
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
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora304.json',
    #'data_hand/hand_keypoint/annotations3d/Flora304/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora304.json',
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
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora302.json',
    #'data_hand/hand_keypoint/annotations3d/Flora302/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora302.json',
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
train_2d_data_list = [
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1111_1_1_twohand_lmdb.json',  #13.2k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1118_1_2_twohand_lmdb.json',  #24k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1125_1_3_twohand_lmdb.json',  #29.7k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1202_1_4_twohand_lmdb.json',  #31.8k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1209_1_5_twohand_lmdb.json',  #24.3k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1216_1_6_twohand_lmdb.json',  #25.1k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1223_1_7_twohand_lmdb.json',  #27.8k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_1230_1_8_twohand_lmdb.json',  #17.2k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0113_1_9_twohand_lmdb.json',  #24k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0127_1_10_twohand_lmdb.json',  #25k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0218_1_11_twohand_lmdb.json',  #16k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0304_1_12_twohand_lmdb.json',  #22k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0905_1_13~15_bad_data_twohand_lmdb.json',  #26.5k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0906_1_16~20_bad_data_twohand_lmdb.json',  #88k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0906_1_21~22_bad_data_twohand_lmdb.json',  #70k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0905_1_23~29_bad_data_twohand_lmdb.json',  #116k
    'data_hand/hand_keypoint/annotations/hand_train_flora_10k_230327_1_cam0_lmdb__point_flora.json',  #10k
    'data_hand/hand_keypoint/annotations/hand_train_flora_20k_230822_1_cam0_lmdb.json',
    'data_hand/hand_keypoint/annotations/hand_train_flora_20k_230829_1_cam0_lmdb.json'
]
train_2d_data_list = [
    os.path.join(data_root, item) for item in train_2d_data_list
]

val_data_list = [
    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0007__20230809_100619__undistort_tar__Flora301.json',
    #'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_080834__pinch__normal__left__1111__0008__undistort_tar__Flora301.json'
]
#val_data_list = [
#    #0809  眼镜离手比较近
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0007__20230809_090503__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0006__20230809_055248__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0006__20230809_055716__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__left__1000__0007__20230809_101132__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0006__20230809_054849__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0006__20230809_060122__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__all__normal__right__1000__0007__20230809_060812__undistort_tar__Flora301.json',
#
#    #0815
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_080834__pinch__normal__left__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_081340__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_081732__pinch__normal__left__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_083351__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_083727__pinch__normal__left__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_084129__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_085606__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_091548__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_094552__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_095319__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#    'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230815_095734__pinch__normal__right__1111__0008__undistort_tar__Flora301.json',
#]
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='MultiSourceSampler', source_ratio=[0.5, 0.5], batch_size=128),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/nreal_hand.py'),
        datasets=[
            dict(
                type=dataset_type,
                data_ratio=1,
                data_file_list=train_data_list,
                data_mode=data_mode,
                pipeline=train_pipeline,
                dataset_weight_list=dataset_weight_list,
                data_root=data_root,
                flip_left_to_right=True,
                point_type='2.5D'
                # indices=1000,
            ),
            dict(
                type='HANDDataset',
                data_file_list=train_2d_data_list,
                data_mode=data_mode,
                pipeline=train_2d_pipeline,
                flip_left_to_right=True,
                data_root=data_root)
        ]),
)
val_2d_list = [
    'data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230627_10k_lmdb.json',  # flora test
    'data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230703_10k_lmdb.json',  # flora test
    'data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230712_8k_lmdb.json'  # flora test
]
val_2d_list = [os.path.join(data_root, item) for item in val_2d_list]
val_3d_dataset = dict(
    type=dataset_type,
    data_file_list=val_data_list,
    data_mode=data_mode,
    # hand template from outside algorithm, such binocular pipeline
    #extern_hand_template_path = '/home/zx_li/workspace/mmpose/work_dirs/binocular_hand_template.npy',
    test_mode=True,
    pipeline=val_pipeline,
    flip_left_to_right=True,
    point_type='2.5D',
    data_root=data_root)
val_2d_dataset = dict(
    type='HANDDataset',
    data_file_list=val_2d_list,
    data_mode=data_mode,
    test_mode=True,
    pipeline=val_2d_pipeline,
    flip_left_to_right=True,
    data_root=data_root)
val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=val_2d_dataset if test_type == '2d' else val_3d_dataset)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [dict(type='EPE'), dict(type='NrealKeypointAP')]
if test_type == '3d':
    val_evaluator += [
        dict(type='MPJPEV2', mode=['mpjpe', 'p-mpjpe']),
    ]

test_evaluator = val_evaluator
find_unused_parameters = True
