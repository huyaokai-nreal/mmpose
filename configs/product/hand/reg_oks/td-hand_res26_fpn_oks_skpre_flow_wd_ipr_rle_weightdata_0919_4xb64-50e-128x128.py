# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=50, val_interval=5)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4, weight_decay=1e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        custom_keys={'head.loss_module': dict(lr_mult=0.0, decay_mult=0.0)}))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=2000, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=train_cfg['max_epochs'],
        convert_to_iter_based=True,
        eta_min=1e-7)
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

# model settings
backbone_out_channels = [64, 96, 128, 160]
model = dict(
    type='TopdownPoseEstimator',
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
        type='DSNTAttrHead',
        in_channels=192,
        deconv_out_channels=(),
        has_final_layer=True,
        in_featuremap_size=(32, 32),
        num_joints=21,
        output_oks=True,
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(
                    type='RLELoss',
                    use_target_weight=False,
                    flow_model_pretrain_path=
                    '/data/hand_group/model_zoo/mmpose/td-hand_rsn50_pre_ipr_rle_lscale_wholedata_4xb64-100e-128x128/epoch_100.pth'
                ),
                dict(type='KeypointMSELoss', use_target_weight=True),
                dict(
                    type='SmoothL1Loss',
                    use_target_weight=False,
                    loss_weight=10)
            ]),
        decoder=codec,
        deploy=False,
        output_sigma=True),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/home/zx_li/workspace/mmpose/work_dirs/td-hand_res26_fpn_sk_weightdata_4xb64-50e_0919data-128x128/epoch_50.pth'
    ),
)

# base dataset settings
dataset_type = 'HANDDataset'
data_mode = 'topdown'

import os

# lmdb root dir, maybe different between beijing and wuxi
data_root = '/data/hand_group/data/'
train_data_list = [
    'data_hand/hand_keypoint/annotations/train_hanco_rgb_gesture_lmdb_refresh.json',  #84k
    'data_hand/hand_keypoint/annotations/train_nreal_baidu1_gesture_right_0930_lmdb.json',  #13.4k
    'data_hand/hand_keypoint/annotations/train_nreal_baidu2_gesture_right_1014_lmdb.json',  #12k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_0107_2_1_lmdb.json',  #16.8k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_220216_2_2_lmdb.json',  #32k
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_220216_2_3_lmdb.json',  #12k
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
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0916_1_30_bad_case_twohand_lmdb.json',
    'data_hand/hand_keypoint/annotations/train_nreal_synth_gesture_2cam_1~30animation_20220803_lmdb.json'  #62k
]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    '/data/hand_group/data/data_hand/hand_keypoint/annotations/test_nreal_gesture_1111_1_1_twohand_gesture_lmdb.json'
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
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root))
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
        data_root=data_root))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='mAP', rule='greater'))

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]
val_evaluator = dict(
    type='NrealKeypointAP', gesture_list=gesture_list, result_dir='./')
test_evaluator = val_evaluator

# fp16 settings
fp16 = dict(loss_scale='dynamic')
# model wrapper
find_unused_parameters = True
