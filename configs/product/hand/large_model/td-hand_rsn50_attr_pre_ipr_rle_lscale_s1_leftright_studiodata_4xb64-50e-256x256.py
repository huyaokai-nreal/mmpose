# flake8: noqa
_base_ = ['../../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=30, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4, weight_decay=1e-4))
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
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=1,
    normalize=False,
    blur_kernel_size=5,
)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='RSN',
        unit_channels=256,
        num_stages=1,
        num_units=4,
        num_blocks=[3, 4, 6, 3],
        num_steps=4,
        image_channels=1,
        norm_cfg=dict(type='BN'),
        output_last_only=True),
    head=dict(
        type='DSNTAttrHead',
        in_channels=256,
        deconv_out_channels=(),
        has_final_layer=True,
        in_featuremap_size=(64, 64),
        deploy_output=['fused_kpt'],
        num_joints=21,
        attr_dim=24,
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='RLELoss', use_target_weight=False),
                dict(type='KeypointMSELoss', use_target_weight=True),
                dict(
                    type='BCELoss',
                    use_target_weight=False,
                    loss_weight=10,
                    with_logits=True)
            ]),
        decoder=codec,
        deploy=False,
        output_sigma=True,
        output_fuse_coord=True),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn50_pre_ipr_rle_lscale_s1_0919data_4xb64-50e-128x128/epoch_45.pth'
    ),
)

# base dataset settings
dataset_type = 'HANDDataset'
data_mode = 'topdown'

import os

# lmdb root dir, maybe different between beijing and wuxi
#data_root = '/data/hand_group/data'
data_root = '/data/AI_DATA_WX'
train_data_list = [
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
    'data_hand/hand_keypoint/annotations/train_nreal_gesture_0916_1_30_bad_case_twohand_gesture_lmdb.json',
    'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.0_flir_230519.json',  #2k
    'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.1_flir_230519.json',  #2k
    'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.2_flir_230526.json',  # 2.5k
    'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.3_flir_230601.json',  # 2.5k
    'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.4_flir_230601.json',  # 2.5k
    'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.8_flir_230616_3k.json',  # 2.5k
    'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_2.2_flir_230816_50k.json',  # 50k
    'data_hand/hand_keypoint/annotations/nreal_studio_all__normal__left__1000__0006__20230809_055248_lmdb_4k.json',
    'data_hand/hand_keypoint/annotations/nreal_studio__all__normal__left__1000__0007__20230809_090503__undistort_tar_4k_lmdb.json',
    'data_hand/hand_keypoint/annotations/nreal_studio__all__normal__right__1000__0006__20230809_054849__undistort_tar_4k_lmdb.json',
    'data_hand/hand_keypoint/annotations/XS__20230830__1111__0005__undistort_tar_flir__30__binocular__lmdb.json',
    'data_hand/hand_keypoint/annotations/XS__20230904__1111__0021__undistort_tar_flir__30__binocular__lmdb.json',
    'data_hand/hand_keypoint/annotations/XS__20230831__1111__0002__undistort_tar_flir__30__binocular__lmdb.json',
    'data_hand/hand_keypoint/annotations/XS__20230830__1111__0019__undistort_tar_flir__30__binocular__lmdb.json',
    'data_hand/hand_keypoint/annotations/XS__20230831__1111__0020__undistort_tar_flir__30__binocular__lmdb.json'
]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
studio_data_num = 15
studio_data_weight = 0.9
dataset_weight_list = [
    (1 - studio_data_weight) / (len(train_data_list) - studio_data_num)
] * (len(train_data_list) - studio_data_num
     ) + [studio_data_weight / studio_data_num] * studio_data_num
val_data_list = [
    #'data_hand/hand_keypoint/annotations/test_nreal_gesture_1111_1_1_twohand_gesture_lmdb.json'
    #'data_hand/hand_keypoint/annotations/nreal_studio/test_nreal_studio_1.0_flir_230519.json',
    #'data_hand/hand_keypoint/annotations/nreal_studio/test_nreal_studio_1.1_flir_230519.json'
    'data_hand/hand_keypoint/annotations/nreal_studio/test_nreal_studio_1.5_flir_230613_3k.json'
    #'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.0_flir_230519.json', #2k
    #'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.1_flir_230519.json', #2k
    #'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.2_flir_230526.json', # 2.5k
    #'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.3_flir_230601.json', # 2.5k
    #'data_hand/hand_keypoint/annotations/nreal_studio/train_nreal_studio_1.4_flir_230601.json', # 2.5k
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
    dict(type='RandomFlip', prob=0.5, cat_id_map={
        1: 2,
        2: 1
    }),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        scale_norm_low=-2.0,
        rotate_factor=90,
        rotate_prob=0.3,
        shift_prob=0.5,
        shift_factor=0.2),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='GenerateAttrLabel',
        num_class=3,
        attr_list=['keypoints_visible', 'cat_id']),
    dict(
        type='GenerateTarget',
        target_type='heatmap+keypoint_label',
        encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='GenerateAttrLabel',
        num_class=3,
        attr_list=['keypoints_visible', 'cat_id']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        ignore_visible=False,
        data_root=data_root))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_file_list=val_data_list,
        data_mode=data_mode,
        test_mode=True,
        flip_left_to_right=False,
        pipeline=val_pipeline,
        ignore_visible=False,
        data_root=data_root))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='mAP', rule='greater'))

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]
val_evaluator = [
    dict(type='NrealKeypointAP', gesture_list=gesture_list),
    dict(type='AttrClsAccuracy'),
]
test_evaluator = val_evaluator

# fp16 settings
fp16 = dict(loss_scale='dynamic')
# model wrapper
find_unused_parameters = False
