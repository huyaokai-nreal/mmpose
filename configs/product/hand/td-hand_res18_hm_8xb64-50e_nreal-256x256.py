# flake8: noqa
_base_ = ['../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=50, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=2400, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        milestones=[30, 40],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='mAP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(type='ResNet', depth=18, in_channels=1),
    head=dict(
        type='HeatmapHead',
        in_channels=512,
        out_channels=21,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'HANDDataset'
data_mode = 'topdown'

# pipelines
train_pipeline = [
    dict(type='Albumentation'),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=30),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', target_type='heatmap', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
train_data_list = [
    '/data/data_hand/hand_keypoint/annotations/train_nreal_baidu2_gesture_right_1014_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0318_1_14_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0616_1_21_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0127_1_10_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_hanco_rgb_gesture_lmdb_refresh.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0517_1_19_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1111_1_1_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_0107_2_1_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_synth_gesture_1008_lmdb_3_arbitrary.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0401_1_15_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1118_1_2_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0624_1_22_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1202_1_4_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0415_1_16_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_220216_2_3_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0510_1_18_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0523_1_20_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0318_1_13_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0304_1_12_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_220216_2_2_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1223_1_7_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1230_1_8_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1125_1_3_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0424_1_17_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1216_1_6_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_baidu1_gesture_right_0930_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_1209_1_5_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0113_1_9_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0218_1_11_twohand_lmdb.json'
]

val_data_list = [
    '/data/data_hand/hand_keypoint/annotations/test_nreal_gesture_1111_1_1_twohand_lmdb.json'
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
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_file_list=val_data_list,
        data_mode=data_mode,
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='NrealKeypointAP',
    ann_file=val_data_list[0],
)
test_evaluator = val_evaluator
