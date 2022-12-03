# flake8: noqa
_base_ = ['../../_base_/default_runtime.py']
# runtime
train_cfg = dict(max_epochs=60, val_interval=5)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    ))
# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=2000, start_factor=0.1,
         by_epoch=False),  # warm-up
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=train_cfg['max_epochs'],
        convert_to_iter_based=True,
        eta_min=1e-7)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='mAP', rule='greater'))
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook'),
    # dict(type='EMAHook'),
]
# codec settings
# multiple kernel_sizes of heatmap gaussian for 'Megvii' approach.
kernel_sizes = [11, 9, 7, 5]
codec = [
    dict(
        type='NrealHeatmap',
        input_size=(128, 128),
        heatmap_size=(32, 32),
        kernel_size=kernel_size) for kernel_size in kernel_sizes
]

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='RSNTiny',
        stage_num=1,
        upsample_chl_num=192,
    ),
    head=dict(
        type='MSPNHead',
        out_shape=(32, 32),
        unit_channels=192,
        out_channels=21,
        num_stages=1,
        num_units=4,
        norm_cfg=dict(type='BN'),
        # each sub list is for a stage
        # and each element in each list is for a unit
        level_indices=[0, 1, 2, 3],
        loss=[dict(type='JointsL2Loss', loss_weight=0.25)] * 3 +
        [dict(type='JointsL2Loss', has_ohkm=True, topk=17, loss_weight=1.)],
        decoder=codec[-1]),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
dataset_type = 'HANDDataset'
data_mode = 'topdown'

# pipelines
train_pipeline = [
    #dict(type='Albumentation'),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=15,
        rotate_prob=0.3,
        shift_prob=0.5,
        shift_factor=0.2),
    dict(type='TopdownAffine', input_size=codec[0]['input_size']),
    dict(
        type='GenerateTarget', target_type='multilevel_heatmap',
        encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='TopdownAffine', input_size=codec[0]['input_size']),
    dict(type='PackPoseInputs')
]
train_data_list = [
    '/data/data_hand/hand_keypoint/annotations/train_nreal_baidu2_gesture_right_1014_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0318_1_14_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0616_1_21_bad_data_twohand_lmdb.json',
    '/data/data_hand/hand_keypoint/annotations/train_nreal_gesture_0127_1_10_twohand_lmdb.json',
    #'/data/data_hand/hand_keypoint/annotations/train_hanco_rgb_gesture_lmdb_refresh.json',
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
    #'/data/data_hand/hand_keypoint/annotations/test_nreal_gesture_1111_1_1_twohand_lmdb.json'
    '/data/data_hand/hand_keypoint/annotations/test_nreal_gesture_3_1_221201_fisheye_vertical_binocular_lmdb.json'
    #'/data/data_hand/hand_keypoint/annotations/test_nreal_gesture_3_2_221201_fisheye_horizontal_binocular_lmdb.json'
]
# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=False,
    prefetch_factor=2,
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
        flip_left_to_right=True,
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

# fp16 settings
fp16 = dict(loss_scale='dynamic')
# model wrapper
find_unused_parameters = False
