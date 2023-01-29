# flake8: noqa
_base_ = ['../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=50, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
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
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='IntegralRegressionLabel',
    input_size=(128, 128),
    heatmap_size=(32, 32),
    sigma=1,
    normalize=False,
    blur_kernel_size=1,
)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='RSNTiny',
        stage_num=1,
        upsample_chl_num=192,
        output_last_only=True),
    head=dict(
        type='DSNTHead',
        in_channels=192,
        deconv_out_channels=(),
        has_final_layer=True,
        in_featuremap_size=(32, 32),
        num_joints=21,
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='RLELoss', use_target_weight=False),
                dict(type='KeypointMSELoss', use_target_weight=True)
            ]),
        decoder=codec,
        output_sigma=True),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/home/zx_li/workspace/mmpose/work_dirs/td-hand_rsntiny_over_8xb64-50e_nreal-128x128/epoch_50.pth'
    ),
)

# base dataset settings
dataset_type = 'HANDDataset'
data_mode = 'topdown'

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
    '/data/data_hand/hand_keypoint/annotations/test_nreal_gesture_1111_1_1_twohand_lmdb.json'
]

# pipelines
train_pipeline = [
    dict(
        type='Albumentation',
        transforms=[
            dict(type='RandomBrightnessContrast', p=0.2),
        ]),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=15,
        rotate_prob=0.3,
        shift_prob=0.5,
        shift_factor=0.2),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='GenerateTarget',
        target_type='heatmap+keypoint_label',
        encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
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
    ))
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
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='mAP', rule='greater'))

# evaluators
val_evaluator = dict(type='NrealKeypointAP', )
test_evaluator = val_evaluator

# fp16 settings
fp16 = dict(loss_scale='dynamic')
# model wrapper
find_unused_parameters = False
