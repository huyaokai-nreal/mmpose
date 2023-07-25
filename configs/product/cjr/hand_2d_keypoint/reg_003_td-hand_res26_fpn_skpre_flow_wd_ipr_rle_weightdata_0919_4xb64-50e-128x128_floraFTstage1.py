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
# param_scheduler = [
#     dict(
#         type='LinearLR', begin=0, end=2000, start_factor=0.001,
#         by_epoch=False),  # warm-up
#     dict(
#         type='CosineAnnealingLR',
#         by_epoch=True,
#         T_max=train_cfg['max_epochs'],
#         convert_to_iter_based=True,
#         eta_min=1e-7)
# ]

param_scheduler = [
    dict(  # scheduler
        type='MultiStepLR',
        begin=0,
        end=50,
        milestones=[20, 40],
        gamma=0.1,
        by_epoch=True)
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
                    '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn50_pre_ipr_rle_lscale_wholedata_4xb64-100e-128x128/epoch_100.pth'
                ),
                dict(type='KeypointMSELoss', use_target_weight=True)
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
        '/data/AI_DATA/data_hand/model/mmpose/td-hand_res26_fpn_sk_weightdata_4xb64-50e_0919data-128x128/epoch_50.pth'
        # '/data/AI_DATA/data_hand/model/mmpose/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128/epoch_50.pth'
    ),
)

# base dataset settings
dataset_type = 'PairHand3DDataset'
# dataset_type = 'HANDDataset'
data_mode = 'topdown'

import os

# lmdb root dir, maybe different between beijing and wuxi
# data_root = '/data/hand_group/data'
# for beijin server
# data_root = '/data/AI_DATA_WX'
data_root = '/data/AI_DATA'
# test only
#data_root = '/data/hand_group/data/data_hand/lmdb_data/'
train_data_list = [
    # 'data_hand/hand_keypoint/annotations3d/flora/flora8_1_binocular_0629_1_0.json',  # right
    # 'data_hand/hand_keypoint/annotations3d/flora/flora8_1_binocular_0710_2_3.json',  # left
    # 'data_hand/hand_keypoint/annotations3d/flora/flora8_1_binocular_0710_2_4.json',  # left
    # 'data_hand/hand_keypoint/annotations3d/flora/flora8_1_binocular_0710_2_5.json',  # right
    'data_hand/hand_keypoint/annotations3d/flora_with_tag/flora8_1_binocular_0629_1_0_right_gesture.json',  # right
    'data_hand/hand_keypoint/annotations3d/flora_with_tag/flora8_1_binocular_0710_2_3_left_gesture.json',  # left
    'data_hand/hand_keypoint/annotations3d/flora_with_tag/flora8_1_binocular_0710_2_4_left_gesture.json',  # left
    'data_hand/hand_keypoint/annotations3d/flora_with_tag/flora8_1_binocular_0710_2_5_right_gesture.json',  # right
    'data_hand/hand_keypoint/annotations3d/flora_with_tag/flora8_1_binocular_0714_2_2_right_gesture.json'
]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    'data_hand/hand_keypoint/annotations3d/flora_with_tag/flora8_1_binocular_0629_1_4_right_gesture.json'
    # 'data_hand/hand_keypoint/annotations3d/flora/flora8_1_binocular_0629_1_4.json'  # right   flora
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
        data_root=data_root,
        # indices=1000
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='mAP', rule='greater'))

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]
val_evaluator = dict(
    type='NrealKeypointAP',
    gesture_list=gesture_list,
    result_dir='./',
    with_tag=False)
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
