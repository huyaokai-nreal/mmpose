# flake8: noqa
import os

_base_ = ['../../../_base_/default_runtime.py']
from mmpose.configs._base_.datasets.xs2d import \
    datasets_info as kpt2d_datasets_info

train_epoch_num = 400
save_checkpoint_interval = 30
val_interval = 30
# runtime
train_cfg = dict(max_epochs=train_epoch_num, val_interval=val_interval)

data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'
test_type = '3d'
camera_layout = 'monocular'
base_lr = 2e-4
min_lr = 2e-7

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
        eta_min=min_lr,
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

backbone_out_channels = [48, 96, 192, 384]
# model settings
model = dict(
    type='TopdownPose3DAndHeldLabelEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    backbone=dict(
        type='ResNet',
        depth='26s',
        in_channels=1,
        stem_channels=32,
        base_channels=48,
        expansion=1,
        out_indices=(3, ),
        norm_cfg=dict(type='BN', momentum=0),    # momentum = 0, free BN running state
        strides=(1, 2, 2, 1),
        zero_init_residual=False,
        bias_in_conv=False,
        out_channels=backbone_out_channels),
    head=dict(
        type='RTMCCIPRHead3DAndClsHead',
        in_channels=384,
        out_channels=21,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2,
        final_layer_kernel_size=3,
        deploy_output='feat',
        output_sigma=False,
        with_gau=False,
        mlp_with_conv=False,
        gau_cfg=dict(
            hidden_dims=128,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        # loss=dict(type='FocalLoss'),
        loss=dict(type='BCELoss',
                with_logits=True),
        decoder=codec),
    test_cfg=dict(flip_test=False, ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/home/byzhou/code/mmpose/work_dirs/new_dataset/rtmtiny_2d_model_20250120_fromstliu/epoch_100.pth'
    ))

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
    checkpoint=dict(save_best='ACC/precision', 
                    interval=save_checkpoint_interval,
                    rule='greater'))
# base dataset settings
backend_args = dict(backend='local')

train_2d_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.85, 1.15],
        rotate_factor=10,
        rotate_prob=0.4,
        shift_prob=0.5,
        shift_factor=0.1),
    dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    # dict(
    #     type='GroupTransformers',
    #     trans_cfg_list=[
    #         dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.5),
    #         dict(
    #             type='Albumentation',
    #             transforms=[
    #                 dict(
    #                     type='CoarseDropout',
    #                     p=0.5,
    #                     max_holes=2,
    #                     max_height=16,
    #                     max_width=16,
    #                 ),
    #             ]),
    #         dict(
    #             type='GenerateNoiseDarkImage',
    #             prob=0.65,
    #             gamma_limit=(0.85, 0.95),
    #             alpha_limit=(0.2, 0.5),
    #             concat_image=False),
    #     ],
    #     enable_epoch_num=int(train_cfg['max_epochs']) - 20),
    dict(type='GenerateTarget', encoder=codec2d),
    dict(type='PackPoseHoldLabelInputs')
]

train_bbox_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.85, 1.15],
        rotate_factor=10,
        rotate_prob=0.4,
        shift_prob=0.5,
        shift_factor=0.1),
    dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    # dict(
    #     type='GroupTransformers',
    #     trans_cfg_list=[
    #         dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.5),
    #         dict(
    #             type='Albumentation',
    #             transforms=[
    #                 dict(
    #                     type='CoarseDropout',
    #                     p=0.5,
    #                     max_holes=2,
    #                     max_height=16,
    #                     max_width=16,
    #                 ),
    #             ]),
    #         dict(
    #             type='GenerateNoiseDarkImage',
    #             prob=0.65,
    #             gamma_limit=(0.85, 0.95),
    #             alpha_limit=(0.2, 0.5),
    #             concat_image=False),
    #     ],
    #     enable_epoch_num=int(train_cfg['max_epochs']) - 20),
    dict(type='GenerateTarget', encoder=codec2d),
    dict(type='PackPoseHoldLabelInputs')
]

val_2d_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec2d['input_size']),
    dict(type='GenerateTarget', encoder=codec2d),
    dict(type='PackPoseHoldLabelInputs')
]

val_bbox_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec2d['input_size']),
    dict(type='GenerateTarget', encoder=codec2d),
    dict(type='PackPoseHoldLabelInputs')
]

data_mode = 'topdown'


train_2d_data_list = [
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_20230811_lmdb.json",
    # "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_20230928_lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_wrist_occ__1__binocular__lmdb.json",
]

train_bbox_data_list = [
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250214__1__binocular__lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250221__1__binocular__lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250224__1__binocular__lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250219__1__binocular__lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250304__1__binocular__lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250306__1__binocular__lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250331__1__binocular__lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250401__1__binocular__lmdb.json", # hard example
    
    # "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_held_240724__1__binocular__lmdb.json",
    # "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_held_240809__1__binocular__lmdb.json",
    # "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_held_240821__1__binocular__lmdb.json",
]

train_hold_object_types = ['phone', 'book', 'EVA', 'cup', 'game_machine', 'gamepad', 'drinks', 'notebook computer', 'Game machine']

# val_data_list = ['/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json']

val_2d_dataset_name_list = [
    'flora_static_finegrain', 'flora_dynamic', 'flora_black',
    'flora_decoration', 'ella', 'near_two_hands', 'dark_light',
    'wrist_occlusion', 'tattoo', 'bad_bg', 'black_hand'
]
val_2d_data_list = dict()

for data_name in val_2d_dataset_name_list:
    val_2d_data_list[data_name] = kpt2d_datasets_info['test_data'][data_name]
    #val_2d_data_list = [item for sublist in val_2d_data_list for item in sublist]
    val_2d_data_list[data_name] = [
        os.path.join(data_root, item) for item in val_2d_data_list[data_name]
    ]

val_2d_data_list = val_2d_data_list[val_2d_dataset_name_list[0]][:1] + ["/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_benchmark_held__1__binocular__lmdb.json"]

val_bbox_data_list = [

    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_room_20240913_2k_lmdb.json",
    "/data/AI_DATA_WX/data_hand/hand_det/det_data_coco/JSON/flora_train_hand_held_occ_250227__1__binocular__lmdb.json",
]

# val_hold_object_types = ['phone', 'book', 'cup', 'gamepad']
val_hold_object_types = ['phone', 'book', 'EVA', 'cup', 'game_machine', 'gamepad', 'drinks', 'notebook computer', 'Game machine']


#val_2d_data_list = [item for sublist in val_2d_data_list for item in sublist]
#val_2d_data_list = [os.path.join(data_root, item) for item in val_2d_data_list]
train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='MultiSourceSampler',
        source_ratio=[0.5, 0.5],
        batch_size=128),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/nreal_hand.py'),
        datasets=[
            dict(
                type='HANDBboxHeldDataset',
                data_file_list=train_2d_data_list,
                # data_file_list=train_bbox_data_list,
                sample_interval=1,
                serialize_data=True,
                data_mode=data_mode,
                hold_object_types=train_hold_object_types,
                pipeline=train_2d_pipeline,
                flip_left_to_right=True,
                data_root=data_root),
            dict(
                type='HANDBboxHeldDataset',
                data_file_list=train_bbox_data_list,
                sample_interval=1,
                serialize_data=True,
                data_mode=data_mode,
                hold_object_types=train_hold_object_types,
                pipeline=train_bbox_pipeline,
                flip_left_to_right=True,
                data_root=data_root),
        ]),
)

val_2d_dataset = dict(
    type='HANDBboxHeldDataset',
    data_file_list=val_bbox_data_list,
    data_mode=data_mode,
    hold_object_types=val_hold_object_types,
    test_mode=True,
    pipeline=val_2d_pipeline,
    flip_left_to_right=True,
    data_root=data_root)
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=val_2d_dataset)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type='SimpleAccuracy')

test_evaluator = val_evaluator
