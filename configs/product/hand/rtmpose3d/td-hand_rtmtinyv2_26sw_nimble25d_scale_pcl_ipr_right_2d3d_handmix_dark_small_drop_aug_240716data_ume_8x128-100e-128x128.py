# flake8: noqa
import os

_base_ = ['../../../_base_/default_runtime.py']
from mmpose.configs._base_.datasets.xs2d import \
    datasets_info as kpt2d_datasets_info
from mmpose.configs._base_.datasets.xs3d import \
    datasets_info as kpt3d_datasets_info
from mmpose.configs._base_.datasets.xs3d_nimble import \
    datasets_info as kpt3d_datasets_info_nimble
from mmpose.configs._base_.datasets.xs3d_ume import datasets_info as kpt3d_ume

# runtime
train_cfg = dict(max_epochs=100, val_interval=10)

data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'
test_type = '3d'
camera_layout = 'monocular'
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

backbone_out_channels = [48, 96, 192, 384]
# model settings
model = dict(
    type='TopdownPose3DEstimator',
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
        strides=(1, 2, 2, 1),
        zero_init_residual=False,
        bias_in_conv=False,
        out_channels=backbone_out_channels),
    head=dict(
        type='RTMCCIPRHead3D',
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
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', use_target_weight=False),
                dict(type='L1Loss', use_target_weight=False),
            ]),
        decoder=codec),
    test_cfg=dict(flip_test=False, ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/data/AI_DATA/data_hand/model/mmpose/td-hand_rtmtinyv2_26sw_25d_scale_pcl_right_2d3d_handmix_aug_240401data_8x128-100e-128x128/epoch_100.pth'
    ),
    root_mode='optimize' if test_type == '3d' else 'gt',
    camera_layout=camera_layout)

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
    checkpoint=dict(save_best='all_mpjpe', rule='less'))
# base dataset settings
backend_args = dict(backend='local')
train_pipeline = [
    dict(type='KeypointTo25DLabel', norm_depth=True),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=15,
        rotate_prob=0,
        shift_prob=0.5,
        shift_factor=0.2),
    dict(type='TopdownPCL', input_size=codec['input_size'][:2]),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.2),
            dict(type='MixTwoHands', prob=0.5),
            dict(
                type='Albumentation',
                transforms=[
                    dict(
                        type='CoarseDropout',
                        p=0.5,
                        max_holes=2,
                        max_height=16,
                        max_width=16,
                    ),
                ]),
            dict(
                type='GenerateNoiseDarkImage',
                prob=0.65,
                gamma_limit=(0.85, 0.95),
                alpha_limit=(0.2, 0.5),
                concat_image=False),
        ],
        enable_epoch_num=int(train_cfg['max_epochs']) - 20),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
train_2d_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=15,
        rotate_prob=0.3,
        shift_prob=0.5,
        shift_factor=0.2),
    dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.5),
            dict(type='MixTwoHands', prob=0.5),
            dict(
                type='Albumentation',
                transforms=[
                    dict(
                        type='CoarseDropout',
                        p=0.5,
                        max_holes=2,
                        max_height=16,
                        max_width=16,
                    ),
                ]),
            dict(
                type='GenerateNoiseDarkImage',
                prob=0.65,
                gamma_limit=(0.85, 0.95),
                alpha_limit=(0.2, 0.5),
                concat_image=False),
        ],
        enable_epoch_num=int(train_cfg['max_epochs']) - 20),
    dict(type='GenerateTarget', encoder=codec2d),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='KeypointTo25DLabel', norm_depth=True),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownPCL', input_size=codec['input_size'][:2]),
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
train_data_list = []
train_date_list = [
    '20230809', '20230815', '20230817', '20230822', '20230824', '20230828',
    '20230906', '20230907', '20231031', '20240220', '20240229', '20240401',
    '20231227', '20240517', '20240425', '20240522'
]
train_glasses_list = ['Flora301', 'Flora302', 'Flora303', 'Flora304']
for data_date in train_date_list:
    for glasses in train_glasses_list:
        if data_date in kpt3d_datasets_info_nimble['train_data']:
            train_data_list += kpt3d_datasets_info_nimble['train_data'][
                data_date].get(glasses, [])
        else:
            train_data_list += kpt3d_datasets_info['train_data'][
                data_date].get(glasses, [])

#simulate_data_keys = ['marker']
#for data_date in simulate_data_keys:
#    for glasses in train_glasses_list:
#        train_data_list += kpt3d_datasets_info_nimble['simu_train_data'][
#            data_date].get(glasses, [])
for hand in ['left', 'right']:
    train_data_list += kpt3d_ume['separate_hand']['training'][hand]
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)
train_2d_datasets = ['ella', 'flora', 'quest_system', 'hoi']
train_2d_data_list = [
    kpt2d_datasets_info['train_data'][key] for key in train_2d_datasets
]
train_2d_data_list = [
    item for sublist in train_2d_data_list for item in sublist
]
train_2d_data_list = [
    os.path.join(data_root, item) for item in train_2d_data_list
]

val_data_list = []
val_date_list = ['20230830']
val_glasses_list = ['Flora301']
val_person_list = ['0005']

for data_date in val_date_list:
    for glasses in val_glasses_list:
        val_data_list += kpt3d_datasets_info['test_data'][data_date].get(
            glasses, [])
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
val_data_list = [
    item for item in val_data_list if item.split('__')[-3] in val_person_list
]
val_2d_dataset_name_list = [
    #'flora_static_finegrain', 'flora_dynamic',
    #'flora_black',
    #'flora_decoration',
    #'ella',
    #'near_two_hands',
    #'dark_light',
    #'wrist_occlusion',
    #'tattoo'
    'bad_bg'
]
val_2d_data_list = dict()
#val_data_list = [item for item in val_data_list if '__right__' in item]
#val_2d_datasets = ['flora_static_finegrain', 'flora_dynamic']
#val_2d_datasets = ['flora_black']
#val_2d_datasets = ['flora_decoration']
#val_2d_datasets = ['ella']
#val_2d_datasets = ['near_two_hands']
#val_2d_datasets = ['dark_light']
#val_2d_datasets = ['wrist_occlusion']
#val_2d_datasets = ['tattoo']
for data_name in val_2d_dataset_name_list:
    val_2d_data_list[data_name] = kpt2d_datasets_info['test_data'][data_name]
    #val_2d_data_list = [item for sublist in val_2d_data_list for item in sublist]
    val_2d_data_list[data_name] = [
        os.path.join(data_root, item) for item in val_2d_data_list[data_name]
    ]
val_2d_datasets = ['bad_bg']
val_2d_data_list = [
    kpt2d_datasets_info['test_data'][key] for key in val_2d_datasets
]
val_2d_data_list = [item for sublist in val_2d_data_list for item in sublist]
val_2d_data_list = [os.path.join(data_root, item) for item in val_2d_data_list]
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
                sample_interval=3,
                filter_kpt_exceed=True,
                data_ratio=1 / 10.0,
                data_file_list=train_data_list,
                data_mode=data_mode,
                pipeline=train_pipeline,
                dataset_weight_list=dataset_weight_list,
                data_root=data_root,
                flip_left_to_right=True,
                point_type='2.5D',
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
val_3d_dataset = dict(
    type=dataset_type,
    data_file_list=val_data_list,
    data_mode=data_mode,
    test_mode=True,
    pipeline=val_pipeline,
    flip_left_to_right=True,
    point_type='2.5D' if camera_layout == 'monocular' else '3D',
    data_root=data_root)
val_2d_dataset = dict(
    type='HANDDataset',
    data_file_list=val_2d_data_list,
    data_mode=data_mode,
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
    dataset=val_2d_dataset if test_type == '2d' else val_3d_dataset)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [dict(type='EPE'), dict(type='NrealKeypointAP', with_tag=True)]
if test_type == '3d':
    val_evaluator += [
        dict(type='MPJPEV2', mode=['mpjpe', 'p-mpjpe'], result_dir='.'),
    ]
test_evaluator = val_evaluator
