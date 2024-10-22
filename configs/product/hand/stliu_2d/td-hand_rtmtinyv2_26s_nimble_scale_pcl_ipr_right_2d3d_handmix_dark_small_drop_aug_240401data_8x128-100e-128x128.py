# flake8: noqa
import os

_base_ = ['../../../_base_/default_runtime.py']

from mmpose.configs._base_.datasets.xs3d_nimble import \
    datasets_info as kpt3d_datasets_info
from mmpose.configs._base_.datasets.xs3d_ume import datasets_info as kpt3d_ume

# runtime
train_cfg = dict(max_epochs=100, val_interval=5)
find_unused_parameters = True

data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'
test_type = '3d'
camera_layout = 'nimble'
base_lr = 1e-5
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

pinch_thre = [20, 40]  # pinch双阈值，单位：mm
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
        type='RTMCCIPRHeadNimble',
        in_channels=384,
        out_channels=21,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2,
        final_layer_kernel_size=3,
        deploy_output='feat',
        output_sigma=True,
        with_gau=False,
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
                # dict(type='L1Loss', use_target_weight=False,
                #      loss_weight=0.5),
                # dict(type='L1Loss', use_target_weight=False),
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # 3d kpts
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # 3d kpts leftcam
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # 3d kpts rightcam
                dict(
                    type='PinchLoss',
                    enter_thre=pinch_thre[0] / 1000,
                    exit_thre=pinch_thre[1] / 1000,
                    loss_weight=30,
                    enable_start_epoch=train_cfg['max_epochs'] // 2),
                dict(type='MSELoss', loss_weight=100),  # nimble trans直接监督
                dict(
                    type='RLELoss',
                    dim=3,
                    enable_start_epoch=train_cfg['max_epochs'] // 2),
                # dict(type='MSELoss', loss_weight=0.002,
                #      enable_start_epoch=train_cfg['max_epochs']-20),
                dict(type='L1Loss', use_target_weight=False),
            ]),
        decoder=codec),
    test_cfg=dict(flip_test=False, ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/data/AI_DATA_WX/share/zuoxin/mmpose/work_dirs/td-hand_rtmtinyv3_26sw_nimble25d_scale_pcl_ipr_right_2d3d_hand_mix_dark_small_drop_aug_aio_240926data_8x128-100e-128x128/epoch_100.pth'
    ),
    root_mode='optimize' if test_type == '3d' else 'gt',
    camera_layout=camera_layout)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]

visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# default_hooks = dict(
#     #visualization=dict(
#     #    type='PoseVisualizationHook', enable=True, draw_3d=True),
#     checkpoint=dict(save_best='all mAP', rule='less'))

default_hooks = dict(
    checkpoint=dict(interval=5, save_best='all_mpjpe', rule='less'),
    run_time_info=dict(type='RuntimeInfoHookV2'))

# base dataset settings
backend_args = dict(backend='local')
train_pipeline = [
    dict(type='KeypointTo25DLabel', norm_depth=True),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(
                type='RandomBBoxTransform',
                scale_factor=[0.75, 1.25],
                rotate_factor=15,
                rotate_prob=0.3,
                shift_prob=0.5,
                shift_factor=0.2),
            dict(type='TopdownPCL', input_size=codec['input_size'][:2]),
            dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.2),
            dict(type='MixTwoHands', prob=0.1),
            dict(
                type='Albumentation',
                transforms=[
                    dict(
                        type='CoarseDropout',
                        p=0.2,
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
        enable_epoch_num=int(train_cfg['max_epochs'])),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='KeypointTo25DLabel', norm_depth=True),
    dict(type='GetBBoxCenterScale', padding=1.0),
    # dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    dict(type='TopdownPCL', input_size=codec['input_size'][:2]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

dataset_type = 'PairHand3DDataset'
data_mode = 'topdown'
train_data_list = []
train_date_list = [
    '20230824', '20230828', '20230906', '20230907', '20240220', '20240229',
    '20240401', '20231227', '20240517', '20240425', '20240522', '20240801',
    '20240816', '20240826', '20240820', '20240903', '20240907', '20240926',
    '20240914', '20240923', '20240930'
]
train_glasses_list = ['Flora301', 'Flora302', 'Flora303', 'Flora304']
for data_date in train_date_list:
    for glasses in train_glasses_list:
        if data_date in kpt3d_datasets_info['train_data']:
            train_data_list += kpt3d_datasets_info['train_data'][
                data_date].get(glasses, [])

# for hand in ['left', 'right']:
#     train_data_list += kpt3d_ume['separate_hand']['training'][hand]

# simulate_data_keys = ['marker']
# for data_date in simulate_data_keys:
#     for glasses in train_glasses_list:
#         train_data_list += kpt3d_datasets_info['simu_train_data'][
#             data_date].get(glasses, [])
# for hand in ['left', 'right']:
#     for v in kpt3d_ume[hand].values():
#         train_data_list += v
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
# train_data_list = [train_data_sin for train_data_sin in train_data_list if "__left__" in train_data_sin]
# train_data_list = [
#     '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora301.json',
#     # '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora301.json'
#     # '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/simulate_binocular_coco_hand/XS__20230907_031309__all__normal__left__1111__0011__undistort_tar__Flora301__marker_20240711110634.json',
#     # '/data/AI_DATA/data_hand/hand_keypoint/annotations3d/ume_data/left/training/user_20/recording_19.json'
# ]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)


val_data_list = []
val_date_list = ['20230830']
val_glasses_list = ['Flora301']
val_person_list = ['0005']

for data_date in val_date_list:
    for glasses in val_glasses_list:
        val_data_list += kpt3d_datasets_info['test_data'][data_date].get(
            glasses, [])
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
# val_data_list = [val_data_sin for val_data_sin in val_data_list if "__left__" in val_data_sin]

# val_data_list = [
#     # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301.json',
#     '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
# ]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        # data_ratio=1.0/30,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        flip_left_to_right=False,
        filter_kpt_exceed=True,
        point_type='2.5D',
        # standard_stereo=standard_stereo
        # point_type='leftcam',
        # indices=1000,
    ),
)

val_3d_dataset = dict(
    type=dataset_type,
    data_file_list=val_data_list,
    data_mode=data_mode,
    # hand template from outside algorithm, such binocular pipeline
    #extern_hand_template_path = '/home/zx_li/workspace/mmpose/work_dirs/binocular_hand_template.npy',
    test_mode=True,
    pipeline=val_pipeline,
    flip_left_to_right=False,
    # mean_bone_template_path=
    # '/data/AI_DATA/data_hand/model/mmpose/mean_hand_bones_230824.npz',
    #point_type='leftcam',
    point_type='2.5D' if camera_layout in  ['monocular', 'nimble'] else '3D',
    data_root=data_root)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=val_3d_dataset)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [dict(type='EPE'), dict(type='NrealKeypointAP', with_tag=True)]
if test_type == '3d':
   val_evaluator += [
       dict(
           type='MPJPEV2',
           mode='mpjpe',
           scale_metric=False,
           with_tag=True,
       ),
       dict(type='MPJPEV2', mode='p-mpjpe', prefix='1'),
   ]
test_evaluator = val_evaluator
