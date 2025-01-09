# flake8: noqa
import os
import random

_base_ = ['../../../_base_/default_runtime.py']
from mmpose.configs._base_.datasets.xs3d import \
    datasets_info as kpt3d_datasets_info
from mmpose.configs._base_.datasets.xs3d_nimble import \
    datasets_info as kpt3d_datasets_info_nimble

# from configs._base_.datasets.xs3d import datasets_info as kpt3d_datasets_info_nimble

train_cfg = dict(max_epochs=60, val_interval=5)

data_root = '/data/AI_DATA_WX'
seq_length = 4

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-6, weight_decay=1e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        custom_keys={
            'backbone': dict(lr_mult=0),
            'head': dict(lr_mult=0),
            'neck': dict(lr_mult=0),
            # 'kpt3d_lift': dict(lr_mult=1),
        }),
    # clip_grad=dict(max_norm=10, norm_type=2),
)
# learning policy
# param_scheduler = [
#     dict(  # scheduler
#         type='MultiStepLR',
#         begin=0,
#         end=100,
#         milestones=[50, 90],
#         gamma=0.1,
#         by_epoch=True)
# ]

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=train_cfg['max_epochs'],
        convert_to_iter_based=True)
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

pinch_thre = [20, 40]  # pinch双阈值，单位：mm
kpt2d_with_depth = False  # liftnet 是否使用2.5d的深度信息
standard_stereo = True  # 是否转换标准双目
# model settings
backbone_out_channels = [48, 96, 192, 384]
model = dict(
    type='TopdownPoseLiftNimbleEstimatorSeqMono',
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
        input_size=(128, 128, 128),
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2,
        final_layer_kernel_size=3,
        deploy_output='feat',
        output_sigma=False,
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
                dict(type='L1Loss', use_target_weight=False),
                dict(type='L1Loss', use_target_weight=False),
            ]),
        decoder=codec),
    kpt3d_lift=dict(
        type='TemporalLiftNimbleHeadStandardE2e2DMono',
        lift_loss=dict(
            type='MultipleLossWrapper',
            losses=[
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
                    loss_weight=5,
                    enable_start_epoch=train_cfg['max_epochs'] // 2),
                dict(type='L1Loss', loss_weight=1),  # nimble trans直接监督
                dict(
                    type='MPJPAELoss',
                    seq_length=seq_length,
                    loss_weight=1,
                    enable_start_epoch=train_cfg['max_epochs'] // 2),
                dict(
                    type='MPJPAELoss',
                    seq_length=seq_length,
                    loss_weight=1,
                    enable_start_epoch=train_cfg['max_epochs'] // 2),
                dict(
                    type='RLELoss',
                    dim=3,
                    use_target_weight=True,
                    enable_start_epoch=train_cfg['max_epochs'] // 2),
                dict(type='L1Loss', loss_weight=0.),  # leftcam: 2d loss
                dict(type='L1Loss', loss_weight=1.),  # leftcam: reproj 2d loss
            ]),
        seq_len=4,
        all_use_kp2d_gt=False,
        kpt2d_with_depth=kpt2d_with_depth,
        use_svd=True,
        lambda_t=train_cfg['max_epochs'],
        reproj=True,
        pose_ncomp=30,
        use_6d_pose_reg=False,
        use_9d_pose_reg=True,
        use_shape_smooth=True,
        data_flip_aug=True,
        reproj_thre=440,
        iou_thre=0.5,
        mono=True,
        random_camera=0),  # 0为左单目，1为右单目
    e2e=True,
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'work_dirs/liftnimble_res26sw_e2e/bino/res26sw_liftnimble_standard_seq_e2e_trans_lr5e-6_data0.3_2d1_data2d_1121data_loss_2d0.1/epoch_20.pth'
    ),
)

# base dataset settings
dataset_type = 'PairHand3DDatasetSeq'
data_mode = 'topdown'

train_data_list = []
train_date_list = [
    '20230824', '20230828', '20230906', '20230907', '20240220', '20240229',
    '20240401', '20231227', '20240517', '20240425', '20240522', '20240801',
    '20240816', '20240826', '20240820', '20240903', '20240907', '20240926',
    '20240914', '20240923', '20240930', '20241018', '20241030', '20241107',
    '20241121', '20241114', '20241216'
]
train_glasses_list = ['Flora302', 'Flora304']
for data_date in train_date_list:
    for glasses in train_glasses_list:
        if data_date in kpt3d_datasets_info_nimble['train_data']:
            train_data_list += kpt3d_datasets_info_nimble['train_data'][
                data_date].get(glasses, [])
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
overlap_train_data_list = [
    os.path.join(data_root, item)
    for item in kpt3d_datasets_info_nimble['overlap_train_data']
]

# train_data_list = [
#     '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora301.json',
# ]
# overlap_train_data_list = [
#     '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora301.json',
# ]

dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)
overlap_dataset_weight_list = [1.0 / len(overlap_train_data_list)
                               ] * len(overlap_train_data_list)
train_2d_data_list = [
    os.path.join(data_root, item)
    for item in kpt3d_datasets_info['convert_2d_to_3d']
]
# train_2d_data_list = ['/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/convert2d_to_3d_e2e/hand_train_flora_e2e_24121220__1__binocular__lmdb.json']
dataset2d_weight_list = [1.0 / len(train_2d_data_list)
                         ] * len(train_2d_data_list)
val_data_list = [
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_075728__all__dark__left__1111__0019__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080158__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080536__pinch__bright__left__1111__0019__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081151__pinch__dark__right__1111__0019__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301.json'

    # '/home/ykhu/new_space/mmpose/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301_bak.json'

    # 边缘半手
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20240508_032300__pinch__normal__right__1101__0006__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20240508_032919__pinch__normal__right__1101__0006__undistort_tar__Flora301.json',

    # 静止
    # 'data_hand/hand_keypoint/annotations3d/fit3d_seqsmooth_auto__binocular_coco/XS__20240516_070512__all__normal__left__1101__0007__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit3d_seqsmooth_auto__binocular_coco/XS__20240516_065421__all__normal__right__1101__0007__undistort_tar__Flora301.json'

    # 黑人
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_054705__all__normal__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_055912__all__bright__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_060453__pinch__bright__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_061312__pinch__bright__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_061915__pinch__normal__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_062634__pinch__dark__right__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_062946__pinch__dark__left__1111__0023__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora301/XS__20230911_063405__pinch__normal__right__1111__0023__undistort_tar__Flora301.json'

    # 反向pinch
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083647__pinch__normal__right__1110__0025__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/manual_fix_kpt/XS__20240229_083322__pinch__normal__left__1110__0025__undistort_tar__Flora301.json',

    # 握拳pinch
    # 'data_hand/hand_keypoint/annotations3d/fit3d_merge_seqsmooth__binocular_coco/XS__20240816_101603__pinch__normal__left__1101__0033__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit3d_merge_seqsmooth__binocular_coco/XS__20240816_101710__pinch__normal__right__1101__0033__undistort_tar__Flora301.json'

    # 竖向 pinch
    # 'data_hand/hand_keypoint/annotations3d/fit3d_merge_seqsmooth__binocular_coco/XS__20240926_061543__pinch__normal__right__1101__0033__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit3d_merge_seqsmooth__binocular_coco/XS__20240926_062106__pinch__normal__left__1101__0033__undistort_tar__Flora301.json',

    # 自然场景
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301__marker_20240711114343.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301__marker_20240711113546.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301__marker_20240711113546.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301__marker_20240711113546.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301__15F_office.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301__marker_20240711112656.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301__15F_office.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301__marker_20240711114913.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora301__marker_20240711113546.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_075728__all__dark__left__1111__0019__undistort_tar__Flora301__marker_20240711112656.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_080158__pinch__normal__right__1111__0019__undistort_tar__Flora301__marker_20240711113546.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_080536__pinch__bright__left__1111__0019__undistort_tar__Flora301__marker_20240711112656.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora301__marker_20240711110634.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_081151__pinch__dark__right__1111__0019__undistort_tar__Flora301__marker_20240711114343.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora301__kangyingjiayuan_house_random.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_nimble_hand_test/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301__kangyingjiayuan_house_random.json',
]
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
# pipelines
train_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(type='TopdownPCL', input_size=codec['input_size']),
            dict(
                type='GenerateNoiseDarkImage',
                prob=0.5,
                gamma_limit=(0.85, 0.95),
                alpha_limit=(0.2, 0.5),
                concat_image=False),
            # dict(type='RandomMonocularOcclusionv2')
        ],
        enable_epoch_num=int(train_cfg['max_epochs'])),
    dict(type='PackPoseInputs')
]
train_2d_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(type='TopdownPCL2D', input_size=codec['input_size']),
            dict(
                type='GenerateNoiseDarkImage',
                prob=0.5,
                gamma_limit=(0.85, 0.95),
                alpha_limit=(0.2, 0.5),
                concat_image=False),
        ],
        enable_epoch_num=int(train_cfg['max_epochs'])),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(type='TopdownPCL', input_size=codec['input_size']),
            dict(
                type='GenerateNoiseDarkImage',
                prob=0,
                gamma_limit=(0.85, 0.95),
                alpha_limit=(0.2, 0.5),
                concat_image=False),
        ],
        enable_epoch_num=0),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    # drop_last=True,
    sampler=dict(
        type='MultiSourceSampler', source_ratio=[1, 1], batch_size=128),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/nreal_hand.py'),
        datasets=[
            dict(
                type=dataset_type,
                data_file_list=train_data_list,
                data_mode=data_mode,
                pipeline=train_pipeline,
                dataset_weight_list=dataset_weight_list,
                data_root=data_root,
                seq_len=seq_length,
                flip_left_to_right=True,
                sample_interval=4. / 5,
                data_ratio=1. / 3,
                serialize_data=True,
            ),
            dict(
                type=dataset_type,
                data_file_list=train_2d_data_list,
                data_mode=data_mode,
                pipeline=train_2d_pipeline,
                dataset_weight_list=dataset2d_weight_list,
                data_root=data_root,
                seq_len=seq_length,
                flip_left_to_right=True,
                sample_interval=4. / 5,
                data_ratio=1. / 3,
                serialize_data=True,
            ),
        ]),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(
        type='DistributedRangeSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='PairHand3DDataset',
        data_file_list=val_data_list,
        data_mode=data_mode,
        test_mode=True,
        pipeline=val_pipeline,
        flip_left_to_right=True,
        data_root=data_root,
        standard_stereo=True),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='all_mpjpe', rule='less'),
    run_time_info=dict(type='RuntimeInfoHookV2'))

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]
# val_evaluator = dict(type='MPJPEMetricLifting', gesture_list=gesture_list)
filter_exceed = True
val_evaluator = [
    dict(
        type='MPJPEV2',
        mode=['mpjpe', 'p-mpjpe'],
        score_metric=True,
        # gesture_list=gesture_list,
        rearrange_result=True,
        result_dir='.'),
    # dict(type='EPE'),
    dict(type='NrealKeypointAP'),
]

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
