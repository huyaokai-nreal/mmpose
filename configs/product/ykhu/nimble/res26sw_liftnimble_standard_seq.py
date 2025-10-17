# flake8: noqa
import os

_base_ = ['../../../_base_/default_runtime.py']
from mmpose.configs._base_.datasets.xs3d_nimble import \
    datasets_info as kpt3d_datasets_info

# from configs._base_.datasets.xs3d import datasets_info as kpt3d_datasets_info

train_cfg = dict(max_epochs=30, val_interval=5)

# data_root = '/data/AI_DATA'
data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'
seq_length = 8

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-5, weight_decay=1e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'head': dict(lr_mult=0.0, decay_mult=0.0),
            'neck': dict(lr_mult=0.0, decay_mult=0.0),
        }),
    clip_grad=dict(max_norm=10, norm_type=2),
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
    type='TopdownPoseLiftNimbleEstimatorSeq',
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
        type='TemporalLiftNimbleHeadStandard',
        lift_loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # 预测根节点位置、旋转约束
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # 预测局部手势约束
                dict(type='L1Loss', use_target_weight=True,
                     loss_weight=1),  # 预测整体约束
                dict(
                    type='PinchLoss',
                    enter_thre=pinch_thre[0] / 1000,
                    exit_thre=pinch_thre[1] / 1000,
                    loss_weight=5),
                dict(type='L1Loss', loss_weight=10),  # 根节点监督
                dict(
                    type='MPJPAELoss',
                    seq_length=seq_length,
                    loss_weight=20,
                    # enable_start_epoch=train_cfg['max_epochs']//2
                ),
                dict(
                    type='MPJPAELoss',
                    seq_length=seq_length,
                    loss_weight=20,
                    # enable_start_epoch=train_cfg['max_epochs']//2
                ),
                dict(
                    type='RLELoss',
                    dim=3,
                    use_target_weight=True,
                    # enable_start_epoch=train_cfg['max_epochs']//2,
                    flow_model_pretrain_path=
                    '/data/AI_DATA_WX/ykhu/model/liftnimble_20250815_depthaug.pth'),
                dict(type='L1Loss', loss_weight=1.),  # 左目重投影误差
                dict(type='L1Loss', loss_weight=1.),  # 右目重投影误差
                dict(type='L1Loss', loss_weight=20.),  # 根节点深度
            ]),
        seq_len=seq_length,
        all_use_kp2d_gt=False,
        kpt2d_with_depth=kpt2d_with_depth,
        undistort=True,
        use_svd=True,
        max_epochs=train_cfg['max_epochs'],
        pose_ncomp=30,
        baseline=0.135,
        use_6d_pose_reg=False,
        use_9d_pose_reg=True,
        use_shape_smooth=True,
        data_flip_aug=True,
        reproj_thre=440,
        iou_thre=0.5,
        pad_2d=0,
        fix_sigma_pars=False,
        reproj=True),
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    init_cfg=dict(type='Pretrained', checkpoint='/data/AI_DATA_WX/ykhu/model/liftnimble_20250815_depthaug.pth'),
)

# base dataset settings
dataset_type = 'PairHand3DDatasetSeq'
data_mode = 'topdown'

train_data_list = []
train_glasses_list = ['Flora301', 'Flora302', 'Flora303', 'Flora304']
for data_date in kpt3d_datasets_info['train_data']:
    for glasses in train_glasses_list:
        train_data_list += kpt3d_datasets_info['train_data'][data_date].get(
            glasses, [])

# train_data_list = [
#     'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20250228_140704__pinch__normal__right__1101__0003__undistort_tar__Flora301.json',
#     'data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora301.json',
# ]

# train_data_list += kpt3d_datasets_info['convert2d_to_3d']
# import random
# train_data_list = random.sample(train_data_list, 50)
train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    # fisheye6202数据
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20250924_111722__thumbup__normal__left__1111__0008__undistort_tar__Flora301.json',
    
    # 横向移动bmk
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20250729_110042__pinch__normal__right__1111__0004__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20250729_110250__pinch__normal__left__1111__0004__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20250804_163016__all__normal__right__1111__0002__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20250804_163247__all__normal__left__1111__0002__undistort_tar__Flora301.json',

    # 通用0005bmk
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_075728__all__dark__left__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080158__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080536__pinch__bright__left__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081151__pinch__dark__right__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301.json'

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
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301__kangyingjiayuan_house_random.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301__marker_20240711110634.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301__marker_20240711112656.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301__marker_20240711114343.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301__marker_20240711114343.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301__marker_20240711114343.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301__marker_20240711112656.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301__marker_20240711112656.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora301__marker_20240711112656.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_075728__all__dark__left__1111__0019__undistort_tar__Flora301__marker_20240711114343.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_080158__pinch__normal__right__1111__0019__undistort_tar__Flora301__kangyingjiayuan_house_random.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_080536__pinch__bright__left__1111__0019__undistort_tar__Flora301__marker_20240711113546.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora301__marker_20240711113546.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_081151__pinch__dark__right__1111__0019__undistort_tar__Flora301__marker_20240711110634.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora301__kangyingjiayuan_house_random.json',
    # 'data_hand/hand_keypoint/annotations3d/overlap_binocular_coco_hand_test/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301__marker_20240711112656.json',
]
# val_data_list = kpt3d_datasets_info['test_data']['20250219']['Flora301']  # 多背景3d真值数据
# val_data_list = kpt3d_datasets_info['test_data']['20250328']['Flora301']  # poke
val_data_list = [os.path.join(data_root, item) for item in val_data_list]
# pipelines
train_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(
                type='RandomBBoxTransform',
                scale_factor=[0.75, 1.25],
                rotate_factor=15,
                rotate_prob=0.2,
                shift_prob=0.2,
                shift_factor=0.2),
            dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
            dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.2),
            dict(type='MixTwoHands', prob=0.1),
            # dict(
            #     type='Albumentation',
            #     transforms=[
            #         dict(
            #             type='CoarseDropout',
            #             p=0.2,
            #             max_holes=2,
            #             max_height=16,
            #             max_width=16,
            #         ),
            #     ]),
            dict(
                type='GenerateNoiseDarkImage',
                prob=0.3,
                gamma_limit=(0.85, 0.95),
                alpha_limit=(0.2, 0.5),
                concat_image=False),
        ],
        enable_epoch_num=int(train_cfg['max_epochs'])),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_ratio=0.6,
        sample_interval=0.6,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        flip_left_to_right=True,
        seq_len=seq_length,
        # point_type='leftcam',
        # indices=1000,
    ),
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
        standard_stereo=True
        # point_type='leftcam'
    ),
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
        # pinch_thre = pinch_thre, # mm
        score_metric=True,
        fit_metric=False,  # True时仅测可见, False全测
        # gesture_list=gesture_list,
        # bmk_save_root='/home/ykhu/workspace/mmpose/work_dirs/bad_case_liftnet/fisheye6202',
        # show_bmk_thr=(0, 10000000),
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
    # dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
