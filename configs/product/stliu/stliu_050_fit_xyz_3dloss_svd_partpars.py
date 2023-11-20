# flake8: noqa
import os

# from configs._base_.datasets.xs3d import datasets_info as kpt3d_datasets_info

_base_ = ['../../_base_/default_runtime.py']

train_cfg = dict(max_epochs=140, val_interval=10)

data_root = '/data/AI_DATA'
# data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4, weight_decay=1e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'head': dict(lr_mult=0.0, decay_mult=0.0),
            'neck': dict(lr_mult=0.0, decay_mult=0.0),
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
# model settings
backbone_out_channels = [64, 96, 128, 160]
model = dict(
    type='TopdownPoseLiftEstimator',
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
        output_depth=True,
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
        output_sigma=False),
    kpt3d_lift=dict(
        type='LiftHead_Rotation',
        lift_loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', loss_weight=1),  # 3d kpts
                dict(type='L1Loss', loss_weight=1),  # 3d kpts leftcam
                dict(type='L1Loss', loss_weight=1),  # 3d kpts rightcam
                dict(type='MSELoss', loss_weight=0, enable_start_epoch=train_cfg['max_epochs'] - 40),  # 2d reprojection left
                dict(type='MSELoss', loss_weight=0, enable_start_epoch=train_cfg['max_epochs'] - 40),  # 2d reprojection right
                dict(
                    type='PinchLoss',
                    enter_thre=pinch_thre[0] / 1000,
                    exit_thre=pinch_thre[1] / 1000,
                    loss_weight=1,
                    enable_start_epoch=train_cfg['max_epochs'] -
                    20),  # 后20 epoch打开pinch loss
                dict(type='L1Loss', loss_weight=0, enable_start_epoch=train_cfg['max_epochs'] - 40),  # xyz比例损失
                dict(type='MSELoss', loss_weight=0),  # nimble pose直接监督
                dict(type='MSELoss', loss_weight=5),  # nimble trans直接监督
            ]),
        channel_num=55,
        use_kp2d_gt=False,
        kpt2d_with_depth=kpt2d_with_depth,
        output_num=83,
        undistort=True,
        use_svd=True,
        use_nimble_part_para = True,
        lambda_t = train_cfg['max_epochs'],
        # pre_xyz_type = 0/1/2, 0 指的是 pre的3个xyz都是通过nimble获得的， 1指的是3个pre的z是通过nimble获得的而xy通过uv计算得到的，2指的是3个pre分别为nimble z+origin xy、nimble xyz、前两个值的平均值。
        pre_xyz_type = 0), 
    test_cfg=dict(
        flip_test=False,
        shift_coords=False,
        shift_heatmap=False,
    ),
    kpt2d_with_depth=kpt2d_with_depth,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        # '/home/zx_li/workspace/mmpose/work_dirs/td-hand_res26_fpn_sk_weightdata_4xb64-50e_0919data-128x128/epoch_50.pth'
        # '/data/AI_DATA/data_hand/model/mmpose/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128/epoch_50.pth'  # ella kp2d
        # '/home/jrchen/git-project/mmpose/work_dirs/pair_hand3d/003_td-stage_two_train_55dim_l1/epoch_60_new.pth'
        # '/home/jrchen/git-project/mmpose/work_dirs/hand_2d_keypoint/td-hand_res26_fpn_skpre_flow_wd_ipr_rle_weightdata_0919_4xb64-50e-128x128_pretrainmodel/epoch_30_FT.pth'
        #f'/data/AI_DATA/jrchen/git-project/mmpose/work_dirs/pair_hand3d/006_td-stage_two_train_55dim_RLE_head_train_flora_finetune/FT_kp2d_add_ella_pretrain_lift.pth'  # flora kp2d + ella kp3d
        # '/home/zx_li/workspace/mmpose/work_dirs/td-hand_rsn26_fpn_25d_ipr_right_pcl_2d3d_4x128-100e-128x128/epoch_100.pth',
        # 'work_dirs/keypoint25d/01_td-hand_rsn26_fpn_25d_ipr_right_pcl_2d3d_4x128-100e-128x128_Affine/best_all_p-mpjpe_epoch_90.pth'
        '/data/AI_DATA/data_hand/model/mmpose/td-hand_rsn26_fpn_25d_scale_ipr_right_2d3d_0915data_simu_4x128-100e-128x128/epoch_100.pth'
    ),
)

# base dataset settings
dataset_type = 'PairHand3DDataset'
data_mode = 'topdown'

# train_data_list = []
# train_date_list = [
#     '20230809', '20230815', '20230817', '20230822', '20230824', '20230828',
#     '20230906', '20230907'
# ]
# train_glasses_list = ['Flora301', 'Flora303']

# for data_date in train_date_list:
#     for glasses in train_glasses_list:
#         train_data_list += kpt3d_datasets_info['train_data'][data_date].get(
#             glasses, [])
# simu_date_list = ['20230809']
# simu_glasses_list = ['Flora301']
# for data_date in simu_date_list:
#     for glasses in simu_glasses_list:
#         train_data_list += kpt3d_datasets_info['simu_train_data'][
#             data_date].get(glasses, [])

# train_data_list = [os.path.join(data_root, item) for item in train_data_list]

train_data_list = [
    # 0824
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora301_update.json',
    
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_060805__all__normal__left__1111__0006__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_062443__all__normal__right__1111__0006__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_063033__all__normal__right__1111__0007__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_063544__all__normal__left__1111__0007__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_064229__all__normal__left__1111__0014__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_064807__all__normal__right__1111__0014__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_065401__all__normal__right__1111__0011__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_070036__all__normal__left__1111__0011__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_070620__all__normal__left__1111__0015__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230824_071050__all__normal__right__1111__0015__undistort_tar__Flora303_update.json',
    
    # #0828
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_062006__point__normal__right__1000__0005__quest__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_062640__all__normal__right__1000__0005__quest__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_063719__pinch__normal__right__1000__0005__quest__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora301_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora301_with_nimble/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora301_update.json',
    
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230828_071551__all__normal__left__1111__0017__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230828_072918__all__normal__right__1111__0017__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230828_073459__all__normal__left__1111__0017__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230828_073946__all__normal__right__1111__0017__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230828_075809__all__normal__left__1111__0018__undistort_tar__Flora303_update.json',
    'data_hand/hand_keypoint/annotations3d/Flora303_with_nimble/XS__20230828_080553__all__normal__right__1111__0018__undistort_tar__Flora303_update.json',
   
]

# test only
#data_root = '/data/hand_group/data/data_hand/lmdb_data/'

train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

val_data_list = [
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',  #
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
    'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',

    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_093420__all__dark__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_094228__pinch__dark__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_094637__pinch__bright__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_094851__pinch__bright__left__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_095839__all__bright__right__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_100545__pinch__normal__right__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_100822__pinch__normal__right__1111__0021__undistort_tar__Flora301.json',
    # 'data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230904_101030__pinch__bright__right__1111__0021__undistort_tar__Flora301.json'
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
        # data_ratio=1.0/30,
        data_file_list=train_data_list,
        data_mode=data_mode,
        pipeline=train_pipeline,
        dataset_weight_list=dataset_weight_list,
        data_root=data_root,
        # point_type='leftcam',
        # indices=1000,
    ),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=8,
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
        # point_type='leftcam'
    ),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(interval=10, save_best='all_mpjpe', rule='less'),
    run_time_info=dict(type='RuntimeInfoHookV2'))

# evaluators
gesture_list = [
    'Click', 'Grab', 'Pinch', 'OpenHand', 'Victory', 'Call', 'Home'
]
# val_evaluator = dict(type='MPJPEMetricLifting', gesture_list=gesture_list)

val_evaluator = [
    dict(
        type='MPJPEV2',
        mode=['mpjpe', 'p-mpjpe'],
        gesture_list=gesture_list,
        result_dir='.'),
    dict(type='EPE'),
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