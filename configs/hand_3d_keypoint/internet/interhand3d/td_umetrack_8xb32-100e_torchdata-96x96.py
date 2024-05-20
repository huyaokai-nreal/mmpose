_base_ = ['../../../_base_/default_runtime.py']

# visualization
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=200, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0002))

# learning policy

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
train_batch = 64
auto_scale_lr = dict(base_batch_size=train_batch)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='MPJPE_all',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=10),
)

# model settings
model = dict(
    type='TopdownUmetrack',
    backbone=dict(
        type='UmeTrackModel',
        input_size=(96, 96),
        loss=dict(type='UmetrackLoss')),
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint='/home/liyilin/workspace/UmeTrack/pretrained_models/pretrained_mmpose_weights.torch'
    # ),
)

# base dataset settings
dataset_type = 'Umetrack3DDataset'
data_mode = 'topdown'
data_root = '/data/UmeTrack_data2/UmeTrack_data/torch_data/'
data_root = '/data/UmeTrack_data2/UmeTrack_data \
    /torch_data/real/separate_hand'

# pipelines
train_pipeline = [
    dict(
        type='PackPoseInputs',
        meta_keys=('extrinsics_xf', 'intrinsics', 'preds_targets',
                   'gt_skel_targets', 'hand_idx', 'orig_pose_data',
                   's_solved_pose_data'))
]
val_pipeline = [
    dict(
        type='PackPoseInputs',
        meta_keys=('extrinsics_xf', 'intrinsics', 'preds_targets',
                   'gt_skel_targets', 'hand_idx', 'orig_pose_data',
                   's_solved_pose_data'))
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch,  # 16  2
    num_workers=8,  # 8
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root,
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=train_batch,  # 128
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root,
        pipeline=val_pipeline,
        test_mode=True,
    ))
test_dataloader = dict(
    batch_size=train_batch,  # 64
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root,
        pipeline=val_pipeline,
        data_ratio=0.5,
        test_mode=True,
    ))

# evaluators
val_evaluator = [dict(type='UmetrackMetric', modes=['MPJPE'])]
test_evaluator = val_evaluator

find_unused_parameters = True

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # this will slow the training process ???
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
