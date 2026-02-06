# flake8: noqa
import os

_base_ = ['../../../_base_/default_runtime.py']

from mmpose.configs._base_.datasets.hot3d import get_quest3_anno_paths
from mmpose.configs._base_.datasets.xs2d import datasets_info as kpt2d_datasets_info
from mmpose.configs._base_.datasets.xs3d import datasets_info as kpt3d_datasets_info
from mmpose.configs._base_.datasets.xs3d_nimble import datasets_info as kpt3d_datasets_info_nimble
from mmpose.configs._base_.datasets.xs3d_ume import datasets_info as kpt3d_ume

train_cfg = dict(max_epochs=100, val_interval=10)

data_root = '/data/AI_DATA_WX'
test_type = '3d'
camera_layout = 'monocular'
base_lr = 3e-5
data_mode = 'topdown'
val_source = 'interhand'

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01, betas=(0.9, 0.999)),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True,),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=5,
        start_factor=0.001,
        end_factor=1.0,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=train_cfg['max_epochs'],
        convert_to_iter_based=True)
]

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


model = dict(
    type='TopdownPose3DEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[0.449 * 255],
        std=[0.226 * 255]),
    backbone=dict(
        type='ResNet',
        depth='26s',
        in_channels=1,
        stem_channels=32,
        base_channels=48,
        expansion=1,
        out_indices=(2, 3),
        strides=(1, 2, 2, 1),
        zero_init_residual=False,
        bias_in_conv=False,
        out_channels=backbone_out_channels),

    head=dict(
        type='RTMCCIPRHead3DJointFormer',
        in_channels=384,
        out_channels=21,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=3,
        deploy=False,
        deploy_output='kpt',
        output_sigma=False,
        with_gau=False,
        mlp_with_conv=False,
        gau_cfg=dict(
            hidden_dims=128,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False,
        ),
        map_type='softmax',
        temperature=(0.85, 0.85, 1.00),
        linspace_denominator='L',
        learnable_temperature=True,
        sigma_temperature_cfg=dict(
            enable=True,
            source='hybrid',
            conf_type='entropy',
            factor_range=(0.70, 1.60),
            axiswise=True,
            detach=True,
            sigma_mode='sigmoid',
            hybrid_alpha=0.45,
            eps=1e-8,
        ),
        loss=dict(
            type='MultipleLossWrapper',
            losses=[
                dict(type='L1Loss', use_target_weight=True),
                dict(type='L1Loss', use_target_weight=True),
            ],
        ),
        decoder=codec,
        refine_cfg=dict(
            refine_feat_indices=[-2, -1],
            num_refine_levels=2,
            refine_in_channels_list=[192, 384],
            refine_upsample_scales=[1, 2],
            upsample_mode='bilinear',
            post_upsample_conv=True,

            proj_channels=128,
            deform_num_points=4,
            offset_range=(0.05, 0.05),
            use_deform_weights=True,
            padding_mode='border',
            align_corners=False,
            detach_coarse_for_sample=True,

            decoder_layers=3,
            decoder_embed_dim=128,
            decoder_heads=4,
            decoder_mlp_ratio=4.0,
            decoder_drop=0.05,
            decoder_attn_drop=0.05,
            decoder_delta_mlp_layers=2,
            decoder_sigma_mlp_layers=2,
            layer_scale_init_value=1e-5,

            enable_sigma_refine=True,
            sigma_cond_dim=16,

            sigma_nll_cfg=dict(
                enable=True,
                type='laplace',
                weight=0.01,
                min_sigma=1e-4,
                eps=1e-8,
            ),

            use_register_tokens=True,
            num_register_tokens=4,

            coords_dim=2,
            use_coord_embed=True,
            add_coords_to_cross_attn=True,
            add_coords_to_delta=True,
            refine_z=True,
            use_inv_sigmoid=True,
            delta_scale=(0.03, 0.03, 0.05),

            refine_loss_weight=1.0,
            coarse_loss_weight=0.60,

            deep_supervision=dict(
                enable=True,
                weight=0.50,
                strategy='linear',
                detach=True,
                include_final=False,
            ),

            bone_loss_weight=0.50,
            bone_loss_3d_only=True,
            bone_min_gt_len=0.01,
            bone_huber_delta=0.05,
            bone_use_relative=True,
            bone_clamp_per_bone=10.0,

            use_oks_loss=True,
            oks_loss_weight=0.60,
            oks_loss_type='1moks',
            coord_is_normalized=True,
            oks_from_bbox=False,
            oks_fallback_kpt_bbox=True,
            oks_eps=1e-4,
        ),
    ),
    test_cfg=dict(flip_test=False, ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='~/epoch_100.pth',
    ),
    root_mode='optimize' if test_type == '3d' else 'gt',
    camera_layout=camera_layout
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    checkpoint=dict(save_best='all mAP', rule='greater'))

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
    dict(
        type='TopdownPCL',
        input_size=codec['input_size'][:2],
        norm_depth=True),
    dict(
        type='GroupTransformers',
        trans_cfg_list=[
            dict(type='RandomDownSampleImage', min_ratio=0.5, prob=0.2),
            dict(type='MixTwoHands', prob=0.5),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='CoarseDropout', p=0.5, max_holes=2, max_height=16, max_width=16),
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
                    dict(type='CoarseDropout', p=0.5, max_holes=2, max_height=16, max_width=16),
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
    dict(type='TopdownPCL', input_size=codec['input_size'][:2], norm_depth=True),
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

train_data_list = []
train_date_list = [
    '20230809', '20230815', '20230817', '20230822', '20231031', '20230824',
    '20230828', '20230906', '20230907', '20240220', '20240229', '20240401',
    '20231227', '20240517', '20240425', '20240522', '20240801', '20240816',
    '20240826', '20240820', '20240903', '20240907', '20240926', '20240914',
    '20240923', '20240930', '20241018', '20241030', '20241107', '20241121',
    '20241114', '20241216', '20250107', '20250113'
]
train_glasses_list = ['Flora301', 'Flora302', 'Flora303', 'Flora304']
for data_date in train_date_list:
    for glasses in train_glasses_list:
        if data_date in kpt3d_datasets_info_nimble['train_data']:
            train_data_list += kpt3d_datasets_info_nimble['train_data'][data_date].get(glasses, [])
        else:
            train_data_list += kpt3d_datasets_info['train_data'][data_date].get(glasses, [])

train_data_list = [os.path.join(data_root, item) for item in train_data_list]
dataset_weight_list = [1.0 / len(train_data_list)] * len(train_data_list)

train_2d_datasets = ['ella', 'flora', 'quest_system', 'hoi', 'bad_bg', 'black_hand', 'e2e']
train_2d_data_list = [kpt2d_datasets_info['train_data'][key] for key in train_2d_datasets]
train_2d_data_list = [item for sublist in train_2d_data_list for item in sublist]
train_2d_data_list = [os.path.join(data_root, item) for item in train_2d_data_list]

pub_train_data_list, _ = get_quest3_anno_paths(data_root)
ume_data_list = []
for hand in ['left', 'right']:
    ume_data_list += kpt3d_ume['separate_hand']['training'][hand]
ume_data_list = [os.path.join(data_root, item) for item in ume_data_list]
pub_train_data_list += ume_data_list

# InterHand
interhand_base = 'interhand2.6m/InterHand2.6M_5fps_batch1'
interhand_lmdb_dir = f'{interhand_base}/lmdb'
interhand_ann_lmdb = f'{interhand_lmdb_dir}/interhand_annotations.lmdb'
interhand_img_lmdb_train = f'{interhand_lmdb_dir}/interhand_images_train.lmdb'
interhand_img_lmdb_val = f'{interhand_lmdb_dir}/interhand_images_val.lmdb'
interhand_img_lmdb_test = f'{interhand_lmdb_dir}/interhand_images_test.lmdb'

interhand_train_keys = dict(
    ann_key='train/InterHand2.6M_train_data.json',
    camera_key='train/InterHand2.6M_train_camera.json',
    joint_key='train/InterHand2.6M_train_joint_3d.json',
)
interhand_val_keys = dict(
    ann_key='val/InterHand2.6M_val_data.json',
    camera_key='val/InterHand2.6M_val_camera.json',
    joint_key='val/InterHand2.6M_val_joint_3d.json',
)

val_splits = [
    # 普通
    dict(date_list=['20230830'], glasses_list=['Flora301'], person_list=['0005']),
    # 握拳 pinch
    # dict(date_list=['20240816'], glasses_list=['Flora301'], person_list=['0033']),
    # 竖向握拳
    # dict(date_list=['20240926'], glasses_list=['Flora301'], person_list=['0033']),
]

val_data_list = []

for split in val_splits:
    split_list = []
    for data_date in split['date_list']:
        for glasses in split['glasses_list']:
            split_list += kpt3d_datasets_info['test_data'][data_date].get(glasses, [])

    split_list = [os.path.join(data_root, item) for item in split_list]
    split_list = [p for p in split_list if p.split('__')[-3] in split['person_list']]
    val_data_list += split_list

val_data_list = sorted(set(val_data_list))


val_2d_dataset_name_list = [
    'flora_static_finegrain', 'flora_dynamic', 'flora_black',
    'flora_decoration', 'ella', 'near_two_hands', 'dark_light',
    'wrist_occlusion', 'tattoo', 'bad_bg', 'black_hand'
]
val_2d_data_list = dict()
for data_name in val_2d_dataset_name_list:
    val_2d_data_list[data_name] = kpt2d_datasets_info['test_data'][data_name]
    val_2d_data_list[data_name] = [os.path.join(data_root, item) for item in val_2d_data_list[data_name]]

# train dataloader
train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='MultiSourceSampler',
        source_ratio=[0.10, 0.50, 0.40, 0.00],
        batch_size=128),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/nreal_hand.py'),
        datasets=[
            dict(
                type=dataset_type,
                sample_interval=10,
                filter_kpt_exceed=True,
                data_ratio=1 / 30.0,
                data_file_list=pub_train_data_list,
                data_mode=data_mode,
                pipeline=train_pipeline,
                dataset_weight_list=None,
                data_root=data_root,
                flip_left_to_right=True,
                point_type='2.5D',
                round_num=2,
                epochs_per_round=5),

            dict(
                type=dataset_type,
                sample_interval=3,
                filter_kpt_exceed=True,
                serialize_data=True,
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
                sample_interval=1,
                serialize_data=True,
                data_mode=data_mode,
                pipeline=train_2d_pipeline,
                flip_left_to_right=True,
                data_root=data_root),

            dict(
                type='InterHandSingle3DDataset',
                data_root=data_root,
                data_mode=data_mode,
                pipeline=train_pipeline,
                test_mode=False,
                serialize_data=True,

                point_type='2.5D',
                data_ratio=1.0,

                ann_lmdb_dir=interhand_ann_lmdb,
                img_lmdb_dir=interhand_img_lmdb_train,
                split='train',

                sample_interval=3,

                min_valid_kpts=12,
                bbox_padding=1.40,
                filter_kpt_exceed=True,

                include_interacting=True,
                flip_left_to_right=True,

                z_min=1e-4,
                joint_cache='auto',
                release_raw_cache=True,

                **interhand_train_keys,
            )
        ]),
)

# val / test dataloader
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

val_dataset_interhand = dict(
    type='InterHandSingle3DDataset',
    data_root=data_root,
    data_mode=data_mode,
    pipeline=val_pipeline,
    test_mode=True,
    serialize_data=True,

    point_type='2.5D',

    ann_lmdb_dir=interhand_ann_lmdb,
    img_lmdb_dir=interhand_img_lmdb_val,
    split='val',

    sample_interval=5,
    min_valid_kpts=12,
    bbox_padding=1.40,
    filter_kpt_exceed=False,

    include_interacting=True,
    flip_left_to_right=True,

    z_min=1e-4,
    joint_cache='auto',
    release_raw_cache=True,

    **interhand_val_keys,
)

val_dataset = val_dataset_interhand if val_source == 'interhand' else (
    val_2d_dataset if test_type == '2d' else val_3d_dataset
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=val_dataset)

test_dataloader = val_dataloader

# evaluators
val_evaluator = [dict(type='EPE'), dict(type='NrealKeypointAP', with_tag=True)]
if test_type == '3d':
    val_evaluator += [dict(type='MPJPEV2', mode=['mpjpe', 'p-mpjpe'], result_dir='.')]
test_evaluator = val_evaluator

