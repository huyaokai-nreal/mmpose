data_root = '/data/AI_DATA_WX'
# data_root = '/data/AI_DATA_LOCAL'
test_type = '3d'
camera_layout = 'nimble'
data_mode = 'topdown'

codec = dict(
    type='RegressionLabel',
    input_size=(128, 128, 128),
    with_depth=True,
    depth_bound=0.4)

calib_pipeline = [
    dict(type='KeypointTo25DLabel', norm_depth=False),
    dict(type='GetBBoxCenterScale', padding=1.0),
    # dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    dict(type='TopdownPCL', input_size=codec['input_size'][:2]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='KeypointTo25DLabel', norm_depth=True),
    dict(type='GetBBoxCenterScale', padding=1.0),
    # dict(type='TopdownAffine', input_size=codec['input_size'][:2]),
    dict(type='TopdownPCL', input_size=codec['input_size'][:2]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

quant_calib_3d_dataset = dict(
    type='PairHand3DDatasetSeq',
    data_file_list=[],
    data_mode=data_mode,
    seq_len=4,
    serialize_data=True,
    test_mode=True,
    pipeline=calib_pipeline,
    flip_left_to_right=True,
    point_type='2.5D' if camera_layout in  ['monocular', 'nimble'] else '3D',
    data_root=data_root,
    choice_one=True)

quant_test_3d_dataset = dict(
    type='PairHand3DDataset',
    data_file_list=[],
    data_mode=data_mode,
    serialize_data=True,
    test_mode=True,
    pipeline=test_pipeline,
    flip_left_to_right=True,
    point_type='2.5D' if camera_layout in  ['monocular', 'nimble'] else '3D',
    data_root=data_root)


# evaluators
val_evaluator = [dict(type='EPE'), dict(type='NrealKeypointAP', with_tag=True)]
if test_type == '3d':
    val_evaluator += [
        dict(
            type='MPJPEV2',
            mode='mpjpe',
            scale_metric=False,
            with_tag=True,
            rearrange_result=True,
        ),
        dict(type='MPJPEV2', mode='p-mpjpe', prefix='1'),
    ]
test_evaluator = val_evaluator