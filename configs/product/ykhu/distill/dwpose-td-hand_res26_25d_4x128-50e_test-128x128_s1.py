_base_ = [
    './td-hand_res26_25d_4x128-50e_test-128x128.py'  # noqa: E501
]

# model settings
find_unused_parameters = False

# config settings
fea = True
logit = True

# method details
model = dict(
    _delete_=True,
    type='DWPoseDistiller3D',
    teacher_pretrained=
    'work_dirs/rtm/large-model/td-hand_res101_25d_4x128-50e_test-128x128.py/epoch_50.pth',  # noqa: E501
    teacher_cfg=
    'configs/product/ykhu/distill/td-hand_res26_25d_4x128-50e_test-128x128.py',  # noqa: E501
    student_cfg=
    'configs/product/ykhu/distill/td-hand_res26_25d_4x128-50e_test-128x128.py',  # noqa: E501
    distill_cfg=[
        dict(methods=[
            dict(
                type='FeaLoss',
                name='loss_fea',
                use_this=fea,
                student_channels=160,
                teacher_channels=160,
                alpha_fea=0.00007,
            )
        ]),
        dict(methods=[
            dict(
                type='KDLoss3D',
                name='loss_logit',
                use_this=logit,
                weight=0.1,
            )
        ]),
    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
)
optim_wrapper = dict(clip_grad=dict(max_norm=1., norm_type=2))
