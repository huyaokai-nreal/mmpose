_base_ = [
    './td-hand_res26_25d_4x128-50e_test-128x128.py'  # noqa: E501
]

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=60, val_interval=5)

# method details
model = dict(
    _delete_=True,
    type='DWPoseDistiller',
    two_dis=second_dis,
    teacher_pretrained=
    'work_dirs/distill/dwpose-td-hand_res26_25d_4x128-50e_test-128x128.py/epoch_2.pth',  # noqa: E501
    teacher_cfg=
    'configs/product/ykhu/distill/td-hand_res26_25d_4x128-50e_test-128x128.py',  # noqa: E501
    student_cfg=
    'configs/product/ykhu/distill/td-hand_res26_25d_4x128-50e_test-128x128.py',  # noqa: E501
    distill_cfg=[
        dict(methods=[
            dict(
                type='KDLoss',
                name='loss_logit',
                use_this=logit,
                weight=1,
            )
        ]),
    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor', mean=[0.449 * 255], std=[0.226 * 255]),
    train_cfg=train_cfg,
)

optim_wrapper = dict(clip_grad=dict(max_norm=1., norm_type=2))
