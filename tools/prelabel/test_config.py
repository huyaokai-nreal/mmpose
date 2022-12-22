# flake8: noqa

pipeline_cfg = dict(
    type='PipelineBase',
    nodes=[
        dict(
            type='TDRegressKeypointModel',
            model_path=
            '/home/zx_li/workspace/mmpose/work_dirs/model_zoo/restiny_pre_ipr_rle_deploy.onnx',
            input_shape=(128, 128)),
        dict(type='PoseVisualizer')
    ])
