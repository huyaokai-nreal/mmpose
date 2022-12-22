from .registry import COMPUTE_NODES
import onnxruntime
import numpy as np
from mmengine.dataset import Compose
from mmpose.datasets.transforms.common_transforms import GetBBoxCenterScale
from mmpose.datasets.transforms.topdown_transforms import TopdownAffine
from mmpose.structures.bbox import bbox_cs2xywh


@COMPUTE_NODES.register_module()
class TDRegressKeypointModel:

    def __init__(self,
                 model_path,
                 input_shape,
                 padding=1.0,
                 mean=0.449,
                 std=0.226) -> None:
        self.input_shape = input_shape
        self.mean = mean
        self.std = std
        self.device_id = 0
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': self.device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        self.model = onnxruntime.InferenceSession(
            model_path, providers=providers)
        self.transform = Compose([
            GetBBoxCenterScale(padding=padding),
            TopdownAffine(input_size=input_shape)
        ])

    def process(self, data):
        objects = data['objects']
        for obj in objects:
            bbox = obj['bbox']
            img = data['img']
            data_info = dict(img=img, bbox=bbox)
            data_info = self.transform(data_info)
            crop_img = data_info['img']
            crop_img = ((crop_img / 255.0) - self.mean) / self.std
            crop_img = crop_img.astype(np.float32)[np.newaxis,
                                                   np.newaxis, :, :]
            inputs = {self.model.get_inputs()[0].name: crop_img}
            outputs = np.concatenate(self.model.run(None, inputs), axis=-1)
            crop_bbox = bbox_cs2xywh(data_info['bbox_center'],
                                     data_info['bbox_scale'])
            batch_keypoints = outputs * crop_bbox[:, 2:] + crop_bbox[:, :2]
            obj['keypoints2d'] = batch_keypoints
        return data
