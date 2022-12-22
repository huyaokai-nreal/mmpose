from .registry import COMPUTE_NODES
import numpy as np
import cv2
from mmpose.datasets.datasets.hand.nreal_hand import HANDDataset
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.visualization.local_visualizer import PoseLocalVisualizer


@COMPUTE_NODES.register_module()
class PoseVisualizer():

    def __init__(self) -> None:
        self.pose_visualizer = PoseLocalVisualizer(radius=1)
        dataset_meta = HANDDataset([]).metainfo
        self.pose_visualizer.set_dataset_meta(dataset_meta)

    def process(self, data):
        gt_instances = InstanceData()
        keypoints = list()
        objects = data['objects']
        for obj in objects:
            print(obj['keypoints2d'].shape)
            keypoints.append(obj['keypoints2d'])
        keypoints = np.concatenate(keypoints, axis=0)
        print(keypoints.shape)
        gt_instances.keypoints = keypoints
        gt_pose_data_sample = PoseDataSample()
        gt_pose_data_sample.gt_instances = gt_instances
        image = data['img']
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.pose_visualizer.add_datasample(
            'image', image, gt_pose_data_sample, draw_pred=False)
        image = self.pose_visualizer.get_image()
        cv2.imwrite(f'work_dirs/test_{data["image_id"]}.png', image)
