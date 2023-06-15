from mmengine.model import BaseModel
import torch
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, ForwardResults, OptConfigType,
                                 Optional, OptSampleList)
from mmpose.registry import MODELS
from mmengine.structures import InstanceData


@MODELS.register_module()
class PoseAttr(BaseModel):

    def __init__(self,
                 backbone: ConfigType,
                 head: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(data_preprocessor=None, init_cfg=None)
        self.backbone = MODELS.build(backbone)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_feat(self, inputs):
        feat = self.backbone(inputs)
        return feat

    def _forward(self, inputs, data_samples):
        output = self.extract_feat(inputs)
        output = self.head(output)
        return output

    def loss(self, inputs, data_samples):
        feat = self.extract_feat(inputs)
        return self.head.loss(feat, data_samples)

    def predict(self, inputs, data_samples):
        feat = self.extract_feat(inputs)
        outputs = self.head.predict(feat, data_samples)
        for i, data_sample in enumerate(data_samples):
            pred_instances = InstanceData()
            pred_instances.set_field(to_numpy(outputs[i:i + 1]), 'attr')
            data_sample.pred_instances = pred_instances
        return data_samples

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[OptSampleList] = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: 'tensor', 'predict' and 'loss':

        - 'tensor': Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - 'predict': Forward and return the predictions, which are fully
        processed to a list of :obj:`PoseDataSample`.
        - 'loss': Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general
            data_samples (list[:obj:`PoseDataSample`], optional): The
                annotation of every sample. Defaults to ``None``
            mode (str): Set the forward mode and return value type. Defaults
                to ``'tensor'``

        Returns:
            The return type depends on ``mode``.

            - If ``mode='tensor'``, return a tensor or a tuple of tensors
            - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
                that contains the pose predictions
            - If ``mode='loss'``, return a dict of tensor(s) which is the loss
                function value
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')
