import copy
import torch
import numpy as np
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.structures import BitMasks, Instances


from scipy.ndimage import laplace
import torch.nn.functional as F


class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and maps it into a format used by semantic segmentation.
    Now it also supports generating edge maps for edge-aware supervision.
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.augmentation = T.AugmentationList(utils.build_augmentation(cfg, is_train))
        self.image_format = cfg.INPUT.FORMAT
        self.semantic_mask_format = cfg.INPUT.MASK_FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict["sem_seg_file_name"], format=self.semantic_mask_format)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentation(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg


        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt

            # Prepare per-category binary masks
            sem_seg_np = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_np)
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_np == class_id)

            if len(masks) == 0:
                instances.gt_masks = torch.zeros((0, sem_seg_np.shape[-2], sem_seg_np.shape[-1]))
            else:
                tensor_masks = [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]
                tensor_masks = [m.squeeze(-1) if m.ndim == 3 else m for m in tensor_masks]
                masks = BitMasks(torch.stack(tensor_masks))
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

            # ✅ 添加边缘图生成（用于 edge loss）
            edge_map = laplace(sem_seg_np.astype(np.float32))
            edge_map = np.abs(edge_map) > 0  # 转成二值
            edge_tensor = torch.from_numpy(edge_map.astype(np.uint8)).float()

            # 确保尺寸匹配
            if edge_tensor.shape != sem_seg_gt.shape:
                edge_tensor = F.interpolate(edge_tensor[None, None], size=sem_seg_gt.shape, mode="nearest").squeeze()

            dataset_dict["edge"] = edge_tensor
        return dataset_dict
