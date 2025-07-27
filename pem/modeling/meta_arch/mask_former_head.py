import logging
from typing import Dict, Tuple

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.pem_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.pem_pixel_decoder import build_pixel_decoder


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, norm="BN"):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, in_channels // 2,
            kernel_size=3, padding=1, norm=get_norm(norm, in_channels // 2)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)  # binary edge mask

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False
            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} has changed! "
                    "Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        edge_loss_weight: float = 1.0,
    ):
        """
        Args:
            input_shape: feature shape from backbone
            num_classes: number of semantic classes
            pixel_decoder: PEM_Pixel_Decoder instance
            loss_weight: for total loss
            ignore_value: ignored label id
            transformer_predictor: PEM transformer decoder
            transformer_in_feature: which feature to feed transformer
            edge_loss_weight: loss weight for edge branch
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.edge_loss_weight = edge_loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

        # Edge decoder
        self.edge_decoder = EdgeDecoder(in_channels=256)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "edge_loss_weight": cfg.MODEL.SEM_SEG_HEAD.get("EDGE_LOSS_WEIGHT", 1.0),
        }

    def forward(self, features, mask=None) -> Dict[str, torch.Tensor]:
        return self.layers(features, mask)

    def layers(self, features, mask=None) -> Dict[str, torch.Tensor]:
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        predictions = self.predictor(multi_scale_features, mask_features, mask)
        edge_map = self.edge_decoder(mask_features)

        predictions["pred_edges"] = edge_map  # key used in loss

        return predictions
