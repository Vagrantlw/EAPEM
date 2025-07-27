import tempfile
import cv2
import cog
import os
import torch

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from pem.config import add_maskformer2_config


class Predictor(cog.BasePredictor):
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("./output/discussion/model/Mask2Former/config.yaml")
        cfg.MODEL.WEIGHTS = './output/discussion/model/Mask2Former/model_final.pth'
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("ade_indoor_sem_seg_val")

    def predict(self, image, outdir, raw_image=None):
        im = cv2.imread(str(image))
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        if raw_image is None:
            outputs = self.predictor(im)
            # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
            rep = 1-outputs["sem_seg"].sum(0)
            rep = rep.unsqueeze(0)
            new_p = torch.cat((outputs["sem_seg"], rep), dim=0)
            label = new_p.argmax(0)-1
            label = torch.where(label == -1, 255, label)
            result = v.draw_sem_seg(label.to("cpu")).get_image()[:, :, ::-1]
            # result = np.concatenate((semantic_result, instance_result), axis=0)[:, :, ::-1] #semantic_result

        else:
            outputs = cv2.imread(str(raw_image.replace('.jpg', '.png')))
            result = v.draw_sem_seg(outputs[:, :, 0]).get_image()[:, :, ::-1]

        out_path = f"{outdir}{image.split('/')[-1]}"
        cv2.imwrite(str(out_path), result)
        return out_path


image_path = './datasets/ADE_indoor/images/validation/'
lable_path = './datasets/ADE_indoor/annotations_detectron2/validation/' #training, validation
outdir = './output/val_prediction/' # val_images,val_prediction，train_images
imagelist = [image_path + _ for _ in os.listdir(image_path)]
re = Predictor()
re.setup()
for image in imagelist:
    out_path = re.predict(image, outdir=outdir, raw_image=None)  # lable_path + image.split('/')[-1],None
