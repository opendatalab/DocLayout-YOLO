from doclayout_yolo.models.yolo.detect import DetectionValidator
from doclayout_yolo.utils import ops
import torch

import pdb

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds, conf=None):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        preds = preds.transpose(-1, -2)
        boxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, self.nc)
        bboxes = ops.xywh2xyxy(boxes)
        # return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        
        preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        if preds.shape[-1] == 6 and conf is not None:  # end-to-end model (BNC, i.e. 1,300,6)
            preds = [pred[pred[:, 4] > conf] for pred in preds]
        return preds