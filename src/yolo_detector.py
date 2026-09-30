"""
yolo_detector.py
----------------
Number-plate localization using a pretrained YOLOv5 model
(fine-tuned specifically to detect license plates).

Model weights are downloaded automatically the first time this runs
(from the Hugging Face Hub, via the `yolov5` package), then cached
locally by that library - no manual download step needed.

If you later train your own model on your own region's plates, just
point MODEL_NAME (in config.py) at your local `best.pt` file path
instead of the Hugging Face model id - the rest of the code is unchanged.
"""
from typing import List, Tuple

import numpy as np

BoxWithConf = Tuple[int, int, int, int, float]  # (x, y, w, h, confidence)


class YoloPlateDetector:
    def __init__(
        self,
        model_name: str = "keremberke/yolov5s-license-plate",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_det: int = 10,
    ):
        # Imported lazily so this module can be imported (e.g. by tests using a
        # fake detector) without requiring the (large) yolov5/torch stack installed.
        import torch
        import yolov5

        # PyTorch 2.6+ made `torch.load()` default to `weights_only=True`, which
        # refuses to unpickle the custom model classes used inside older YOLO
        # checkpoints (this yolov5 package predates that change). We trust this
        # specific, well-known public checkpoint, so we relax that default here.
        _original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load

        print(f"Loading YOLO plate-detection model '{model_name}' "
              f"(first run downloads it - this may take a minute)...")
        self.model = yolov5.load(model_name)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.max_det = max_det

    def detect(self, frame_bgr: np.ndarray) -> List[BoxWithConf]:
        """
        Run YOLO on a single BGR frame (as produced by cv2.VideoCapture / cv2.imread).
        Returns a list of (x, y, w, h, confidence) boxes.
        """
        # YOLO expects RGB ordering.
        rgb = frame_bgr[:, :, ::-1]
        results = self.model(rgb, size=640)

        preds = results.pred[0].cpu().numpy()  # rows: x1, y1, x2, y2, conf, cls
        boxes: List[BoxWithConf] = []
        for x1, y1, x2, y2, conf, _cls in preds:
            w, h = int(x2 - x1), int(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            boxes.append((int(x1), int(y1), w, h, float(conf)))

        return boxes
