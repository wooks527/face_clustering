"""RetinaFace class.

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from __future__ import print_function
from typing import Dict, List, Tuple

import torch
import cv2
import torch.backends.cudnn as cudnn
import numpy as np

from Pytorch_Retinaface.data import cfg_mnet, cfg_re50
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from Pytorch_Retinaface.models.retinaface import RetinaFace as OriginalRetinaFace
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm


class RetinaFace:
    """RetinaFace Model"""

    def __init__(
        self,
        network: str = "resnet50",
        trained_model: str = "./Pytorch_Retinaface/weights/Resnet50_Final.pth",
        cpu: bool = False,
    ) -> None:
        """Initializae instances.

        Args:
            network: backbone network
            trained_model: pre-trained model file path
            cpu: use cpu or not
        """
        torch.set_grad_enabled(False)
        self.cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50

        # net and model
        self.net = OriginalRetinaFace(cfg=self.cfg, phase="test")
        self.net = self.load_model(self.net, trained_model, cpu)
        self.net.eval()
        print("Finished loading model!")
        cudnn.benchmark = True
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = self.net.to(self.device)

    def check_keys(self, model: OriginalRetinaFace, pretrained_state_dict) -> bool:
        """Check keys.

        Args:
            model: face detection model
            pretrained_state_dict: dictionary of model parameters

        Returns:
            boolean value whether keys are valid or not
        """
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print("Missing keys:{}".format(len(missing_keys)))
        print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        print("Used keys:{}".format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
        return True

    def remove_prefix(self, state_dict: Dict, prefix: str) -> Dict:
        """Old style model is stored with all names of parameters
           sharing common prefix 'module.'

        Args:
            state_dict: model parameters
            prefix: removed prefix of parameters

        Returns:
            dictionary of parameters removed with prefix 'module.'
        """
        print("remove prefix '{}'".format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x  # noqa
        return {f(key): value for key, value in state_dict.items()}

    def load_model(
        self, model: OriginalRetinaFace, pretrained_path: str, load_to_cpu: bool
    ) -> OriginalRetinaFace:
        """Load model.

        Args:
            model: Face detection model
            pretrained_path: pretrained model weight file path
            load_to_cpu: use cpu or not

        Returns:
            model: Face detection model with pretrained weights
        """
        print("Loading pretrained model from {}".format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
            )
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, "module.")
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def detect(
        self,
        img_raw: np.ndarray = None,
        confidence_threshold: float = 0.02,
        top_k: int = 5000,
        nms_threshold: float = 0.4,
        keep_top_k: int = 750,
        save_image: bool = False,
        vis_thres: float = 0.6,
    ) -> List[Tuple]:
        """Detect face bounding boxes with facial landmarks.

        Args:
            img_raw: raw image
            confidence_threshold: confidence threshold for confidence score
            top_k: the number of bboxes which will be used before NMS
            nms_threshold: NMS threshold
            keep_top_k: the number of bboxes which will be used finally
            save_image: save image or not
            vis_thres: final confidence threshold

        Returns:
            dets_filtered: final bounding boxes
        """
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        resize = 1

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, nms_threshold, force_cpu=cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets_with_landms = np.concatenate((dets, landms), axis=1)

        # show image
        if save_image:
            for b in dets_with_landms:
                if b[4] < vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(
                    img_raw,
                    text,
                    (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                )

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # save image
            name = "test.jpg"
            cv2.imwrite(name, img_raw)

        dets_filtered = []
        for left, top, right, bottom, conf in dets:
            if conf >= vis_thres:
                dets_filtered.append(tuple(map(int, (top, right, bottom, left))))
        return dets_filtered
