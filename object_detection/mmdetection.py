import os
import logging
import boto3
import io
import json
import ddddocr
import requests

from typing import List, Dict

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_image_size,
    get_single_tag_keys,
    DATA_UNDEFINED_NAME,
)
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
det = ddddocr.DdddOcr(det=True)


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""
    
    def __init__(
        self,
        config_file=None,
        checkpoint_file=None,
        image_dir=None,
        labels_file=None,
        score_threshold=0.5,
        device="cpu",
        **kwargs,
    ):
        super(MMDetection, self).__init__(**kwargs)
        (
            self.from_name,
            self.to_name,
            self.value,
            self.labels_in_config,
        ) = get_single_tag_keys(self.parsed_label_config, "RectangleLabels", "Image")

        
    

    def predict(self, tasks: List[Dict], **kwargs):
        if len(tasks) > 1:
            print(
                "==> Only the first task will be processed to avoid ML backend overloading"
            )
            tasks = [tasks[0]]

        predictions = []
        for task in tasks:
            prediction = self.predict_one_task(task)
            predictions.append(prediction)
        return predictions

    def predict_one_task(self, task: Dict):
        print(f">>> TASK: {task}")
        url = task["data"]["image"].split("d=")[-1]
        bboxes = det.detection(requests.get(url).content)
        results = []
        all_scores = []
        img_width, img_height = get_image_size(url)

        for bbox in bboxes:
            bbox = list(bbox)
            if not bbox:
                continue

            output_label, score = self.get_label(url)

            x, y, xmax, ymax = bbox[:4]
            results.append(
                {
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [output_label],
                        "x": float(x) / img_width * 100,
                        "y": float(y) / img_height * 100,
                        "width": (float(xmax) - float(x)) / img_width * 100,
                        "height": (float(ymax) - float(y)) / img_height * 100,
                    },
                    "score": score,
                }
            )
            all_scores.append(score)
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        print(f">>> RESULTS: {results}")
        return {"result": results, "score": avg_score, "model_version": "mmdet"}

    def get_label(self, image_path):
        import requests
        import json

        url = "http://localhost:8001/predict"

        payload = json.dumps(
            {
                "front": image_path,
                "type": 1,
            }
        )
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)

        predict = response.json()["body"]["predict"]
        prob = response.json()["body"]["prob"]
        return predict, prob
