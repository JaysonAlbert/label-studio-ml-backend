import logging
import requests
import time


def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time of {func.__name__}: {end - start} seconds")
        return result

    return wrapper


from typing import List, Dict

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_image_size,
    get_single_tag_keys,
)

logger = logging.getLogger(__name__)


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

    @time_function
    def predict(self, tasks: List[Dict], **kwargs):
        predictions = []
        for task in tasks:
            prediction = self.predict_one_task(task)
            predictions.append(prediction)
        return predictions

    @time_function
    def predict_one_task(self, task: Dict):
        url = task["data"]["image"].split("d=")[-1]
        results = []
        all_scores = []
        img_width, img_height = get_image_size(url)

        predicts = self.get_predict(url)
        for predict in predicts:
            bbox = list(predict["bbox"])
            output_label = predict["predict"]
            score = predict["prob"]
            if not bbox:
                continue

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

    def get_predict(self, image_path):
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
        return response.json()["body"]
