from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.objects import PredictionValue
import os
import ddddocr
import requests

base_dir = "/opt/label-studio/files"

det = ddddocr.DdddOcr(
    show_ad=False,
    import_onnx_path="/home/wangjie/PycharmProjects/dddd_trainer/projects/piaoxingqiu/models/efficientnet_v2_s_3_v2009_32_d05_lr01_pretrained_merge/piaoxingqiu_0.984375_26_1500_2024-04-10-19-28-57.onnx",
    charsets_path="/home/wangjie/PycharmProjects/dddd_trainer/projects/piaoxingqiu/models/efficientnet_v2_s_3_v2009_32_d05_lr01_pretrained_merge/charsets.json",
)


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model"""

    def setup(self):
        """Configure any paramaters of your model here"""
        self.set("model_version", "0.0.1")

    # example for simple classification
    # return [{
    #     "model_version": self.get("model_version"),
    #     "score": 0.12,
    #     "result": [{
    #         "id": "vgzE336-a8",
    #         "from_name": "sentiment",
    #         "to_name": "text",
    #         "type": "choices",
    #         "value": {
    #             "choices": [ "Negative" ]
    #         }
    #     }]
    # }]
    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(
            f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}"""
        )

        result = []

        for task in tasks:
            url = task["data"]["captioning"].split("d=")[-1]

            pred = det.classification(requests.get(url).content)
            result.append(
                {
                    "value": {"text": [pred]},
                    "type": "textarea",
                    "from_name": "caption",
                    "to_name": "image",
                }
            )

        return ModelResponse(
            predictions=[
                PredictionValue(
                    model_version=self.get("model_version"), score=0.5, result=result
                )
            ],
        )

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")
