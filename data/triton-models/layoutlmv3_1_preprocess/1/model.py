import io
import json
import logging
import os

import numpy as np
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from PIL import Image
from pytesseract import apply_tesseract
from transformers import LayoutLMv3Processor

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        self.preprocessor = LayoutLMv3Processor.from_pretrained(
            "{}/preprocessing_config".format(
                os.path.realpath(os.path.dirname(__file__))
            ),
            apply_ocr=False,
        )

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "input_ids"
        )

        output1_config = pb_utils.get_output_config_by_name(
            model_config, "attention_mask"
        )

        output2_config = pb_utils.get_output_config_by_name(
            model_config, "offset_mapping"
        )

        output3_config = pb_utils.get_output_config_by_name(
            model_config, "bbox"
        )

        output4_config = pb_utils.get_output_config_by_name(
            model_config, "pixel_values"
        )

        output5_config = pb_utils.get_output_config_by_name(
            model_config, "raw_image_shape"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output2_config["data_type"]
        )

        self.output3_dtype = pb_utils.triton_string_to_numpy(
            output3_config["data_type"]
        )

        self.output4_dtype = pb_utils.triton_string_to_numpy(
            output4_config["data_type"]
        )

        self.output5_dtype = pb_utils.triton_string_to_numpy(
            output5_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype
        output3_dtype = self.output3_dtype
        output4_dtype = self.output4_dtype
        output5_dtype = self.output5_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get input
            in_0 = pb_utils.get_input_tensor_by_name(
                request, "raw_image_array"
            )
            img = in_0.as_numpy()
            image = Image.open(io.BytesIO(img.tobytes())).convert("RGB")
            text, boxes = apply_tesseract(
                image, lang="eng", tesseract_config="--oem 1"
            )
            h, w = image.size

            encoding = self.preprocessor(
                image, text=text, boxes=boxes, return_offsets_mapping=True
            )

            ipnut_ids = np.expand_dims(
                np.array(encoding["input_ids"]).astype(output0_dtype), axis=0
            )
            out_tensor_0 = pb_utils.Tensor("input_ids", ipnut_ids)
            # logger.info(ipnut_ids.shape)

            attention_mask = np.expand_dims(
                np.array(encoding["attention_mask"]).astype(output1_dtype),
                axis=0,
            )
            out_tensor_1 = pb_utils.Tensor("attention_mask", attention_mask)
            # logger.info(attention_mask.shape)

            offset_mapping = np.expand_dims(
                np.array(encoding["offset_mapping"]).astype(output2_dtype),
                axis=0,
            )
            out_tensor_2 = pb_utils.Tensor("offset_mapping", offset_mapping)
            # logger.info(offset_mapping.shape)

            bbox = np.expand_dims(
                np.array(encoding["bbox"]).astype(output3_dtype), axis=0
            )
            out_tensor_3 = pb_utils.Tensor("bbox", bbox)
            # logger.info(bbox.shape)

            pixel_values = np.array(encoding["pixel_values"]).astype(
                output4_dtype
            )
            out_tensor_4 = pb_utils.Tensor("pixel_values", pixel_values)
            # logger.info(pixel_values.shape)

            raw_image_shape = np.expand_dims(
                np.array((h, w)).astype(output5_dtype), axis=0
            )
            out_tensor_5 = pb_utils.Tensor("raw_image_shape", raw_image_shape)
            # logger.info(raw_image_shape.shape)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    out_tensor_0,
                    out_tensor_1,
                    out_tensor_2,
                    out_tensor_3,
                    out_tensor_4,
                    out_tensor_5,
                ]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Preprocessing cleaning up...")
