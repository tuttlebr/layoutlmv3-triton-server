import json
import logging

import numpy as np
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

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

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "true_predictions"
        )

        output1_config = pb_utils.get_output_config_by_name(model_config, "true_boxes")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        self.id2label = {
            0: "O",
            1: "B-HEADER",
            2: "I-HEADER",
            3: "B-QUESTION",
            4: "I-QUESTION",
            5: "B-ANSWER",
            6: "I-ANSWER",
        }

    def unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]

    def iob_to_label(self, label):
        label = label[2:]
        if not label:
            return "other"
        return label

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

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get inputs
            in_0 = pb_utils.get_input_tensor_by_name(request, "logits")
            in_1 = pb_utils.get_input_tensor_by_name(request, "bbox")
            in_2 = pb_utils.get_input_tensor_by_name(request, "offset_mapping")
            in_3 = pb_utils.get_input_tensor_by_name(request, "raw_image_shape")

            width, height = in_3.as_numpy().squeeze()
            token_boxes = in_1.as_numpy().squeeze().tolist()
            is_subword = np.array(in_2.as_numpy().squeeze().tolist())[:, 0] != 0

            predictions = in_0.as_numpy().argmax(-1).squeeze().tolist()

            true_predictions = [
                self.id2label[pred]
                for idx, pred in enumerate(predictions)
                if not is_subword[idx]
            ]

            true_boxes = [
                self.unnormalize_box(box, width, height)
                for idx, box in enumerate(token_boxes)
                if not is_subword[idx]
            ]

            out_tensor_0 = pb_utils.Tensor(
                "true_predictions", np.array(true_predictions).astype(output0_dtype)
            )

            out_tensor_1 = pb_utils.Tensor(
                "true_boxes", np.array(true_boxes).astype(output1_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
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
        print("Postprocess cleaning up...")
