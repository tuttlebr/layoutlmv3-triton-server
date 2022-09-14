import argparse
import json
import logging
import os
import sys

import numpy as np
import tritongrpcclient

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    """
    return np.fromfile(img_path, dtype="uint8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="layoutlmv3_0_ensemble",
        help="Model name",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    inputs = []
    outputs = []
    input_name = "raw_image_array"
    output_names = ["true_predictions", "true_boxes"]
    image_data = load_image(args.image)
    image_data = np.expand_dims(image_data, axis=0)
    inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))

    for i in output_names:
        outputs.append(tritongrpcclient.InferRequestedOutput(i))

    inputs[0].set_data_from_numpy(image_data)
    results = triton_client.infer(
        model_name=args.model_name, inputs=inputs, outputs=outputs
    )

    for output_name in output_names:
        print(results.as_numpy(output_name))
    # maxs = np.argmax(output0_data, axis=1)
    # print(maxs)
    # print("Result is class: {}".format(labels_dict[maxs[0]]))
