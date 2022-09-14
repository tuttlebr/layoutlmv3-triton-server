# LayoutLMv3

## An Ensemble Model for NVIDIA Triton Server

## Overview

The LayoutLMv3 model was proposed in [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387) by Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. LayoutLMv3 simplifies [LayoutLMv2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2) by using patch embeddings (as in [ViT](https://huggingface.co/docs/transformers/model_doc/vit)) instead of leveraging a CNN backbone, and pre-trains the model on 3 objectives: masked language modeling (MLM), masked image modeling (MIM) and word-patch alignment (WPA).

The abstract from the paper is the following:

_Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis._

<p align="left">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png" alt="layoutlmv3_architecture" width="800"/>
</p>

## Getting Started

<p align="left">
    <img style="center;" src="https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/enterprise-ai-platform/nvidia-diagram-accelerated-computing-egx-overview.svg" alt="nvidia-diagram-accelerated-computing-egx-overview" width="800"/>
</p>

This model uses NVIDIA's enterprise containers available on NGC with a valid API_KEY. These containers provide best-in-class development tools and frameworks for the AI practitioner and reliable management and orchestration for the IT professional to ensure performance, high availability, and security. For more information on how to obtain access to these containers, check the [NVIDIA website](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

0. Login to nvcr.io with your enterprise API key.
   ```sh
   docker login nvcr.io
   ```
1. Build the images. This will also compile Tesseract for OCR to run on NVIDIA GPUs.
   ```sh
   bash build.sh
   ```
2. Launch the Jupyter Lab and follow the workflow within `LayoutLMv3-notebook.ipynb`
   ```sh
   docker compose up layoutlmv3-pytorch
   ```
3. Launch Triton Inference Server with your Models from the prior step.
   ```sh
   docker compose up layoutlmv3-triton-server
   ```
4. In another terminal, send a sample inference image.
   ```sh
   docker compose up layoutlmv3-triton-client
   ```

## Optional

1. Run NVIDIA Model Analyzer. [NVIDIA Triton Model Analyzer](https://developer.nvidia.com/blog/identifying-the-best-ai-model-serving-configurations-at-scale-with-triton-model-analyzer/) is a versatile CLI tool that helps with a better understanding of the compute and memory requirements of models served through NVIDIA Triton Inference Server. This enables you to characterize the tradeoffs between different configurations and choose the best one for your use case.

   ```sh
   docker compose up layoutlmv3-model-analyzer
   ```

2. Run NVIDIA Performance Analyzer. The [perf_analyzer](https://github.com/triton-inference-server/server/blob/3589c2c72d249392809fe9ae3de8057fe0437135/docs/user_guide/perf_analyzer.md) application generates inference requests to your model and measures the throughput and latency of those requests. To get representative results, perf_analyzer measures the throughput and latency over a time window, and then repeats the measurements until it gets stable values. By default perf_analyzer uses average latency to determine stability but you can use the --percentile flag to stabilize results based on that confidence level. For example, if --percentile=95 is used the results will be stabilized using the 95-th percentile request latency.
   ```sh
   docker compose up layoutlmv3-triton-client-query
   ```
