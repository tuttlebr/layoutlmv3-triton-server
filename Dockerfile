# syntax=docker/dockerfile:1

ARG FROM_BASE_IMAGE
FROM ${FROM_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN curl -s https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add - && \
    echo 'deb https://deb.nodesource.com/node_18.x focal main' \
    > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y \
    nodejs \
    asciidoc \
    autoconf \
    automake \
    clinfo \
    g++ \
    git \
    libleptonica-dev \
    libpng-dev \
    libjpeg8-dev \
    libtiff5-dev \
    libtool \
    libicu-dev \
    libpango1.0-dev \
    libcairo2-dev \
    nvidia-cuda-toolkit \
    xsltproc \
    pkg-config \
    zlib1g-dev

COPY requirements.txt .
RUN pip install -q -r requirements.txt

COPY --link=false --from=huggingspace:tesseract /workspace/tesseract /workspace/tesseract
COPY --link=false --from=huggingspace:tesseract /workspace/tessdata /workspace/tessdata