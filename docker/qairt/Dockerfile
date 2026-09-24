FROM --platform=linux/amd64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libpython3.12 \
        libc++1 \
        libatomic1 \
        wget \
        unzip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python environment for QAIRT host-side tools
# ------------------------------------------------------------

RUN python3 -m venv /opt/qairt-venv

ENV PATH="/opt/qairt-venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "numpy<2.0" \
        "onnx==1.20.0" \
        "onnxruntime<1.24" \
        onnxsim \
        pandas \
        pyyaml \
        setuptools

# ------------------------------------------------------------
# Android NDK r26c
#
# QAIRT 2.50 documentation specifies r26c for Android targets.
# This is the Linux x86_64 NDK because QAIRT host tools run
# inside this Linux/amd64 container.
# ------------------------------------------------------------

RUN cd /opt && \
    wget -q https://dl.google.com/android/repository/android-ndk-r26c-linux.zip && \
    unzip -q android-ndk-r26c-linux.zip && \
    rm android-ndk-r26c-linux.zip

ENV ANDROID_NDK_ROOT="/opt/android-ndk-r26c"
ENV PATH="${ANDROID_NDK_ROOT}:${PATH}"

WORKDIR /workspace

CMD ["/bin/bash"]