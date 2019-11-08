FROM nvcr.io/nvidia/l4t-base:r32.2
WORKDIR /home
RUN apt-get update && apt-get install -y --fix-missing python3-pip git
RUN wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl \
    && pip3 install numpy torch-1.3.0-cp36-cp36m-linux_aarch64.whl && rm torch-1.3.0-cp36-cp36m-linux_aarch64.whl
RUN apt-get update && apt install -y --fix-missing libjpeg-dev zlib1g-dev \
  && git clone -b v0.5.0 https://github.com/pytorch/vision torchvision --depth 1 \
  && cd torchvision 
#     && python3 setup.py install \
#     && cd .. && rm -rf torchvision
COPY test.py test.py
CMD [ "bash" ]

