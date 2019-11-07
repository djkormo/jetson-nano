FROM nvcr.io/nvidia/l4t-base:r32.2
WORKDIR /home
RUN apt-get update && apt-get install -y --fix-missing make g++
RUN apt-get update && apt-get install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools
RUN apt-get update && apt-get install -y python3-h5py
RUN pip3 install --pre --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
RUN pip3 install -U numpy
COPY test.py test.py
CMD [ "bash" ]