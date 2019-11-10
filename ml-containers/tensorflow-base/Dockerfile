FROM nvcr.io/nvidia/l4t-base:r32.2
WORKDIR /home
RUN apt-get update && apt-get install -y --fix-missing make g++ libblas-dev liblapack-dev libatlas-base-dev gfortran
RUN apt-get update && apt-get install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools libfreetype6-dev
RUN apt-get update && apt-get install -y python3-h5py
RUN pip3 install --pre --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
RUN pip3 install -U --pre --no-cache-dir numpy 
RUN pip3 install --no-cache-dir folium seaborn  keras
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
COPY test.py test.py
CMD [ "bash" ]