# https://dev.to/faizanbashir/building-python-data-science-container-usingdocker-3f8p
FROM nvcr.io/nvidia/l4t-base:r32.2
WORKDIR /home
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_PACKAGES="\
    cython \
    numpy \
    matplotlib \
    folium \
    seaborn \
    scipy \
    scikit-learn \
    pandas \
    nltk \
"

RUN apt-get update && apt-get install -y --fix-missing make g++
RUN apt-get update && apt-get install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools
RUN apt-get update && apt-get install -y python3-h5py

RUN apt-get install -y  python3-dev python3-matplotlib libfreetype6-dev python-pil python3-scipy

RUN pip3  install --no-cache-dir $PYTHON_PACKAGES
COPY test.py test.py
CMD [ "bash" ]

