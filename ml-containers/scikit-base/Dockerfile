# https://dev.to/faizanbashir/building-python-data-science-container-usingdocker-3f8p
FROM nvcr.io/nvidia/l4t-base:r32.2
WORKDIR /home
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_PACKAGES="\
    cython \
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    nltk \
"

RUN apt-get update && apt install -y --fix-missing make g++
RUN apt-get update && apt install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools
RUN apt-get update && apt install -y python3-h5py

RUN apt-get install -y  python3-dev python3-matplotlib libfreetype6-dev python-pil python3-scipy
#sudo apt-get install python3-matplotlib
#sudo apt-get install libfreetype6-dev  build-essential libatlas-base-dev  # for matplot
#sudo apt-get install python3-pil liblapack-dev cmake libatlas-base-dev gfortran # for matplot

RUN pip3  install --no-cache-dir $PYTHON_PACKAGES
COPY test.py test.py
CMD [ "bash" ]

