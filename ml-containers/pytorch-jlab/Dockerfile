FROM djkormo/jetson-pytorch-base:0.2.0
WORKDIR /home

RUN pip3 install --no-cache-dir ipykernel jupyter notebook jupyterlab
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
EXPOSE 8888

COPY test.py test.py
COPY test-mnist.py test-mnist.py
COPY test-cifar10.py test-cifar10.py

COPY test-mnist.ipynb test-mnist.ipynb
COPY test-cifar10.ipynb test-cifar10.ipynb

CMD ["bash","-c","jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token='' --no-browser "]
