FROM djkormo/jetson-tensorflow-base:0.1.0
WORKDIR /home

RUN pip3 install --no-cache-dir ipykernel jupyter notebook jupyterlab
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
EXPOSE 8888
COPY test.py test.py
CMD ["bash","-c","jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token='' --no-browser "]
