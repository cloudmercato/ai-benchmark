FROM tensorflow/tensorflow 
ENV UNIT=cpu

WORKDIR /ai-benchmark
ADD . /ai-benchmark

RUN pip install .
