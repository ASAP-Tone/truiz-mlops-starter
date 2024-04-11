

FROM gcr.io/deeplearning-platform-release/base-cpu
WORKDIR /kfp-cli
ADD pipeline.py ./pipeline.py
ADD requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
CMD python ./pipeline.py
