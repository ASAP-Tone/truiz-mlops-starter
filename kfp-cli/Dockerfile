

FROM gcr.io/deeplearning-platform-release/base-cpu
WORKDIR /app
ADD Dockerfile ./Dockerfile
ADD pipeline.py ./pipeline.py
ADD requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
# RUN pip install kfp==2.7.0
# RUN pip install google-cloud-aiplatform==1.44.0
CMD python ./pipeline.py
#ENTRYPOINT ["/bin/bash"]
