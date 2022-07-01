FROM tensorflow/tensorflow:2.9.1

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training
COPY requirements.txt /requirements.txt
RUN mkdir /model
# copy ./model .
RUN pip3 install --upgrade pip
# RUN echo "PWD is: $PWD"
RUN pip3 install -r requirements.txt
# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py

