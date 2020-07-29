# use base image
FROM tensorflow/tensorflow:2.3.0-gpu

# set working directory
WORKDIR /root

# copy package source
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install tensorboard tensorflow_datasets tfa-nightly matplotlib google-cloud-storage
RUN pip install --no-deps -e .

# run training script
ENTRYPOINT ["python", "-m", "image_augmentation.ai_platform.train_wrn"]
