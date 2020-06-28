# use base image
FROM tensorflow/tensorflow:nightly

# set working directory
WORKDIR /root

# copy package source
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install tensorflow_datasets tensorflow_addons matplotlib
RUN pip install --no-deps -e .

# run training script
ENTRYPOINT ["python", "-m", "image_augmentation.ai_platform.train_wrn"]
