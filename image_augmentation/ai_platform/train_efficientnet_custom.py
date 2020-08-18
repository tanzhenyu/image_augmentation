"""Train EfficientNet(s) using custom training loop on ImageNet using Cloud TPUs."""

import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

from image_augmentation.datasets import large_imagenet
from image_augmentation.image import RandAugment
from image_augmentation.preprocessing.efficientnet_preprocess import preprocess_fn_builder
from image_augmentation.optimizer_schedules import WarmupExponentialDecay


def get_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet on ImageNet dataset')

    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints of models and other results')
    parser.add_argument(
        '--epochs',
        type=int,
        default=350,
        help='number of times to go through the data, default=120')
    parser.add_argument(
        '--train-batch-size',
        default=2048,
        type=int,
        help='size of each training batch, default=2048')
    parser.add_argument(
        '--val-batch-size',
        default=1000,
        type=int,
        help='size of each validation batch, default=64')
    parser.add_argument(
        '--model-name',
        default='efficientnet-b0',
        choices=['efficientnet-b0', 'efficientnet-b1',
                 'efficientnet-b2', 'efficientnet-b3',
                 'efficientnet-b4', 'efficientnet-b5',
                 'efficientnet-b6', 'efficientnet-b7'],
        type=str,
        help='EfficientNet architecture that is to be used, default=efficientnet-b0')
    parser.add_argument(
        '--auto-augment',
        default=False,
        const=True,
        action='store_const',
        help='apply AutoAugment policy for data augmentation on training, off by default (no AutoAugment)')
    parser.add_argument(
        '--rand-augment-n',
        default=0,
        type=int,
        help='apply RandAugment with number of (N) image transforms for data augmentation on training, '
             'default=0, off (no RandAugment)')
    parser.add_argument(
        '--rand-augment-m',
        default=10,
        type=int,
        help='magnitude (M) of applying each image transform for RandAugment, '
             '(only when using RandAugment) default=10')
    parser.add_argument(
        '--data-dir',
        default=None,
        type=str,
        help='local or GCS location for accessing data with TFDS '
             '(directory for tensorflow_datasets)')
    parser.add_argument(
        '--optimizer',
        default='rmsprop',
        choices=["rmsprop", "sgd", "adam"],
        help='optimizer that is to be used for training, default="rmsprop"')
    parser.add_argument(
        '--optimizer-decay',
        default=0.9,
        type=float,
        help='optimizer decay rate for RMSprop, default=0.9')
    parser.add_argument(
        '--optimizer-momentum',
        default=0.9,
        type=float,
        help='optimizer momentum rate for RMSprop and SGD, default=0.9')
    parser.add_argument(
        '--label-smoothing',
        default=0.1,
        type=int,
        help='amount of label smoothing to be applied on '
             'categorical cross entropy (softmax) loss, default=0.1')
    parser.add_argument(
        '--base-lr',
        default=0.016,
        type=float,
        help='initial learning rate for training with a base batch size of 256, '
             'the value will be scaled according to batch size specified, default=0.016')
    parser.add_argument(
        '--lr-decay-rate',
        default=0.97,
        type=float,
        help='amount of learning rate (exponential) decay to be applied, default=0.97')
    parser.add_argument(
        '--lr-decay-epochs',
        default=2.4,
        type=float,
        help='number of epochs to wait before each exponential decay, default=10')
    parser.add_argument(
        '--warmup-epochs',
        default=5.,
        type=float,
        help='number of warmup epochs, default=5.')
    parser.add_argument(
        '--l2-regularization',
        default=1e-5,
        type=float,
        help='amount of L2 regularization (weight decay rate) to be applied on all weights '
             'of the network, default=1e-5')
    parser.add_argument(
        '--early-stopping',
        default=False,
        const=True,
        action='store_const',
        help='use early stopping based on mini-val split (if available), '
             'default=off')
    parser.add_argument(
        '--tpu',
        default=None,
        type=str,
        help='the gRPC URL of the TPU that is to be used, default=None (TPU not used)')
    parser.add_argument(
        '--resume-training-from',
        default=None,
        type=int,
        help='number of epochs to resume training from '
             '(will load weights from an earlier checkpoint), default=off')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args = parser.parse_args()
    return args


EFFICIENTNET = {
    'efficientnet-b0': {
        'image_size': 224,
        'model_builder': keras.applications.efficientnet.EfficientNetB0
    },
    'efficientnet-b1': {
        'image_size': 240,
        'model_builder': keras.applications.efficientnet.EfficientNetB1
    },
    'efficientnet-b2': {
        'image_size': 260,
        'model_builder': keras.applications.efficientnet.EfficientNetB2
    },
    'efficientnet-b3': {
        'image_size': 300,
        'model_builder': keras.applications.efficientnet.EfficientNetB3
    },
    'efficientnet-b4': {
        'image_size': 380,
        'model_builder': keras.applications.efficientnet.EfficientNetB4
    },
    'efficientnet-b5': {
        'image_size': 456,
        'model_builder': keras.applications.efficientnet.EfficientNetB5
    },
    'efficientnet-b6': {
        'image_size': 528,
        'model_builder': keras.applications.efficientnet.EfficientNetB6
    },
    'efficientnet-b7': {
        'image_size': 600,
        'model_builder': keras.applications.efficientnet.EfficientNetB7
    }
}

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def main(args):
    # set level of verbosity
    logging.getLogger("tensorflow").setLevel(args.verbosity)

    # display script args
    logging.info(str(args))

    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        # connect to the TPU
        tf.config.experimental_connect_to_cluster(resolver)
        # initialize the TPU system
        tf.tpu.experimental.initialize_tpu_system(resolver)
        # obtain TPU distribution strategy
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        # use default distribution strategy
        strategy = tf.distribute.get_strategy()
    logging.info('Number of available devices: {}'.format(strategy.num_replicas_in_sync))

    image_size = EFFICIENTNET[args.model_name]['image_size']
    logging.info('Using image size: {}'.format(image_size))

    with strategy.scope():
        model_builder = EFFICIENTNET[args.model_name]['model_builder']
        model = model_builder(include_top=True,
                              weights=None)

        # normalize images using (ImageNet) RGB mean normalization
        normalization_layer = model.get_layer('normalization')
        norm_weights = [np.array(MEAN_RGB),
                        np.array(STDDEV_RGB),
                        np.array(0)]
        normalization_layer.set_weights(norm_weights)

        model.summary()

    ds = large_imagenet(args.data_dir)
    num_classes = ds['info'].features['label'].num_classes

    train_ds = ds['train_ds']
    minival_ds = ds['minival_ds']
    val_ds = ds['val_ds']

    # preprocess the inputs
    train_preprocess = preprocess_fn_builder(image_size, num_classes, is_training=True)
    val_preprocess = preprocess_fn_builder(image_size, num_classes, is_training=False)

    train_ds = train_ds.map(train_preprocess, tf.data.experimental.AUTOTUNE)
    minival_ds = minival_ds.map(val_preprocess, tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(val_preprocess, tf.data.experimental.AUTOTUNE)

    # apply AutoAugment (data augmentation) on training pipeline
    if args.auto_augment:
        raise NotImplementedError("AutoAugment preprocessing have not been implemented for TPUs yet. "
                                  "Consider using RandAugment preprocessing instead.")

    # apply RandAugment on training pipeline
    if args.rand_augment_n:
        logging.info("Using RandAugment based pre-processing")
        rand_augment = RandAugment(args.rand_augment_m, args.rand_augment_n,
                                   # set hyper parameters to appropriate size
                                   translate_max=100, cutout_max_size=40)
        train_ds = train_ds.map(lambda image, label: (rand_augment.apply_on_image(image), label),
                                tf.data.experimental.AUTOTUNE)

    if args.tpu:
        # use float32 image and labels in case of TPU
        # (as TPUs do not support uint8 ops)
        def cast_to_float(image, label):
            return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

        train_ds = train_ds.map(cast_to_float, tf.data.experimental.AUTOTUNE)
        minival_ds = minival_ds.map(cast_to_float, tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(cast_to_float, tf.data.experimental.AUTOTUNE)

    # shuffle and batch the dataset
    train_ds = train_ds.shuffle(1024, reshuffle_each_iteration=True).batch(args.train_batch_size,
                                                                           drop_remainder=True)
    minival_ds = minival_ds.batch(args.val_batch_size, drop_remainder=True)
    val_ds = val_ds.batch(args.val_batch_size, drop_remainder=True)

    # prefetch dataset for faster access
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    minival_ds = minival_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # calculate steps per epoch for optimizer schedule num steps
    steps_per_epoch = tf.data.experimental.cardinality(train_ds)
    steps_per_epoch = steps_per_epoch.numpy()  # helps model optimizer become JSON serializable

    init_lr = args.base_lr * (args.train_batch_size / 256.)

    with strategy.scope():
        if args.lr_decay_rate:
            # use a few starting warmup epochs with exponentially decayed LR
            if args.warmup_epochs:
                logging.info("Using %d warmup epochs", args.warmup_epochs)
                lr = WarmupExponentialDecay(init_lr, int(steps_per_epoch * args.lr_decay_epochs),
                                            args.lr_decay_rate, int(steps_per_epoch * args.warmup_epochs),
                                            staircase=True)
            # do not use warmup
            else:
                lr = keras.optimizers.schedules.ExponentialDecay(init_lr, int(steps_per_epoch * args.lr_decay_epochs),
                                                                 args.lr_decay_rate, staircase=True)
        # do not use exponential decay
        else:
            lr = init_lr

        if args.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr,
                                                 rho=args.optimizer_decay,
                                                 momentum=args.optimizer_momentum,
                                                 epsilon=0.001)
        elif args.optimizer == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=lr,
                                             momentum=args.optimizer_momentum)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        crossentropy_loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smoothing,
            reduction=keras.losses.Reduction.NONE)

        def loss_fn(labels, predictions):
            per_example_loss = crossentropy_loss(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss,
                                              global_batch_size=args.train_batch_size)

            for var in model.trainable_variables:
                if 'bn' not in var.name:
                    loss += tf.nn.scale_regularization_loss(
                        args.l2_regularization * tf.nn.l2_loss(var))
            return loss

        minival_loss = tf.keras.metrics.Mean(name='minival_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='train_accuracy')
        minival_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='minival_accuracy')
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='val_accuracy')

        train_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='train_top_k_accuracy')
        minival_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='minival_top_k_accuracy')
        val_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='val_top_k_accuracy')

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def train_step(dataset_inputs):
        images, labels = dataset_inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        train_k_accuracy.update_state(labels, predictions)
        return loss

    def eval_step_builder(loss_metric, accuracy_metric, top_k_accuracy_metric):
        def eval_step(dataset_inputs):
            images, labels = dataset_inputs

            predictions = model(images, training=False)
            cce_loss = crossentropy_loss(labels, predictions)

            loss_metric.update_state(cce_loss)
            accuracy_metric.update_state(labels, predictions)
            top_k_accuracy_metric.update_state(labels, predictions)
        return eval_step

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_minival_step(dataset_inputs):
        eval_step = eval_step_builder(minival_loss, minival_accuracy, minival_k_accuracy)
        return strategy.run(eval_step, args=(dataset_inputs,))

    @tf.function
    def distributed_val_step(dataset_inputs):
        eval_step = eval_step_builder(val_loss, val_accuracy, val_k_accuracy)
        return strategy.run(eval_step, args=(dataset_inputs,))

    tensorboard_path = args.job_dir + '/tensorboard'
    splits = ["train", "minival", "val"]
    writers = {
        split: tf.summary.create_file_writer(tensorboard_path + "/" + split)
        for split in splits
    }

    def log_tensorboard(writer, tags, values, epoch):
        with writer.as_default():
            for tag, value in zip(tags, values):
                tf.summary.scalar(tag, value, step=epoch)

    checkpoint_path = args.job_dir + '/checkpoint'
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    minival_dist_ds = strategy.experimental_distribute_dataset(minival_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

    for epoch in range(args.epochs):

        # Training Loop
        total_loss = 0.0
        num_batches = 0
        for inputs in train_dist_ds:
            total_loss += distributed_train_step(inputs)
            num_batches += 1

        train_loss = total_loss / num_batches

        checkpoint_manager.save()
        print("Saved model checkpoint to: {}".format(checkpoint_path))

        tags = ['epoch_loss', 'epoch_accuracy', 'epoch_top_k_accuracy']

        log_tensorboard(writers['train'], tags,
                        [train_loss, train_accuracy.result(), train_k_accuracy.result()],
                        epoch)

        # Evaluation on Minival split
        for inputs in minival_dist_ds:
            distributed_minival_step(inputs)

        log_tensorboard(writers['minival'], tags,
                        [minival_loss.result(), minival_accuracy.result(), minival_k_accuracy.result()],
                        epoch)

        # Evaluation on Validation split
        for inputs in val_dist_ds:
            distributed_val_step(inputs)

        log_tensorboard(writers['val'], tags,
                        [val_loss.result(), val_accuracy.result(), val_k_accuracy.result()],
                        epoch)

        if hasattr(optimizer.learning_rate,  '__call__'):
            current_lr = optimizer.learning_rate(optimizer.iterations)
        else:
            current_lr = optimizer.learning_rate

        dashes = "=" * 130
        template = ("{}\n"
                    "Epoch {}/{}, learning_rate: {},\n"
                    "loss: {}, accuracy: {}, top_k_accuracy: {},\n"
                    "minival_loss: {}, minival_accuracy: {}, minival_top_k_accuracy: {},\n"
                    "val_loss: {}, val_accuracy: {}, val_top_k_accuracy: {}\n"
                    "{}")
        values = (epoch + 1, args.epochs, current_lr,
                  train_loss,
                  train_accuracy.result() * 100,
                  train_k_accuracy.result() * 100,
                  minival_loss.result(),
                  minival_accuracy.result() * 100,
                  minival_k_accuracy.result() * 100,
                  val_loss.result(),
                  val_accuracy.result() * 100,
                  val_k_accuracy.result() * 100)
        print(template.format(dashes, *values, dashes))

        train_accuracy.reset_states()
        train_k_accuracy.reset_states()

        minival_loss.reset_states()
        minival_accuracy.reset_states()
        minival_k_accuracy.reset_states()

        val_loss.reset_states()
        val_accuracy.reset_states()
        val_k_accuracy.reset_states()

    print("Training finished")

    save_path = args.job_dir + '/keras_model'
    model.save(save_path)
    print("Saved keras model to: {}".format(save_path))


if __name__ == '__main__':
    args = get_args()
    main(args)