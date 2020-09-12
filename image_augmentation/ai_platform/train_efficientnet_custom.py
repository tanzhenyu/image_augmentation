"""Train EfficientNet(s) using custom training loop on ImageNet using Cloud TPUs."""

import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from tensorflow.python.keras.applications import efficientnet

from image_augmentation.datasets import large_imagenet
from image_augmentation.image import autoaugment_policy, PolicyAugmentation, RandAugment
from image_augmentation.preprocessing.efficientnet_preprocess import preprocess_fn_builder
from image_augmentation.optimizer_schedules import WarmupExponentialDecay


def get_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet (using a custom training loop)'
                                                 ' on ImageNet dataset')

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
        type=float,
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
        '--moving-average-decay',
        default=0.9999,
        type=float,
        help='rate of moving average decay applied on trainable variables, '
             'batch norm moving mean and variance, default=0.9999')
    parser.add_argument(
        '--val-every',
        default=10,
        type=int,
        help='number of epochs to set how frequent '
             'validation would be performed, default=10')
    parser.add_argument(
        '--tpu',
        default=None,
        type=str,
        help='the gRPC URL of the TPU that is to be used, default=None (TPU not used)')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args = parser.parse_args()
    return args


EFFICIENTNET = {
    'efficientnet-b0': {
        'image_size': 224,
        'model_builder': efficientnet.EfficientNetB0
    },
    'efficientnet-b1': {
        'image_size': 240,
        'model_builder': efficientnet.EfficientNetB1
    },
    'efficientnet-b2': {
        'image_size': 260,
        'model_builder': efficientnet.EfficientNetB2
    },
    'efficientnet-b3': {
        'image_size': 300,
        'model_builder': efficientnet.EfficientNetB3
    },
    'efficientnet-b4': {
        'image_size': 380,
        'model_builder': efficientnet.EfficientNetB4
    },
    'efficientnet-b5': {
        'image_size': 456,
        'model_builder': efficientnet.EfficientNetB5
    },
    'efficientnet-b6': {
        'image_size': 528,
        'model_builder': efficientnet.EfficientNetB6
    },
    'efficientnet-b7': {
        'image_size': 600,
        'model_builder': efficientnet.EfficientNetB7
    }
}

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def main(args):
    # set level of verbosity
    logging.getLogger("tensorflow").setLevel(args.verbosity)

    logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.getLogger().setLevel(args.verbosity)

    # display script args
    logging.info("Arguments: %s", str(args))

    # use cross-replica distribute Strategy-aware batch normalization layers
    if args.tpu:
        # inject SyncBatchNormalization at module level so as to speed up training on TPU
        # by using this the keras applications EfficientNet architecture need not be rewritten
        # for use on a TPU
        # Note: SyncBatchNormalization Keras layer has support for cross-replica
        # while teh standard BatchNormalization layer can only support a single node
        efficientnet.layers.BatchNormalization = tf.keras.layers.experimental.SyncBatchNormalization

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
    logging.info('Number of available devices: %d', strategy.num_replicas_in_sync)

    image_size = EFFICIENTNET[args.model_name]['image_size']
    logging.info('Using image size: %d', image_size)

    with strategy.scope():
        logging.info("Using architecture: %s", args.model_name)
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
        logging.info("Using AutoAugment pre-processing")
        policy = autoaugment_policy('imagenet', efficientnet=True)
        auto_augment = PolicyAugmentation(policy, cutout_max_size=100, translate_max=250)
        train_ds = train_ds.map(lambda image, label: (auto_augment.apply_on_image(image), label),
                                tf.data.experimental.AUTOTUNE)

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
    train_ds = train_ds.shuffle(1024,
                                reshuffle_each_iteration=True).repeat().batch(args.train_batch_size,
                                                                              drop_remainder=True)
    minival_ds = minival_ds.batch(args.val_batch_size, drop_remainder=True)
    val_ds = val_ds.batch(args.val_batch_size, drop_remainder=True)

    # prefetch dataset for faster access
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    minival_ds = minival_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # calculate steps per epoch for optimizer schedule num steps
    steps_per_epoch = tf.data.experimental.cardinality(ds['train_ds']) // args.train_batch_size
    steps_per_epoch = int(steps_per_epoch.numpy())  # helps model optimizer become JSON serializable

    init_lr = args.base_lr * (args.train_batch_size / 256.)

    with strategy.scope():
        if args.lr_decay_rate:
            logging.info("Using exponentially decayed learning rate: %f", args.lr_decay_rate)
            # use a few starting warmup epochs with exponentially decayed LR
            if args.warmup_epochs:
                logging.info("Using warmup epochs: %0.3f", args.warmup_epochs)
                lr = WarmupExponentialDecay(init_lr, int(steps_per_epoch * args.lr_decay_epochs),
                                            args.lr_decay_rate, int(steps_per_epoch * args.warmup_epochs),
                                            staircase=True)
            # do not use warmup
            else:
                lr = keras.optimizers.schedules.ExponentialDecay(init_lr, int(steps_per_epoch * args.lr_decay_epochs),
                                                                 args.lr_decay_rate, staircase=True)
        # do not use exponential decay
        else:
            logging.info("Using constant learning rate: %f", init_lr)
            lr = init_lr

        logging.info("Using optimizer: %s", args.optimizer)
        # use an RMSprop optimizer
        if args.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr,
                                                 rho=args.optimizer_decay,
                                                 momentum=args.optimizer_momentum,
                                                 epsilon=0.001)
        # use SGD optimizer w/ or w/o momentum
        elif args.optimizer == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=lr,
                                             momentum=args.optimizer_momentum)
        # use adam optimizer
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        has_ema = args.moving_average_decay > 0
        if has_ema:
            optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=args.moving_average_decay)

        crossentropy_loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smoothing,
            reduction=keras.losses.Reduction.NONE)

        # use Cross Entropy with L2 Regularization loss
        def loss_fn(labels, predictions):
            per_example_loss = crossentropy_loss(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss,
                                              global_batch_size=args.train_batch_size)

            # use tf.nn.l2_loss for L2 regularization of
            # all model trainable variables except those
            # being used for batch normalization
            for var in model.trainable_variables:
                if 'bn' not in var.name:
                    loss += tf.nn.scale_regularization_loss(
                        args.l2_regularization * tf.nn.l2_loss(var))
            return loss

        # evaluation metrics
        minival_loss = tf.keras.metrics.Mean(name='minival_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='train_accuracy')
        minival_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='minival_accuracy')
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='val_accuracy')

        # use top-5 accuracy (since, dealing with ImageNet)
        train_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='train_top_k_accuracy')
        minival_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='minival_top_k_accuracy')
        val_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='val_top_k_accuracy')

        # allow model to checkpoint
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def train_step(dataset_inputs):
        """One single step of training."""
        images, labels = dataset_inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        if not has_ema:
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            grad_list = gradients
            var_list = model.trainable_variables
            # also apply EMA on Batch Norm moving mean and variance
            for var in model.variables:
                if ('moving_mean' in var.name) or ('moving_variance' in var.name):
                    var_list.append(var)
                    # as moving_mean and moving_variance are
                    # non-trainable parameters, apply zero grad to each
                    grad_list.append(tf.zeros_like(var))
            # in this way, the MovingAverage optimizer
            # would keep track of BN non-trainable vars also
            optimizer.apply_gradients(zip(grad_list, var_list))

        train_accuracy.update_state(labels, predictions)
        train_k_accuracy.update_state(labels, predictions)
        return loss

    def eval_step_builder(loss_metric, accuracy_metric, top_k_accuracy_metric):
        """Returns a function that can be used for evaluation with given eval metrics.
        (allows re-usability across minival, val splits)"""
        def eval_step(dataset_inputs):
            """One single step of evaluation with specified eval metrics."""
            images, labels = dataset_inputs

            predictions = model(images, training=False)
            cce_loss = crossentropy_loss(labels, predictions)

            loss_metric.update_state(cce_loss)
            accuracy_metric.update_state(labels, predictions)
            top_k_accuracy_metric.update_state(labels, predictions)
        return eval_step

    @tf.function
    def distributed_train_step(dataset_inputs):
        """Distribute the training step across strategy aware replicas."""
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_minival_step(dataset_inputs):
        """Distribute the minival evaluation step across strategy aware replicas."""
        eval_step = eval_step_builder(minival_loss, minival_accuracy, minival_k_accuracy)
        return strategy.run(eval_step, args=(dataset_inputs,))

    @tf.function
    def distributed_val_step(dataset_inputs):
        """Distribute the val evaluation step across strategy aware replicas."""
        eval_step = eval_step_builder(val_loss, val_accuracy, val_k_accuracy)
        return strategy.run(eval_step, args=(dataset_inputs,))

    # setup TensorBoard logging (separately for train, minival, val)
    tensorboard_path = args.job_dir + '/tensorboard'
    splits = ["train", "minival", "val"]
    writers = {
        split: tf.summary.create_file_writer(tensorboard_path + "/" + split)
        for split in splits
    }

    def log_tensorboard(writer, tags, values, epoch):
        """Log scalars (specific to given tags) for one step (epoch-wise) of training / eval."""
        with writer.as_default():
            for tag, value in zip(tags, values):
                tf.summary.scalar(tag, value, step=epoch)

    # use a checkpoint manager to keep only the 10 latest checkpoints
    checkpoint_path = args.job_dir + '/checkpoint'
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=10)

    # distribute the dataset pipelines in run with the respective strategy
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    minival_dist_ds = strategy.experimental_distribute_dataset(minival_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

    # epoch wise model training and evaluation
    for epoch in range(args.epochs):

        # Training Loop
        total_loss = 0.0
        num_batches = 0
        for epoch_step, inputs in enumerate(train_dist_ds):
            if epoch_step >= steps_per_epoch:
                break  # epoch finished
            total_loss += distributed_train_step(inputs)
            num_batches += 1

        train_loss = total_loss / num_batches

        # backup non averaged model (training) weights
        if has_ema:
            non_avg_weights = model.get_weights()
            optimizer.assign_average_vars(model.variables)

        # save checkpoint
        checkpoint_manager.save()
        logging.info("Saved model checkpoint to: %s", checkpoint_path)

        tags = ['epoch_loss', 'epoch_accuracy', 'epoch_top_k_accuracy']

        log_tensorboard(writers['train'], tags,
                        [train_loss, train_accuracy.result(), train_k_accuracy.result()],
                        epoch)

        # perform validation only as frequent as specified
        should_eval = (epoch + 1) % args.val_every == 0

        if should_eval:
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

        # restore non average weights after checkpoint and evaluation
        # for training model ahead
        if has_ema:
            model.set_weights(non_avg_weights)

        # helpful to print learning rate
        if hasattr(optimizer.learning_rate,  '__call__'):
            current_lr = optimizer.learning_rate(optimizer.iterations)
        else:
            current_lr = optimizer.learning_rate

        dashes = "=" * 130
        template = ("learning_rate: {},\n"
                    "loss: {}, accuracy: {}, top_k_accuracy: {},\n")
        values = (current_lr,
                  train_loss,
                  train_accuracy.result() * 100,
                  train_k_accuracy.result() * 100)

        if should_eval:
            template += ("minival_loss: {}, minival_accuracy: {}, minival_top_k_accuracy: {},\n"
                         "val_loss: {}, val_accuracy: {}, val_top_k_accuracy: {}\n")
            values += (minival_loss.result(),
                       minival_accuracy.result() * 100,
                       minival_k_accuracy.result() * 100,
                       val_loss.result(),
                       val_accuracy.result() * 100,
                       val_k_accuracy.result() * 100)

        template = ("{}\nEpoch {}/{} " + template + "{}")
        # print training loss and other aggregated metrics (for this epoch)
        logging.info("\n%s\n", template.format(dashes, epoch + 1, args.epochs,
                                               *values, dashes))

        # reset train metrics for next epoch
        train_accuracy.reset_states()
        train_k_accuracy.reset_states()

        # reset eval metrics
        if should_eval:
            minival_loss.reset_states()
            minival_accuracy.reset_states()
            minival_k_accuracy.reset_states()

            val_loss.reset_states()
            val_accuracy.reset_states()
            val_k_accuracy.reset_states()

    logging.info("Training finished")

    save_path = args.job_dir + '/keras_model'
    model.save(save_path)
    logging.info("Saved keras model to: %s", save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
