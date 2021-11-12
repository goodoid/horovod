#
# ResNet-50 model training using Keras and Horovod.
#
# This model is an example of a computation-intensive model that achieves good accuracy on an image
# classification task.  It brings together distributed training concepts such as learning rate
# schedule adjustments with a warmup, randomized data reading, and checkpointing on the first worker
# only.
#
# Note: This model uses Keras native ImageDataGenerator and not the sophisticated preprocessing
# pipeline that is typically used to train state-of-the-art ResNet-50 model.  This results in ~0.5%
# increase in the top-1 validation error compared to the single-crop top-1 validation error from
# https://github.com/KaimingHe/deep-residual-networks.
#
import argparse
import keras
from tensorflow.compat.v1.keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import horovod.keras as hvd
import os
import pickle
import numpy as np


# Get a ResNet50 model
def resnet50_model(classes=1000, *args, **kwargs):
    # Load a model if we have saved one
    if (os.path.isfile('C:\\DATA\\Python-data\\CIFAR-10\\models\\resnet_50.h5') == True):
        return keras.models.load_model('C:\\DATA\\Python-data\\CIFAR-10\\models\\resnet_50.h5')
    # Create an input layer
    input = keras.layers.Input(shape=(None, None, 3))
    # Create output layers
    output = keras.layers.ZeroPadding2D(padding=3, name='padding_conv1')(input)
    output = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(output)
    output = keras.layers.BatchNormalization(axis=3, epsilon=1e-5, name='bn_conv1')(output)
    output = keras.layers.Activation('relu', name='conv1_relu')(output)
    output = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(output)
    output = conv_block(output, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    output = identity_block(output, 3, [64, 64, 256], stage=2, block='b')
    output = identity_block(output, 3, [64, 64, 256], stage=2, block='c')
    output = conv_block(output, 3, [128, 128, 512], stage=3, block='a')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='b')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='c')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='d')
    output = conv_block(output, 3, [256, 256, 1024], stage=4, block='a')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='b')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='c')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='d')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='e')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='f')
    output = conv_block(output, 3, [512, 512, 2048], stage=5, block='a')
    output = identity_block(output, 3, [512, 512, 2048], stage=5, block='b')
    output = identity_block(output, 3, [512, 512, 2048], stage=5, block='c')
    output = keras.layers.GlobalAveragePooling2D(name='pool5')(output)
    output = keras.layers.Dense(classes, activation='softmax', name='fc1000')(output)
    # Create a model from input layer and output layers
    model = keras.models.Model(inputs=input, outputs=output, *args, **kwargs)
    # Print model
    print()
    print(model.summary(), '\n')
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.01, clipnorm=0.001),
                  metrics=['accuracy'])
    # Return a model
    return model


# Create an identity block
def identity_block(input, kernel_size, filters, stage, block):
    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create layers
    output = keras.layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                                 name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(output)
    output = keras.layers.add([output, input])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output


# Create a convolution block
def conv_block(input, kernel_size, filters, stage, block, strides=(2, 2)):
    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create block layers
    output = keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal',
                                 name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                                 name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(output)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal',
                                   name=conv_name_base + '1')(input)
    shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
    output = keras.layers.add([output, shortcut])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output


parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

args = parser.parse_args()


print("got args:", args)
# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.compat.v1.Session(config=config))

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

img_width, img_height = 32, 32
# img_width, img_height = 224, 224

# Training data iterator.
train_gen = image.ImageDataGenerator(rescale=1./255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)
train_iter = train_gen.flow_from_directory(args.train_dir,
                                           class_mode='categorical',
                                           batch_size=args.batch_size,
                                           color_mode='rgb',
                                           shuffle=True,
                                           target_size=(img_width, img_height))

# Validation data iterator.
test_gen = image.ImageDataGenerator(
    zoom_range=(0.875, 0.875), preprocessing_function=keras.applications.resnet50.preprocess_input)
test_iter = test_gen.flow_from_directory(args.val_dir,
                                         class_mode='categorical',
                                         color_mode='rgb',
                                         shuffle=True,
                                         batch_size=args.val_batch_size,
                                         target_size=(img_width, img_height))

# Set up standard ResNet-50 model.
#model = keras.applications.resnet50.ResNet50(classes=10)
model = resnet50_model(10)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: adjust learning rate based on number of GPUs.
initial_lr = args.base_lr * hvd.size()

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast both model and optimizer weights
# to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch),
                           compression=compression)
else:
    # ResNet-50 model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = keras.models.Model.from_config(model_config)
    opt = keras.optimizers.SGD(lr=initial_lr, momentum=args.momentum)

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt, compression=compression)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

model.fit(train_iter, steps_per_epoch=1, epochs=1, callbacks=None)
state = hvd.elastic.KerasState(model, batch=0, epoch=0)
def on_state_reset():
    tf.keras.backend.set_value(state.model.optimizer.lr,  0.001 * hvd.size())
    # Re-initialize, to join with possible new ranks
    state.model.fit(train_iter, steps_per_epoch=1, epochs=1, callbacks=None)

state.register_reset_callbacks([on_state_reset])

callbacks = [
    hvd.elastic.UpdateEpochStateCallback(state),
    hvd.elastic.UpdateBatchStateCallback(state),
    hvd.elastic.CommitStateCallback(state),
]

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
    callbacks.append(keras.callbacks.TensorBoard(args.log_dir))

# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data helps to increase probability that every validation
# example will be evaluated.
@hvd.elastic.run
def train():
    model.fit(train_iter,
              steps_per_epoch=len(train_iter) // hvd.size(),
              callbacks=callbacks,
              epochs=args.epochs,
              verbose=verbose,
              workers=4,
              initial_epoch=resume_from_epoch,
              validation_data=test_iter,
              validation_steps=3 * len(test_iter) // hvd.size())

# Evaluate the model on the full data set.
score = hvd.allreduce(model.evaluate_generator(test_iter, len(test_iter), workers=4))
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])