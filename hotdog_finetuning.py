# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow as tf
import numpy as np
import time

from tensorflow import keras
from tensorflow.python import ipu

# For distributed training with poprun
from tensorflow.python.ipu import horovod as hvd
import popdist
hvd.init()

import argparse

parser = argparse.ArgumentParser(
  description='TF2 Finetuning of mobilenet model',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

parser.add_argument('-is', '--image-side', default=224, type=int, help='Image Side')
parser.add_argument('-ne', '--num-epochs', default=8, type=int, help='Epochs')
parser.add_argument('-bs', '--batch-size', default=32, type=int, help='Batch Size')
parser.add_argument('-gac', '--gradient-accumulation-count', default=1, type=int, help='Gradient Accumulation Count per Replica')
parser.add_argument('-t', '--target', default="IPU", type=str, help='Hardware Target')
parser.add_argument('-nio', '--num-io-tiles', default=0, type=int, help='Number of I/O Tiles')
parser.add_argument('-nr', '--num-replicas', default=1, type=int, help='Number of Replicas')
parser.add_argument('-np', '--num-pipeline-stages', default=1, type=int, help='Number of Pipeline Stages')
parser.add_argument('-popvg', '--create-popvision-graph-report', action='store_true', help='Enable popvision graph report generation') #False if argument is not used!
parser.add_argument('-popvs', '--create-popvision-system-report', action='store_true', help='Enable popvision system report generation') #False if argument is not used!
parser.add_argument('-mn', '--model-name', default="EfficientNetB4", choices=['EfficientNetB4', 'MobileNetV2'], help='Specify the model name') #False if argument is not used!
parser.add_argument('-idt', '--input-dtype', default='fp32', choices=['fp32', 'fp16', 'int8'], help="Choose input data type") 

args = parser.parse_args()

param_id_string = f'bs{args.batch_size}_gac{args.gradient_accumulation_count}_is{args.image_side}_nio{args.num_io_tiles}'

print(args)

# Training Data
train_data_dir = "./ai_transfer_learning/training_and_validation_images"

#Hyperparameters
IMAGE_SIDE = args.image_side
INPUT_SHAPE = (IMAGE_SIDE, IMAGE_SIDE, 3)
IMAGE_SHAPE = (IMAGE_SIDE, IMAGE_SIDE)

# Training Parameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
DROPOUT_RATE = 0.2
VALIDATION_SPLIT = 0.2
VERBOSE = 1

import os
file_count = sum(len(files) for _, _, files in os.walk(train_data_dir))
NUM_SAMPLES = file_count #All files used for training and validation
print(NUM_SAMPLES)
#NUM_SAMPLES = 3200
global_batch_size = BATCH_SIZE * popdist.getNumTotalReplicas()
effective_batch_size = BATCH_SIZE * args.gradient_accumulation_count
STEPS_PER_EPOCH = NUM_SAMPLES // global_batch_size # stpep
STEPS_PER_EPOCH *= args.num_replicas
print(f"STEPS_PER_EPOCH {STEPS_PER_EPOCH}")

############################ ResNet50 ##############################
# NUM_EPOCHS = 4
# validation accuracy is: 0.9534 
# test accuracy (using test images not seen during training): 0.7188,
# walltime: 340s 

# NUM_EPOCHS = 8
# validation accuracy is: 0.9547 
# test accuracy (using test images not seen during training): 0.7188,
# walltime: 659s

############################ ResNet50V2 ##############################
# validation accuracy is: 0.9975 
# test accuracy (using test images not seen during training): 0.9375
# walltime: 320s

############################ MobileNetV2 ##############################
# Epoch 8/8
# IPU: 10/10 [==============================] - 4s 374ms/step - loss: 0.1518 - accuracy: 0.9594 - val_loss: 0.0295 - val_accuracy: 1.0000
# CPU: 10/10 [==============================] - 3s 344ms/step - loss: 0.1236 - accuracy: 0.9563 - val_loss: 0.0167 - val_accuracy: 1.0000

def create_model():
  import importlib
  ### Import model
  #from tensorflow.keras.applications.resnet50 import ResNet50 as model
  #from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as model
  if args.model_name == "EfficientNetB4":
    from tensorflow.keras.applications.efficientnet import EfficientNetB4 as model
  elif args.model_name == "MobileNetV2":
    from tensorflow.keras.applications import MobileNetV2 as model
  # no worky: 
  #model = importlib.import_module("tensorflow.keras.applications", "MobileNetV2")
  #model = importlib.import_module("..MobileNetV2", "tensorflow.keras.applications.MobileNetV2")
  print(model)
  
  # Create headless model
  headless_model = model(include_top=False, input_shape=INPUT_SHAPE)
  headless_model.trainable = False 

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  # New classifier to be trained with finetuning / transfer learning
  classifier = tf.keras.layers.Dense(2)
  
  # Assemble the model using functional API adding the classification head
  inputs = tf.keras.Input(shape=INPUT_SHAPE)
  x = headless_model(inputs)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
  outputs = classifier(x)

  model = tf.keras.Model(inputs, outputs, name=f"{headless_model.name}_hotdog")

  model.summary()
  return model

# Training Data Generation
# Cannot create Dataset from this 
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#train_datagen = ImageDataGenerator(rescale=1./255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    validation_split=VALIDATION_SPLIT) # set validation split

# Train the model
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['accuracy'])
    self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()

def optimise_io(dataset):
  # Enable prefetch to decrease StreamCopyBegin proportion
  dataset = dataset.cache() 
  dataset = dataset.prefetch(args.gradient_accumulation_count) # Will prefetch gradient_accumulation_count batches!
  # this doesn't 
  # Note: swapping this doesn't get rid of this warining:
  #   The calling iterator did not fully read the dataset being cached. 
  #   In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset will be discarded. 
  #   This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.

  return dataset

def labeled_data_mapfun(data, labels):
  return(tf.cast(data, np.uint8), labels)

def configure_dataset(dataset):
    # Workaround: unbatch and then batch with explicit BATCH_SIZE

    dataset = dataset.unbatch()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).repeat()

    dataset = dataset.cache()
    dataset = dataset.prefetch(16)

    if args.input_dtype == 'fp16':
       	dataset = dataset.map(lambda x : tf.cast(x, np.float16))
    elif args.input_dtype == 'int8':
        dataset = dataset.map(lambda x : (tf.cast(x[0], np.uint8), x[1]))
        #dataset = dataset.map(lambda x : x*x)

    dataset = optimise_io(dataset)
  
    # shard for distributed training with poprun
    dataset = dataset.shard(
      num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())
    return dataset

# I followed https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/examples_tf2.html#training-on-the-ipu 
# to a large extent
def train_model(model):
    print(f"Executing model.compile(...)")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        steps_per_execution=STEPS_PER_EPOCH,
        metrics=['accuracy']
    )

    training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        validation_split=VALIDATION_SPLIT, # Use 20% of data for validation
        subset='training',
        seed=1234,
        label_mode='categorical',
        image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        #This does not achieve a defined shape in dimention 1: batch_size=BATCH_SIZE
        )

    print(training_dataset)
    print(training_dataset.element_spec)

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        validation_split=VALIDATION_SPLIT, # Use 20% of data for validation
        subset='validation',
        seed=1234,
        label_mode='categorical',
        image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        #This does not achieve a defined shape in dimention 1: batch_size=BATCH_SIZE
        )

    # computes the wrong value
    # total_samples = len(training_dataset) + len(validation_dataset)
    #print(f"Total samples in the Dataset: {total_samples}")

    training_dataset = configure_dataset(training_dataset)

    validation_dataset = configure_dataset(validation_dataset)

    print(f"Executing model.fit(...)")
    history = model.fit(
        training_dataset,
        epochs = NUM_EPOCHS,
        validation_data=validation_dataset,
        steps_per_epoch = STEPS_PER_EPOCH,
        validation_steps = int(VALIDATION_SPLIT*STEPS_PER_EPOCH), # why is this not automatically derived?
        verbose=VERBOSE,
        callbacks=[batch_stats_callback]
    )

# Configure the IPU system
def config_ipu(verbose_model_name):
  cfg = ipu.config.IPUConfig()
  cfg.auto_select_ipus = args.num_replicas
  if args.create_popvision_graph_report:
    report_name = f'report_{verbose_model_name}'   
    print(f"Adding compilation_poplar_options to config to create report {report_name}")
    cfg.compilation_poplar_options = {
      #'POPLAR_ENGINE_OPTIONS': {
        "autoReport.all":"true", 
        "autoReport.directory":f'{report_name}',
      #}
    }
  if args.num_io_tiles>0:
    print(f"Enabling {args.num_io_tiles} IO Tiles")
    cfg.io_tiles.num_io_tiles = args.num_io_tiles
    cfg.io_tiles.place_ops_on_io_tiles = True
  cfg.configure_ipu_system()

def save_model(verbose_model_name):
  timestamp = time.time()
  export_path = f"./saved_models/retrained_{verbose_model_name}_{int(timestamp)}"
  model.save(export_path)
  print(f"saved model to {export_path}")

if __name__ == "__main__":

  if args.target=="IPU":
      print("Training on IPU")
      # Create an IPU distribution strategy.
      strategy = ipu.ipu_strategy.IPUStrategy()
      with strategy.scope():
        model = create_model()
        model.set_pipelining_options(gradient_accumulation_steps_per_replica=args.gradient_accumulation_count)
        verbose_model_name = f'IPU_{model.name}_{param_id_string}'
        config_ipu(verbose_model_name)
        train_model(model)
        save_model(verbose_model_name)
  else:
      print("Training on CPU")
      strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
      with strategy.scope():
        model = create_model()
        verbose_model_name = f'CPU_{model.name}_{param_id_string}'
        train_model(model)
        save_model(verbose_model_name)

# Evaluate is done as part of fit with validation_data
'''
  # Get the test data
  batch_size = 100
  data_dir = "/localdata/thomasg/ai_transfer_learning/test_images"

  # Not supported yet in TF 2.1
  test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    shuffle=False,
    label_mode='categorical',
    image_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
    batch_size=BATCH_SIZE)

  #classes = test_generator.class_names
  #print("classes: "+str(classes))

  # Evaluate the model
  model.evaluate(test_dataset)

  #import onnx
  #import keras2onnx

  #onnx_model_name = "keras_onnx_model"
  #
  #onnx_model = keras2onnx.convert_keras(model, model.name)
  #onnx.save_model(onnx_model, onnx_model_name)
'''
