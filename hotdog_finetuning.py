# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow as tf
import numpy as np
import time

from tensorflow import keras
from tensorflow.python import ipu

import argparse

parser = argparse.ArgumentParser(
  description='TF2 Finetuning of mobilenet model',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

parser.add_argument('-is', '--image-side', default=224, type=int, help='Image Side')
parser.add_argument('-ne', '--num-epochs', default=8, type=int, help='Epochs')
parser.add_argument('-t', '--target', default="IPU", type=str, help='Hardware Target')
args = parser.parse_args()

#Hyperparameters
IMAGE_SIDE = args.image_side
INPUT_SHAPE = (IMAGE_SIDE, IMAGE_SIDE, 3)
IMAGE_SHAPE = (IMAGE_SIDE, IMAGE_SIDE)

# Training Parameters
NUM_EPOCHS = args.num_epochs
DROPOUT_RATE = 0.2
VALIDATION_SPLIT = 0.2
VERBOSE = 1

BATCH_SIZE=32

STEPS_PER_EPOCH = 10
STEPS_PER_EXECUTION = 100

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
  ### Import model
  #from tensorflow.keras.applications.resnet50 import ResNet50 as model
  #from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as model
  #from tensorflow.keras.applications.efficientnet import EfficientNetB4 as model
  from tensorflow.keras.applications import MobileNetV2 as model
  
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


train_data_dir = "/localdata/thomasg/ai_transfer_learning/training_and_validation_images"
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


# I followed https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/examples_tf2.html#training-on-the-ipu 
# to a large extent
def train_model(model):
    print(f"Executing model.compile(...)")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        steps_per_execution=STEPS_PER_EXECUTION,
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

    # Workaround: unbatch and then batch with explicit BATCH_SIZE
    training_dataset = training_dataset.unbatch()
    training_dataset = training_dataset.batch(BATCH_SIZE, drop_remainder=True)

    validation_dataset = validation_dataset.unbatch()
    validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)

    print(f"Executing model.fit(...)")
    history = model.fit(
        training_dataset.repeat(),
        epochs = NUM_EPOCHS,
        validation_data=validation_dataset.repeat(),
        steps_per_epoch = STEPS_PER_EPOCH,
        validation_steps = int(VALIDATION_SPLIT*STEPS_PER_EPOCH), # why is this not automatically derived?
        verbose=VERBOSE,
        callbacks=[batch_stats_callback]
    )

if __name__ == "__main__":
  if args.target=="IPU":
      # Configure the IPU system
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()
      # Create an IPU distribution strategy.
      strategy = ipu.ipu_strategy.IPUStrategy()
  else:
      #strategy = DummyStrategy()
      strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
      #strategy = tf.distribute.Strategy("CPU")
  
  with strategy.scope():
        model = create_model()
        print(f"Model name: {model.name}")
        train_model(model)

  timestamp = time.time()

  export_path = "./saved_models/retrained_{}_{}".format(model.name, int(timestamp))
  model.save(export_path)
  print(f"saved model to {export_path}")

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