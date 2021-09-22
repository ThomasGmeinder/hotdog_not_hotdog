Keras example: MobilenetV2 finetuning on the IPU
------------------------------------------------

This example creates a headless pretrained computer vision model (e.g. MobileNetV2) and adds a classification layer that will be trained to classify images into the categories hotdog and not-hotdog.

Execute ``hotdog_finetuning.py --help`` for information on arguments and default settings.
E.g. the ``--target`` argument can be used to switch between IPU (default) and CPU.

Requirements:
* Installed and enabled Poplar
* Installed the Graphcore port of TensorFlow 2

Refer to the Getting Started guide for your IPU System for instructions.

#### File Structure
* `hotdog_finetuning.py` A demonstration script, where code is edited to illustrate the differences between running a Keras model on the CPU and IPU
* `README.md` This file

