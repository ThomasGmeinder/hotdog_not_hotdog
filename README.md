Keras example: Computervision finetuning on the IPU
---------------------------------------------------

This example creates a headless pretrained computer vision model (e.g. MobileNetV2) and adds a classification layer that will be trained to classify images into the categories hotdog and not-hotdog.

Requirements:
* Installed and enabled Poplar. On Graphcloud follow https://docs.graphcore.ai/projects/graphcloud-getting-started/en/latest/installation.html#setting-up-the-sdk-environment
* Installed the Graphcore port of TensorFlow 2. On Graphcloud follow https://docs.graphcore.ai/projects/graphcloud-getting-started/en/latest/installation.html#setting-up-tensorflow-for-the-ipu

#### File Structure
* `hotdog_finetuning.py` A demonstration script, where code is edited to illustrate the differences between running a Keras model on the CPU and IPU
* `README.md` This file

#### Steps to run the finetuning
* Change into a run directory of your choice. For best performance make sure that a SSD drive is mounted to that directory. 
* Download the training data by executing: ``https://github.com/ThomasGmeinder/ai_transfer_learning``
* Execute the script ``python3 <your path>/hotdog_not_hotdog/hotdog_finetuning.py``
* Use configuration parameters to vary configurations. Execute with ``--help`` for more information on configuration and default settings.

