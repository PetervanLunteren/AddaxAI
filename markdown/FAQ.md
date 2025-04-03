<html>
  <head>
    <meta name="google-site-verification" content="e3pjuBzSKEpVE4ec45Um-xPLb2t-p2hWdyXdNoI5VNo" />
  </head>
<body>
</body>
</html>

# Frequently asked questions
On this page I answer some frequently asked questions regarding training object detection models using the YOLOv5 architecture. It corresponds with [this Medium post on Towards AI](https://medium.com/towards-artificial-intelligence/train-and-deploy-custom-object-detection-models-without-a-single-line-of-code-a65e58b57b03).

## FAQ 1 - What kind of computer should I use for object detection?
[AddaxAI](https://github.com/PetervanLunteren/AddaxAI) is available for Windows, Mac and Linux systems. Please note that machine learning can ask quite a lot from your computer in terms of processing power. Besides a minimum of 8GB of RAM, there are no hard system requirements since it is largely hardware-agnostic. However, I would recommend at least 16GB of RAM, but preferably 32GB. Although it will run on an old laptop only designed for text editing, it's probably not going to train any accurate models. Faster machines will analyze images quicker and produce more accurate results. The best models are trained on computers with GPU acceleration. AddaxAI will automatically run on NVIDIA and Apple Silicon GPU if you have the appropriate hardware available, see [this thread](https://github.com/PetervanLunteren/AddaxAI#gpu-support) for more info. If you have multiple computers at hand, choose the fastest one. Please note that - at the time of writing (May 2023) - training on Apple Silicon GPU is still in beta version.

## FAQ 2 - What should a good training dataset look like?
It's important to have a well labelled and representative dataset. In computer science there is a concept called GIGO - "garbage in equals garbage out". It means that a model trained on poor quality data will produce poor results. The model can only be as good as the data it learns from. Below are some guidelines for a good dataset provided by the guys at [Ultralytics](https://ultralytics.com/).
* **Images per class** - 1500 or more images per class recommended.
* **Instances per class** - 10000 or more instances (labeled objects) per class recommended.
* **Image variety** - Must be representative of deployed environment. For real-world use cases we recommend images from different times of day, different seasons, different weather, different lighting, different angles, different sources (scraped online, collected locally, different cameras) etc.
* **Label consistency** - All instances of all classes in all images must be labelled. Partial labelling will not work.
* **Label accuracy** - Labels must closely enclose each object. No space should exist between an object and its bounding box. No objects should be missing a label.
* **Background images** - Background images are images with no objects that are added to a dataset to reduce false positives. We recommend about 0–10% background images. No labels are required for background images.
* **Class balance** - Try to balance the classes in terms of number of instances as much as possible. See `labels.jpg` after training starts for the class instance bar graph (top left).

Please note that these are not hard requirements. Don't worry if you don't have thousands of images. You can retrain from an existing model and apply data augmentations when working with limited datasets. See [FAQ 4](https://github.com/PetervanLunteren/AddaxAI/blob/main/FAQ.md#faq-4-which-model-should-i-choose-when-transfer-learning) and [FAQ 10](https://github.com/PetervanLunteren/AddaxAI/blob/main/FAQ.md#faq-10-what-can-i-do-to-avoid-overfitting-and-optimize-mymodel) for more information about that.

## FAQ 3 - How to choose between transfer learning or training from scratch?
In machine learning, it is possible to reuse an existing model as the starting point for a new model. For example, the MegaDetector model is excellent at detecting animals in camera trap images. We can transfer the knowledge of what an animal looks like to a new model so that it doesn't have to be trained again. If your dataset is relatively small (i.e., not tens of thousands of images), it's advised to train your model using transfer learning. Of course, there is also the option to train from scratch, but it is usually not recommended. Only train from scratch if you know what you are doing and have a very large dataset (i.e., around 150.000 images or more).

## FAQ 4 - Which model should I choose when transfer learning?
There are five pre-trained YOLOv5 models which you can use to transfer knowledge from (see image below). These go from small to large and are trained on the [COCO dataset](https://cocodataset.org/#home) consisting of more than 330,000 images of 80 classes. These pre-trained models already _know_ what life on earth looks like, so you don't have to teach your model again. In general, the larger the model, the better the results - but also the more processing power required. The nano model is the smallest and fastest, and is most suitable for mobile solutions and embedded devices. The small model is perfect for a laptop without GPU acceleration. The medium-sized model provides a good balance between speed and accuracy, but you'll probably want a GPU for this. The large model is ideal for detecting small objects, and the extra-large model is the most accurate of them all, but it takes considerable time and processing power to train and deploy. The last two models are recommended for cloud deployments.

If you're training a species classifier for camera trap images, go with either MegaDetector 5a or b. Version a and b differ only in their training data. Each model can outperform the other slightly, depending on your data. Deploy them both on some test images and see which one works best for you. If you really don't have a clue, just stick with the default 'MegaDetector 5a'. More info about MegaDetector models on [its GitHub page](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md).

<p align="center">
  <img src="https://github.com/PetervanLunteren/AddaxAI/blob/main/imgs/yolo_models.png" width=80% height="auto" />
</p>

## FAQ 5 - How does it look when I run out of memory?
It's not uncommon for computers to run out of memory when training. These error messages can be very misleading and can look something like the following. See [FAQ 6](https://github.com/PetervanLunteren/AddaxAI/blob/main/FAQ.md#faq-6-what-should-i-do-when-i-run-out-ofmemory) to find out what to do in these cases.
```
* CUDA out of memory
* Killed
* MemoryError
* Your system has run out of application memory
* DefaultCPUAllocator: not enough memory
* The instruction at XXX referenced memory at XXX. Memory could not be written.
* OSError: [WinError 1455] The paging file is too small for this operation to complete.
* Error loading '...\torch\lib\caffe2_detectron_ops_gpu.dll' or one of its dependencies
* There appear to be 1 leaked semaphore objects to clean up at shutdown
* warnings.warn('resource_tracker: There appear to be %d '
* and probably many more...
```

## FAQ 6 - What should I do when I run out of memory?
When your computer runs out of memory, first close all applications and reboot. Then we'll have to adjust the training parameters to avoid running out of memory again. There are a couple of options to choose from.
* **Increase paging size** - If you are on a Windows computer, setting the paging file (virtual memory) to a fixed size can overcome memory issues. See the steps [here](https://mcci.com/support/guides/how-to-change-the-windows-pagefile-size/) for Windows 10 and [here](https://www.windowscentral.com/software-apps/windows-11/how-to-manage-virtual-memory-on-windows-11) for Windows 11. For the best results and if free storage permits, set the initial size to 1.5 times your RAM capacity and the maximum size to 3 times your RAM capacity. For example, if you have 16 GB of RAM, the initial and maximum sizes should be set to 24.000 MB and 48.000 MB, respectively. Linux and macOS users don't have to worry about this since their memory allocation is arranged differently.
* **Reduce the batch size** - This is the most obvious method of reducing the workload. The batch size is the number of images your computer takes per iteration. The larger the batch size, the more processing power you'll need. If you leave its entry box in [AddaxAI](https://github.com/PetervanLunteren/AddaxAI) empty, it'll automatically check and use the maximum batch size your device can handle. However, this check only works if you have NVIDIA GPU acceleration. If your device doesn't have this, it'll revert to the default batch size of 16 - which might be too large for your computer. So, if you are not training on NVIDIA GPU, try lowering to a batch size of e.g., 4 or even 1 and see if you still run into out-of-memory errors. If not, try increasing it again to find the maximum batch size your hardware allows for.
* **Reduce model size** - Training a model with fewer layers will ask less processing power and thus makes out-of-memory errors less likely. Try lowering model size from `XLarge` > `Large` > `Medium` > `Small` > `Nano`. If you are transfer-learning from a custom model or MegaDetector, you don't have the option to reduce the model size.
* **Reduce the number of workers** - Usually, not all images can be loaded into your computer's RAM at once. Subsets of images are therefore loaded at every iteration. The default of 4 workers should be fine for most computers, but it might help to lower this number to 2 or 1 to avoid out-of-memory errors.
* **Upgrade your hardware** - This option is the most obvious one. Try running it on a faster device or add processing power in terms of GPU acceleration.
* **Reduce the image size** - Larger image sizes usually lead to better results, but take longer to process and require more processing power. An image that is twice as large will have 4 times as many pixels to learn from. Best results are obtained if the same image size is used as the original model you are retraining. Therefore, if you leave the image size entry box empty, AddaxAI will take the image size of the pretrained model you selected (1280 for MegaDetector and 640 for the YOLO models). If you selected a custom model or are training from scratch, the default is 640. You can try to lower the default image size to avoid out-of-memory errors, but I'd advise you to first try the options above - especially if you're trying to detect small objects.

## FAQ 7 - How can I recognize overfitting and when does my model have the best fit?
An overfitted model is a model that performs well on training data but performs poorly on previously unseen data (i.e., validation images). It basically means that your model has memorized the training data instead of learning generic features of the object in question. In other words, the model is memorizing random noise in the training set. Overfitting can be easily recognized in `results.png`, see the example below.

<p align="center">
  <img src="https://github.com/PetervanLunteren/AddaxAI-metadata/blob/main/imgs/FAQ-overfitting.png" width=80% height="auto" />
</p>

The x-axes represent the number of epochs, while the y-axes depict the metrics printed above the graphs. In the `results.png` shown here, you can see that after epoch 50 the validation loss (`val/obj_loss`) increases, whilst the training loss (`train/obj_loss`) further decreases. That is a sign of overfitting, where the model performs well on the training data, but not on the validation data. We would call the model before this turning point underfitted, meaning the model is not sufficiently complex to capture the relationship between features and labels. If you don't see an increase in validation losses in your `results.png`, your model either has a good fit, or is (most likely) still underfit. Overfit your model to get an idea of where the underfitting stops and overfitting starts. The model has the best fit when the validation loss is lowest, thus when the model goes from under- to overfit.

## FAQ 8 - What are all the files in the training folder and how should I interpret them?
If you navigate to your destination folder, you'll see some graphs and files. I'll briefly explain them below.
<p align="center">
  <img src="https://github.com/PetervanLunteren/AddaxAI-metadata/blob/main/imgs/FAQ-files-in-results-folder.png" width=80% height="auto" />
</p>

* `weights/best.pt` - This is your model. `best.pt` will be saved every time a new best fitness is observed on your validation set. Fitness is defined as a weighted combination of metrics, but mostly `mAP_0.5:0.95`.
* `weights/last.pt` - This is the last model saved, but not necessarily the best one. If the training unexpectedly stopped before completing, you use this file to continue training from.
* `results.png` - Here you'll see graphs for different metrics plotted against the number of epochs. The `box`, `obj` and `cls` losses are plotted for both training and validation data and should be as low as possible. The `precision`, `recall` and `mAP` should be as high as possible. Below you can find a short description per metric.
  * `box_loss` - A loss that measures how close the predicted bounding boxes are to the ground truth. _How accurate is the model in predicting bounding boxes?_
  * `obj_loss` - A loss that measures the probability that an object exists in the predicted bounding box. _Is there an object present in the predicted bounding box?_
  * `cls_loss` - A loss that measures the correctness of the classification of each predicted bounding box. _If there is an object present, does the model classify it correctly?_ If you only have one class, there are no class mis-identifications and this value will always be zero.
  * `precision` - A value of how much of the bounding box predictions are correct.
  * `recall` - A value of how much of the true bounding boxes were correctly predicted.
  * `mAP_0.5` - The mean average precision at the intersection over union threshold of 0.5.
  * `mAP_0.5:0.95` - Same as `mAP_0.5`, but then the average over multiple thresholds, ranging from 0.5 to 0.95. This metric is mostly used to determine model fitness.
* `hyp.yaml` - These are the hyperparameters used during training.
* `opt.yaml` - This file contains all the input arguments used for training, including the hyperparameters.
* `results.csv` - These are the same results as in `results.png`, but in tabular format.
* `P_curve.png` - This graph shows the precision per confidence value. The more area under the line, the better.
* `R_curve.png` - Same concept as the `P_curve`, but with recall instead of precision.
* `PR_curve.png` - This is the precision-recall curve and shows the tradeoff between precision and recall. A high area under the curve represents both a high recall and high precision.
* `F1_curve.png` - The F1 score measures the model's accuracy and combines recall and precision scores. This curve shows you the optimum F1 value over the confidence values. The higher the F1 value, the better the model.
* `confusion_matrix.png` - This visualizes the proportions of predicted classes against true classes. Ideally, you would like to see a dark blue diagonal from the top left to the bottom right, meaning the predicted classes are actually the true classes.
* `labels_correlogram.jpg` - A group of histograms showing each axis of your data against each other axis. It shows you what the input images and its bounding box details look like.
* `labels.jpg` - These visualizations provide insight into the shape and location of the bounding boxes, as well as how balanced your training set is.
* `train_batch*.jpg` - These files give you an example of how your computer sees the images and labels when training. Make sure the labels are correctly displayed.
* `val_batch0_labels.jpg` and val_batch0_pred.jpg - Here you can see some example predictions to give you an insight into the model's performance.

## FAQ 9 - What's the general workflow of an object detection training?
It's encouraged to perform multiple trainings to get an insight into your data, model performance, and possible areas of improvement. The first thing you should do when starting a new project is to train a model with default settings. This will act as your performance baseline for all further models. Make sure this baseline model is overfit so that you can establish the point where the model has the best fit and get a good understanding of the dataset (see [FAQ 7](https://github.com/PetervanLunteren/AddaxAI/blob/main/FAQ.md#faq-7-how-can-i-recognize-overfitting-and-when-does-my-model-have-the-bestfit)). Once you have the baseline, you can experiment with optimization techniques (see [FAQ 10](https://github.com/PetervanLunteren/AddaxAI/blob/main/FAQ.md#faq-10-what-can-i-do-to-avoid-overfitting-and-optimize-mymodel)) and compare the new metrics with the baseline model. Finally, when you're satisfied with the results, you'll only have to do one more training to capture the model before it overfits, and you're ready for deployment.

## FAQ 10 - What can I do to avoid overfitting and optimize my model?
It's possible to postpone the point of overfitting in order to train longer and reduce validation losses. The issue with overfitted models is that they are too complex for the data and focus too much on random noise. With the following techniques you can either reduce model complexity or reduce random noise - both of which will result in more accurate models.
* **Increase the dataset** - If you increase the number images, the crucial features to be extracted from the objects become more prominent. That means the model can recognize the relationship between the input attributes and the output variable easier. If the dataset already contains all the images you own, take a look at online databases such as [Google Dataset Search](https://datasetsearch.research.google.com/), [RoboFlow](https://public.roboflow.com/object-detection), or [PapersWithCode](https://paperswithcode.com/datasets?task=object-detection). For ecologists: [LILA BC](https://lila.science/), [Florida Wildlife Camera Trap Dataset](https://lila.science/), and [eMammal](https://emammal.si.edu/).
* **Increase augmentation** - Data augmentation is a method to avoid your model from learning on noise by simply altering the images to express more variety between epochs (see image below). This can consist of position augmentation (scaling, cropping, flipping, etc.) and color augmentation (brightness, contrast, saturation, etc.). The idea is that if you vary the random noise between epochs, the model won't see it as a generic feature. Via [AddaxAI](https://github.com/PetervanLunteren/AddaxAI) this can be done automatically by selecting one of the preloaded hyperparameter configuration files during training. See AddaxAI's 'Help' tab for more information.
* **Evolve hyperparameters** - YOLOv5 has about 30 hyperparameters used for various training settings. These include parameters like learning rates, loss gains, anchors, and augmentation values. You can calculate the optimum values for your project automatically by evolving the default hyperparameters. See AddaxAI's 'Help' tab for more information. Please note that this can cost an enormous amount of time and processing power, since the baseline model is trained multiple times.
* **Reducing model complexity** - An option to avoid overfitting is to reduce the model complexity, i.e., `XLarge` > `Large` > `Medium` > `Small` > `Nano` ([FAQ 4](https://github.com/PetervanLunteren/AddaxAI/edit/main/FAQ.md#faq-4-which-model-should-i-choose-when-transfer-learning)). Please note that - although overfitting might be postponed and training can continue for more epochs - model performance might not improve compared to the more complex model you started with.
* **Increase weight decay** - Weight decay is a hyperparameter that determines the penalty on model complexity. It's a regularization technique to avoid the model from focusing on random noise. The higher the weight decay, the simpler the model. You can do this in AddaxAI by creating a custom hyperparameter configuration file and selecting that for training.

<p align="center">
  <img src="https://github.com/PetervanLunteren/AddaxAI/blob/main/imgs/data_augmentations.jpg" width=80% height="auto" />
</p>
