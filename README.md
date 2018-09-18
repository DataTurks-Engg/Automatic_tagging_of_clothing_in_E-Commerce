# **Automatic tagging of clothing in E-Commerce, Using Tensorflow and GCP.**

Classify the clothing products into various categories using Machine Learning.

![](https://cdn-images-1.medium.com/max/1600/1*npSKkU2trCVSIJEeld1EuQ.png)

*****

Shameless plugin: We are a data annotation platform to make it super easy for
you to build ML datasets. Just upload data, invite your team and build datasets
super quick. [Check us out.](https://dataturks.com/index.php?s=blg)

*****

### **Why E-commerce and tagging of clothes?**

From startups to small businesses right through to huge brands, there are a huge
number of companies that can benefit from their own ecommerce website, where
they can sell their own products or services. In today’s competitive and
convenience focused society, no longer do consumers want to venture to the high
street in order to buy items, instead consumers want to shop from their own
homes, making ecommerce a flexible solution for both businesses and buyers.

With E-commerce gaining more popularity day after day the number of products
available for shopping are also increasing. With this increasing trend it is
extremely difficult to tag products like clothes which come in so many varieties
to be tagged manually. So this was a small attempt made to use machine learning
for easing out this task.

### Image classification versus object detection

People often confuse image classification and object detection scenarios. In
general, if you want to classify an image into a certain category, you use image
classification. On the other hand, if you aim to identify the location of
objects in an image, and, for example, count the number of instances of an
object, you can use object detection.

![](https://cdn-images-1.medium.com/max/1600/1*46fE66L6jCtKo1dN14jKYg.png)

### **ML Models:**

The models used were the inbuilt TensorFlow models for object detection
customized for the classification of our data. The categories for the
classification were: Shirts, T-shirts, Jackets, Jeans, Trousers, Sunglasses,
Shoes, Tops, Skirts. There are multiple models available in TensorFlow details
of which can be found at this
[link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

For my purpose, I have used a special class of convolutional neural networks
called MobileNets. MobileNets do not provide as good of an accurate model as
produced by a full-fledged deep neural network. However, the accuracy is
surprisingly very high and good enough for many applications. The main
difference between the MobileNet architecture and a “traditional” CNN’s is
instead of a single 3x3 convolution layer followed by batch norm and ReLU,
MobileNets split the convolution into a 3x3 depthwise conv and a 1x1 pointwise
convolution.

![](https://cdn-images-1.medium.com/max/1600/1*imj3Psc_dqPoMApVpC2gxg.png)

There are multiple sources available online which have tutorials on builing
custom classifiers using MobileNets or and other built-in model.

### **Dataset:**

The images were downloaded from the leading E-commerce websites. This data was
then cleaned(removal of duplicate images, removing unwanted images) and uploaded
to the dataturks platform. Using the annotating tool available at the platform,
these images were annotated with rectangular bounding box (check out the
annotating [tool](https://dataturks.com/projects/Dataturks/Demo Image Bounding
Box Project)) .

![](https://cdn-images-1.medium.com/max/1600/1*7V2Qn-qVSx4_ZUmMiBu1XA.png)

The annotated data was downloaded (into a folder containing images and the JSON
documents) and split into training and test data (80–20 split). Then training
and test folders were given as an input to the python script for converting the
data to Pascal VOC
([demo](https://dataturks.com/help/ibbx_dataturks_to_pascal_voc_format.php)
here) . The xml documents and images obtained as a result from json to Pascal
VOC script as then converted to a set of csv records and images. Finally, this
data was then converted to tensor records for training and testing.

**(Make sure the dataset contains at least 100–150 images per class for training
after splitting the data.)**

Here is the Datatset (open to use): [E-commerce Tagging for
clothing](https://dataturks.com/projects/devika.mishra/E-commerce Tagging for
clothing)

#### [Code](https://github.com/DataTurks-Engg/Automatic_tagging_of_clothing_in_E-Commerce)
for data preparation.

In case of any queries you can contact me at **devika.mishra@dataturks.com.**

### **Training :**

The training of models was done on a Ubuntu 16.04 LTS GPU box having NVIDIA
TESLA K80 and 7GB RAM, 12GB Graphics card along with 30 GB hard disk. The
training was done for 10,000 steps which took 3 hours approximately .

The entire process of training has taken a lot of effort and hard work from
creating a dataset, getting it into the right format, setting up a Google Cloud
Instance(gcp instance), training the model and finally testing it. Given below
is a detailed overview of how the steps after creating the dataset were carried
out.

**Setting up the GCP instance:** Google Cloud Platform is a cloud computing
infrastructure which provides secure, powerful, high-performance and
cost-effective frameworks. It’s not just for data analytics and machine
learning, but that’s for another time. Check it out over
[here](https://cloud.google.com/). Google is giving away $300 dollars of credit
and 12 months as a free tier user.

The step by step guide to how to set up the instance is available on the Google
cloud platform documentation and on multiple blogs (like
[here](https://medium.com/@jamsawamsa/running-a-google-cloud-gpu-for-fast-ai-for-free-5f89c707bae6)).

Once the instance is finally created the set up will look like this:

![](https://cdn-images-1.medium.com/max/1600/1*3ePeCd0ISur0oJnRziu1Mg.png)

**Install TensorFlow GPU on Ubuntu 16.04 LTS:**

    #

    sudo apt-get update

    sudo apt-get — assume-yes upgrade

    sudo apt-get — assume-yes install tmux build-essential gcc g++ make binutils

    sudo apt-get — assume-yes install software-properties-common


    Search for additional drivers in menu and open it. wait for minute and select nvidia driver and hit apply and restart.

    .

    Download cuda-8.0 .deb package and install it

    sudo dpkg -i cuda-repo-ubuntu1604–8–0-local-ga2_8.0.61–1_amd64.deb (this is the deb file you’ve downloaded)

    sudo apt-get update

    sudo apt-get install cuda export PATH=/usr/local/cuda-8.0/bin ${PATH:+:${PATH}}

    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


    tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz

    sudo cp cuda/include/cudnn.h /usr/local/cuda/include

    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*


    sudo apt-get install libcupti-dev


    bash Anaconda3–4.4.0-Linux-x86_64.sh

    pip install tensorflow-gpu==1.2


    python models-master/tutorials/image/imagenet/classify_image.py

    or

    python models-master/tutorials/image/cifar10/cifar10_train.py

### **Modifying the files for your custom model:**

As, I have used the MobileNet model, I have made changes to the configuration
file of the same. The trained models can be downloaded
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

**Changes in config file:**

    model {
      ssd {
       
        box_coder {
          faster_rcnn_box_coder {
            y_scale: 10.0
            x_scale: 10.0
            height_scale: 5.0
            width_scale: 5.0
          }
        }

#### Later section:

    train_config: {
      batch_size: 10
      optimizer {
        rms_prop_optimizer: {
          learning_rate: {
            exponential_decay_learning_rate {
              initial_learning_rate: 0.004
              decay_steps: 800720
              decay_factor: 0.95
            }
          }
          momentum_optimizer_value: 0.9
          decay: 0.9
          epsilon: 1.0
        }
      }
      fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
      from_detection_checkpoint: true
      data_augmentation_options {
        random_horizontal_flip {
        }
      }
      data_augmentation_options {
        ssd_random_crop {
        }
      }
    }
    train_input_reader: {
      tf_record_input_reader {
       
      }
      
    }
    eval_config: {
      num_examples: 40
    }
    eval_input_reader: {
      tf_record_input_reader {
        input_path: 
      }
      label_map_path: 
      shuffle: false
      num_readers: 1
    }

**clothes-detection.pbtxt:**

    item { 
    id: 1
    name: ‘Jackets’
    } 
    item { 
    id: 2
    name: ‘Jeans’
    } 
    item { 
    id: 3
    name: ‘Shirts’
    } 
    item { 
    id: 4
    name: ‘Shoes’
    } 
    item { 
    id: 5
    name: ‘Skirts’
    } 
    item { 
    id: 6
    name: ‘sunglasses’
    } 
    item { 
    id: 7
    name: ‘Tops’
    } 
    item { 
    id: 8
    name: ‘Trousers’
    } 
    item { 
    id: 9
    name: ‘Tshirts’
    }

### Running the Training Job

A local training job can be run with the following command:

    # From the tensorflow/models/research/ directory
    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --train_dir=${PATH_TO_TRAIN_DIR}

where `${PATH_TO_YOUR_PIPELINE_CONFIG}` points to the pipeline config and
`${PATH_TO_TRAIN_DIR}` points to the directory in which training checkpoints and
events will be written to. By default, the training job will run indefinitely
until the user kills it or it completes the number of steps mentioned.

    INFO:tensorflow:global step 11788: loss = 0.6717 (0.398 sec/step)
    INFO:tensorflow:global step 11789: loss = 0.5310 (0.436 sec/step)
    INFO:tensorflow:global step 11790: loss = 0.6614 (0.405 sec/step)
    INFO:tensorflow:global step 11791: loss = 0.7758 (0.460 sec/step)
    INFO:tensorflow:global step 11792: loss = 0.7164 (0.378 sec/step)
    INFO:tensorflow:global step 11793: loss = 0.8096 (0.393 sec/step)

My total loss graphs looks like:

### Running the Evaluation Job

Evaluation is run as a separate job. The eval job will periodically poll the
train directory for new checkpoints and evaluate them on a test dataset. The job
can be run using the following command:

    # From the tensorflow/models/research/ directory
    python object_detection/eval.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
        --eval_dir=${PATH_TO_EVAL_DIR}

where `${PATH_TO_YOUR_PIPELINE_CONFIG}` points to the pipeline config,
`${PATH_TO_TRAIN_DIR}` points to the directory in which training checkpoints
were saved (same as the training job) and `${PATH_TO_EVAL_DIR}` points to the
directory in which evaluation events will be saved. As with the training job,
the eval job run until terminated by default.

### Running TensorBoard

Progress for training and eval jobs can be inspected using Tensorboard. If using
the recommended directory structure, Tensorboard can be run using the following
command:

    tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}


    ssh -i public_ip -L 6006:localhost:6006 


where `${PATH_TO_MODEL_DIRECTORY}` points to the directory that contains the
train and eval directories. Please note it may take Tensorboard a couple minutes
to populate with data.

When successfully loaded the TensorBoard looks like:

Given below are a few images of graphs like the learning rate, batch size,
losses from the tensorboard for my model. All these graphs along with the others
can be found on the tensorboard opened in your browser. Moving the cursor on the
graph gives information like smoothed, step, value etc.

### **Testing** :

In order to test the model locally I exported the files to google drive so that
it becomes easier to test using the object-detection’s
object_detection_tutorial.ipynb file. Below given are the steps to do the same
and easing out the task. Doing so will enable you to run the saved model
multiple times for testing without being charged for the instance and access it
locally at any point of time.

First zip the entire folder where your model checkpoints and graphs are saved.


    cd ~
    wget 


    mv uc\?id\=0B3X9GlR6EmbnWksyTEtCM0VfaFE gdrive

    .

    chmod +x gdrive

    .

    sudo install gdrive /usr/local/bin/gdrive


    gdrive list


    gdrive upload trained.tar.gz

Now, the zipped file can be downloaded from the drive and tested locally.

A few changes to be made to the object_detection_tutorial.ipynb file are
mentioned below:

**In the cell for variables under model preparation**:

    # What model to download.
    MODEL_NAME = ‘ssd_mobilenet_v1_coco_2017_11_17’
    MODEL_FILE = MODEL_NAME + ‘.tar.gz’
    DOWNLOAD_BASE = ‘

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + ‘/
    ’ #name of your inference graph 

    Any model exported using the 
     tool can be loaded here simply by changing 
     to point to a new .pb file.

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(‘data’, 
    ) #name of your pbtxt file. 


#### **In the first cell under detection**:


    PATH_TO_TEST_IMAGES_DIR 
     
    #path to your test images

    TEST_IMAGE_PATHS 
     [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) 
     i 
     range
     ] #i would range from 1 to number of test images+1 


    IMAGE_SIZE 
     

This would complete the testing of your model.

Following are a few results after I tested my model:

### Conclusion:

The results were quite impressive and gave the right classification for around
48 of 50 test images. The confidence level of the classification ranged from 88%
to 98%.

The results seemed convincing for MobileNets models which are said not to be
having a great accuracy. Multiple other models like Faster RCNN or InceptionNets
etc all available on GitHub can also be tried out. These customized models for
image classification can be deployed as Android apps or can be used by the
e-commerce websites.

**I would love to hear any suggestions or queries. Please write to me at
devika.mishra@dataturks.com.**

*****

![](https://cdn-images-1.medium.com/max/1600/1*ouK9XR4xuNWtCes-TIUNAw.png)

*Shameless plugin: We are a data annotation platform to make it super easy for
you to build ML datasets. Just upload data, invite your team and build datasets
super quick. *[Check us out.](https://dataturks.com/index.php?s=blg)

![](https://cdn-images-1.medium.com/max/1600/1*7iEsgGZHGz_GvDtLkr9-Rg.jpeg)

#### [Data Annotations Made Easy](https://dataturks.com/index.php?s=blg)

*****

* [Machine
Learning](https://blog.usejournal.com/tagged/machine-learning?source=post)
* [Artificial
Intelligence](https://blog.usejournal.com/tagged/artificial-intelligence?source=post)
* [Ecommerce](https://blog.usejournal.com/tagged/ecommerce?source=post)
* [Image
Recognition](https://blog.usejournal.com/tagged/image-recognition?source=post)


### [DataTurks: Data Annotations Made Super
Easy](https://blog.usejournal.com/@dataturks)

Data Annotation Platform. Image Bounding, Document Annotation, NLP and Text
Annotations. #HumanInTheLoop #AI, #TrainingData for #MachineLearning.

