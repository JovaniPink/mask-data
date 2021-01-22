# Face and Mask Detection Models

> This sub project folder holds our data, data wrangling, and machine learning models for Masky.

## Project

### OVERVIEW

One contentious topic today in social media and playing out in our culture is the wearing of masks. We wanted to tackle this subject as best as possible with researching the facts, data, and cultural perception in regards to the pandemic. We also had a technical question/challenge of developing a machine learning model to categorize mask wearing in images.

We went on to survey the Kaggle Competetions so see how the community was solving similar problems. After looking at past solutions, investigating how Kaggle competitions are solving the problems, and our knowledge of machine learning algorithms - we ran into a distinction between classification and detection. Categorization? Yes, after testing possible machine learning solutions - we reframed the the problem as categorization oppsed to detection.

General Machine Learning Pipeline:

 - take an image
 - detect if a face exists
 - if so then classify that face as mask or no mask.

### MODELS

Greater understanding https://www.quora.com/What-are-the-different-machine-learning-algorithms-for-image-classification-and-why-is-CNN-used-the-most

The distinction between classification and detection matters in the way we approach the problem, classification algorithms:

- Artificial Neural Network
- Convolutional Neural Network
- K Nearest Neighbor (KNN)
- Decision Tree
- Support-vector Machines

But we wanted not only to classify but also detect a face within an image, so we had to narrow down the algorithms to one that would assist us in both classifying and detection. But what is detection?

One class of Deep Learning algorithm stood out, the convolutional neural network (CNN, or ConvNet). We went ahead and started to learn and understand CNN and how to use Tensorflwo Keras with it.


#### Convolutional Neural Network (CNN)

[Keras Conv2D and Convolutional Layers](https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/) was such a jump off point to understanding Convolutional Layers



### DATA POINT AND FEATURES

Stepping back and looking at the problem and understanding the basics of Computer Vision because how do we convert our knowledge and steps of us viewing a digital image and/or video to the computer viewing the digital image and/or video.

Let us take a look at this image….. And where we got our data:
[Prajna Bhandary](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/) on LinkedIn: [Mask classifier](https://github.com/prajnasb/observations)
[Face Mask Detection Dataset](https://www.kaggle.com/omkargurav/face-mask-dataset) | Kaggle

I want to insert one image and go through the complexity of the image and how that will also push us to CNN and similar algorithms. Also I want to go over the features of image and image dataset.

### 📊 WRANGLING THE DATA

EDA

ML Data Wrangling.ipynb

https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923

Image Pre-processing

ML Face Detection.ipynb

### CONVOLUTIONAL NEURAL NETWORK

https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c

Image files of faces with and without face mask.

data folder has four sub folders that house the image data we are using to train the model & using to test the model's predictions.

- processed has all the model images in two categories.
- external has completely septate images to test the finished model against.

### 😀 FACE DETECTION MODEL

OpenCV provides the CascadeClassifier class that can be used to create a cascade classifier for face detection. The constructor can take a filename as an argument that specifies the XML file for a pre-trained model.

OpenCV provides a number of pre-trained models as part of the installation. These are available on your system and are also available on the OpenCV GitHub project.

Download a pre-trained model for frontal face detection from the OpenCV GitHub project and place it in your current working directory with the filename ‘haarcascade_frontalface_default.xml‘.

BUT

MTCNN to the recuse [Multi-task Cascaded Convolutional Networks (MTCNN) for Face Detection and Facial Landmark Alignment](https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923)

### 😷 MASK CLASSIFICATION MODEL

Use a cascade classifier to detect a face in a webcam window, extract the face and pass it to a convolutional neural network to classify the image as "mask" or "no mask".

![Project Overview](resources/ML_Project_Overview.png)

## 🖥 Environment

Included in our repository is our "requirements.txt" With this file in the repository, you can create the new environment by running:

```
python -m venv myenv
pip -r requirements.txt
```

### METHODS USED

- Machine Learning
- Deep Learning
- Convolutional Neural Network (CNN)
- Data Visualization

### TECHNOLOGIES USED

- Python 3
- numpy
- Pandas
- OpenCV
- MTCNN
- Tensorflow/Keras
- matplotlib

## 📑 Todo Checklist

**For a more detailed view of project tasks visit the Projects tab**

A helpful checklist for the things that need to be accomplished:

- [ ] Improve the image pre-processing
- [ ] Add a proper venv and requirements file

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

1. Fork this repository;
2. Create your branch: `git checkout -b my-new-feature`;
3. Commit your changes: `git commit -m 'Add some feature'`;
4. Push to the branch: `git push origin my-new-feature`.

**After your pull request is merged**, you can safely delete your branch.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more information.
