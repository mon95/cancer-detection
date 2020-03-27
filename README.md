# Introduction

Cancer is the second leading cause of death globally and was responsible for an estimated 9.6 million deaths in 2018. Lung cancer is the leading cause of cancer death in the United States with an estimated 160,000 deaths in the past year[[1]](#ref1). Early detection of cancer, therefore, plays a key role in its treatment, in turn improving long-term survival rates.

There are several barriers to the early detection of cancer, such as a global shortage of radiologists. In addition to the shortage, detecting malignant tumors in X-rays can be difficult and challenging even for experienced radiologists. This time-consuming process typically leads to fatigue-based diagnostic errors and discrepancies[[2]](#ref2).

Our project focuses on detecting the presence of malignant tumors in chest X-rays. In order to aid radiologists around the world, we propose to exploit supervised and unsupervised Machine Learning algorithms for lung cancer detection. We aim to showcase ‘explainable’ models [[3]](#ref3) that could perform close to human accuracy levels for cancer-detection. We envision our models being used to assist radiologists and scaling cancer detection to overcome the lack of diagnostic bandwidth in this domain. We can also potentially export our models to personal devices, which would allow for easier, cheaper and more accessible cancer detection. 

__________

# Dataset

We used the CheXpert Chest radiograph datase[[4]](#ref4) to build our initial dataset of images. To build our dataset, we sampled data corresponding to the presence of a ‘lung lesion’ which was a label derived from either the presence of “nodule” or “mass” (the two specific indicators of lung cancer). 

The initial (unaugmented) dataset:

#### Train:
Benign images (Negative class): 6488 images <br>
Malignant (Positive class): 6287 images

#### Validation:
Benign images (Negative class): 1500 images <br>
Malignant (Positive class): 1450 images

#### Test:
Benign (Negative class): 1500 images <br>
Malignant (Positive class): 1449 images


In the training phase, we treated all images with transformations to augment our data by performing random resized crop and lateral inversions with a 50% probability. 

In addition to the above all images were normalized using the channel-wise mean and standard deviation values computed on the ImageNet dataset. 


____

# Supervised Learning

## Approach: 
We use a transfer learning approach to perform supervised binary classification of images as 'benign' or 'malignant' based on the presence of malignant tumors. 

## Reason behind adopting this approach:

It is known that tumors are of different shapes and sizes and can occur at different locations, which makes their detection challenging[[5]](#ref5). In addition to this, deep learning approaches have been showing expert-level performance in medical image interpretation tasks in the recent past (for eg., Diabetic Retinopathy[[6]](#ref6)). This can be attributed to both - availability of large labeled data sets and the ability of deep neural networks to extract complex features from within the image. 

It would be tedious (and maybe near impossible) to hand-design the features that one would need to build models for this task. This, in combination with the fact that we were dealing with a dataset containing a significantly smaller amount of images directly points to using a transfer learning approach where we initialize the parameters from a network pre-trained on ImageNet data and modify the final fully connected layer of the pre-trained network to a new fully-connected layer producing 2 responses indicative of the predicted probabilities of the two classes.


## What’s new in our approach:

1. The overall architecture of feature_extraction + grad_cam visualization + data augmentation via VAEs is new and has not been approached on a medical image dataset to the best of our knowledge.
2. If our approach can show improved results, it could mean that we do not necessarily have to collect a large amount of data at all times and would be able to manage with smaller datasets. 


### Proposed System Architecture

![](./images/image7.jpg)

<p align="center" style="font-size:16px">
Fig 1. System Architecture: The classifier is trained on the training dataset and the generated data from the Variational AutoEncoders. The model classifies a test X-ray as benign or malignant and highlights the region that contributes most to the classification. 
</p>

_____

# Grad CAM

Among the most important areas of research in deep learning today is that of interpretability, i.e, being able to demystify the black-box nature (owing to its non-convex nature) of a neural network and identify the key reasons for making its predictions. Various approaches have been proposed to help with this exercise, the most recent of which involves gradient-based class activation mappings that highlight the specific pixels (or regions) of an image that most strongly activate a certain class of the model’s prediction. Abbreviated as Grad-CAM, this approach has become a universally accepted yardstick for interpretability in the deep learning research community across a wide range of tasks such as image classification, object detection, image captioning and visual question answering. 

This becomes a particularly relevant addition to a medical diagnostic tool considering the serious implications of algorithmic decision making in this domain. It is essential to build trust in the algorithms among doctors and patients alike. Critically, it also sheds light on the imperfections of a trained model when it makes incorrect predictions or when it makes the right predictions for the wrong reasons. While somewhat intellectually dissatisfying, it shouldn’t surprise us that these cases are plenty in number because the training paradigm in deep learning problems simply maps input data to output labels, with no scope for detailed reasoning on the causal relationships behind this mapping. These heatmaps generated from Grad-CAM are an important step in the direction of making our model’s predictions more trustworthy, and in this domain, would aid a radiologist in examining potentially relevant areas in the X-ray images that are likely to be of diagnostic importance.

We demonstrate a few applications of Grad-CAM to our problem and showcase its usefulness (and occasional unreliability) in the following examples.

<p align="center">
  <img src="./images/image28.png" /> <img src="./images/image11.png" />
</p>

<p align="center" style="font-size:16px">
Fig 2. An example of the working mechanism of Grad-CAM. This is an image classification task where a deep neural network has predicted the left image to correspond to the ‘elephant’ class, while the right image highlights the precise region of the image that most strongly activated the ‘elephant’ class.
</p>

<p align="center">
  <img src="./images/image16.png" /> <img src="./images/image10.png" />
</p>

<p align="center" style="font-size:16px">
Fig 3. On the left is the original lateral chest X-ray image that has been correctly classified as malignant. On the right is the Grad-CAM heatmap that points to the precise region in the X-ray where there’s a clumping of cells that explains the prediction of malignancy. 
</p>

<p align="center">
  <img src="./images/image3.png" /> <img src="./images/image20.png" />
</p>

<p align="center" style="font-size:16px">
Fig 4. On the left is the original lateral chest X-ray image that has been correctly classified as malignant. On the right is the Grad-CAM heatmap that points to the precise region in the X-ray where a radiologist ought to be looking at for cues on potential malignancy.
</p>

<p align="center">
  <img src="./images/image19.png" /> <img src="./images/image24.png" />
</p>

<p align="center" style="font-size:16px">
Fig 5. Grad-CAM is also useful to remind ourselves that when deep neural networks get predictions right, they don’t always do so for the correct underlying reasons. Here on the left is the original frontal chest X-ray that has been correctly classified as malignant. On the right is the Grad-CAM heatmap that points to seemingly irrelevant regions in the X-ray that are unrelated to potential malignancy.
</p>

<p align="center">
  <img src="./images/image30.png" /> <img src="./images/image4.png" />
</p>

<p align="center" style="font-size:16px">
Fig 6. This example points to yet another fallibility of deep neural networks that Grad-CAM brings to light. While the original frontal chest X-ray on the left has been correctly classified as malignant, we see in the heatmap on the right that there are multiple regions of interest, one of which may be the appropriate region of malignancy. Domain experts could use these maps as cues for further manual investigation.
</p>

________

# Unsupervised Learning

In the clinical setting, it becomes extremely important to train a model that can handle the entire range of variations in the patient’s X-ray scan. However, it  becomes nearly impossible to obtain all possible variations of input. In addition to this, one of the biggest challenges in the medical field is the lack of sufficient image data, which are laborious and costly to obtain. With data privacy being especially important in the medical domain, it is difficult to obtain the sufficient amount of data that is required for building robust models. Therefore, data augmentation emerges as an essential technique that could be leveraged to increase the variability of the dataset, thus reducing the risk of overfitting. Conventional transformation methods (eg: flip, rotation) can be used to augment our training corpus, but their outputs are highly dependent on the original data. Hence, we propose to make use of an unsupervised technique of generating new samples having similar properties as that of the training dataset. 

A Variational Autoencoder (VAE) is one such deep generative model that estimates the probability density function of the dataset. VAE is an architecture which comprises of an encoder and a decoder, and is trained to minimise the reconstruction error between the encoded-decoded data and the initial data. The encoder projects each input datapoint onto a latent space that follows a normal distribution. Thus it converts the input into a d-dimensional latent vector that can be sampled with mean  and standard deviation  through reparametrization. The decoder then decodes these latent representations and reconstructs the input data. The loss function of the variational autoencoder is the sum of the reconstruction loss and the regularizer.


![](./images/image5.png)
<p align="center" style="font-size:10px;">
Eq 1. Loss function of a Variational Autoencoder
</p>

The first term is the reconstruction loss, or the expected negative log-likelihood of the i-th datapoint. The expectation is taken with respect to the encoder’s distribution over the representations. This term encourages the decoder to learn to reconstruct the data. The second term is a regularizer which in our case is the Kullback-Leibler divergence between the encoder’s distribution and the standard Gaussian distribution.

We designed a deep VAE having the architecture described in Figure 7 and sampled a thousand images for each category ( benign and malignant ). These sampled images were added to the training dataset and our supervised models were re-tested on the newly augmented training corpus.


![](./images/VAE2.png)
<p align="center" style="font-size:10px;">
Fig 7. Architecture of CNN based Variational AutoEncoder.
</p>

<br>
<br>

<p align="center">
  <img src="./images/VAE_images.png" width="500" height="500">
</p>
<p align="center" style="font-size:10px;">
Fig 8. Images sampled from VAE. The top two belong to the benign and bottom two are malignant X-ray scans.
</p>

_____

# Evaluation

<table class="rich-diff-level-zero"> <tbody class="rich-diff-level-one">
<tr>
  <th></th>    
  <th colspan="2"> AlexNet </th>
  <th colspan="2"> DenseNet </th>
</tr><tr style="text-align:center"> 
  <th  style="width: 150px;">Metric</th>
  <th style="width: 150px;">Initial</th>
  <th  style="width: 150px;">Augmented</th>
<th style="width: 150px;"> Initial</th>
<th style="width: 150px;"> Augmented </th> </tr> 
<tr style="text-align:center"> 
<th>Accuracy</th>
<td>68.83</td>
<td>71.48</td>
<td>70.53</td>
<td>73.11</td>
</tr>
<tr style="text-align:center"> 
<th>Precision</th>
<td style="
">65.30</td>
<td>69.31</td>
<td>70.50</td>
<td>70.12</td>
</tr>
<tr style="text-align:center"> 
<th>Recall</th>
<td>78.05</td>
<td>75.29</td>
<td>68.80</td>
<td>78.89</td>
</tr>
<tr style="text-align:center"> 
<th style="
    align: center;
">F1-Score</th>
<td>71.10</td>
<td>72.18</td>
<td>69.65</td>
<td>74.24</td>
</tr>      </tbody> </table>

<p align="center">
Table 1: Summary of results obtained in the supervised binary classification task using two different network architectures. Initial refers to the results obtained by training the network on the initial dataset and augmented refers to the results obtained by training the network on the augmented dataset. The results show a marked improvement in accuracy and recall post augmentation on both network architectures without a significant reduction in precision.
</p>


# Results

  
![](./images/image14.png)         |  ![](./images/image18.png)
:-------------------------:|:-------------------------:
The validation accuracy of AlexNet over different epochs for models trained with initial data and augmented data. |  The test accuracy of AlexNet over different epochs for models trained with only the initial data and augmented data. 

<p align="center" style="font-size:16px;">
The model trained with the augmented data outperforms the model trained with the only the initial data. 
</p>



![](./images/Confusion_matrix_Alexnet_initial_data.png)         |  ![](./images/Confusion_matrix_Alexnet_augmented_data.png)
:-------------------------:|:-------------------------:
Confusion matrix of the AlexNet model trained using the initial data  |  Confusion matrix of the AlexNet model trained using VAE augmented data. 

<p align="center" style="font-size:16px;">
The alexnet model trained with the augmented data outperforms the model trained with only the initial data.
</p>



![](./images/Confusion_matrix_densenet_initial_data.png)         |  ![](./images/Confusion_matrix_densenet_augmented_data.png)
:-------------------------:|:-------------------------:
Confusion matrix of the DenseNet model trained using the initial data  |  Confusion matrix of the DenseNet model trained using the VAE augmented data.


<p align="center" style="font-size:16px;">
 The densenet model trained with the augmented data outperforms the model trained with only the initial data.
</p>

### Performance Metrics for both networks: AlexNet and DenseNet

<!-- ![](./images/image21.png) -->
<p align="center" style="font-size:16px;">
  <img src = "./images/image21.png" /><br>
Accuracy
</p>


<!-- ![](./images/image15.png) -->
<p align="center" style="font-size:16px;">
  <img src = "./images/image15.png" /><br>
Recall
</p>


<!-- ![](./images/image8.png) -->
<p align="center" style="font-size:16px;">
  <img src = "./images/image8.png" /><br>
Precision
</p>


<!-- ![](./images/image27.png) -->
<p align="center" style="font-size:16px;">
  <img src = "./images/image27.png" /><br>
F1 Score
</p>

_____________


# Conclusion

In this project, we developed a machine learning solution to address the requirement of clinical diagnostic support in oncology by building supervised and unsupervised algorithms for cancer detection. From a supervised learning perspective, we demonstrated the effectiveness of transfer learning by using pre-trained convolutional classifiers and fine-tuning them to achieve reasonably good results in our complex domain. We showcased ‘explainable’ models that could reason about their predictions and reduce ambiguity. We also presented a way to overcome inherent data accessibility limitations in the medical field and avoid overfitting by implementing a data augmentation technique using variational autoencoders, resulting in a clear increase in accuracy, thus tightly entangling the supervised and unsupervised components of our solution. We view this as a comprehensive solution that tackles the multiple challenges of data limitations, interpretability and accuracy that are integral to algorithmic successes in the medical domain, and foresee strong potential for its widespread deployment in production, especially on embedded devices equipped with cameras that could provide instant assistance to radiologists around the world.


_______

# References

<a name="ref1"></a> 1. Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T. et.al., M. P. (2017). Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning. <br><br>
<a name="ref2"></a> 2. Yongsik Sim,  Myung Jin Chung et al. Deep Convolutional Neural Network–based Software Improves Radiologist Detection of Malignant Lung Nodules on Chest Radiographs, Radiology, 2019. <br><br>
<a name="ref3"></a> 3. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, 2017, pp. 618-626. <br><br>
<a name="ref4"></a> 4. Kingma P, Welling M., An Introduction to Variational Autoencoders, arXiv:1906.02691. <br>
<a name="ref5"></a> 5. Ardila, D., Kiraly, A.P., Bharadwaj, S. et al. End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography. Nat Med 25, 954–961 (2019). <br><br>
<a name="ref6"></a> 6. Gulshan V, Peng L, Coram M, et al. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. JAMA. 2016. <br><br>
<a name="ref7"></a> 7. Irvin, Jeremy & Rajpurkar, Pranav & Ko, Michael & Yu, Yifan & Ciurea-Ilcus, Silviana & Chute, Chris & Marklund, Henrik & Haghgoo, Behzad & Ball, Robyn & Shpanskaya, Katie & Seekins, Jayne & Mong, David & Halabi, Safwan & Sandberg, Jesse & Jones, Ricky & Larson, David & Langlotz, Curtis & Patel, Bhavik & Lungren, Matthew & Ng, Andrew. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. <br><br>
<a name="ref8"></a> 8. Xianxu Hou, Linlin Shen, Ke Sun, and Guoping Qiu. Deep feature consistent variational autoencoder. In Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on, pp. 1133–1141. IEEE, 2017. <br><br>
<a name="ref9"></a> 9. Javier Jorge, Jesús Vieco, Roberto Paredes, Joan-Andreu Sánchez, and José-Miguel Benedí. Empirical evalua- tion of variational autoencoders for data augmentation. <br><br>


____________

# Contributors


1. Conceptualization of the project's architecture and details : Equal contribution from all. 
2. Literature survey : Equal contribution from all. 
3. Supervised learning : Shalini and Sreehari. 
4. Unsupervised learning : Shreya and Arvind. 
5. Grad-CAM : Sameer.
6. Report writing and editing : Equal contribution from all.
7. GitHub page maintained by : Shalini.

