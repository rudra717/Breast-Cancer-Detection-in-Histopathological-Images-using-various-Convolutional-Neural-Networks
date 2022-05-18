# Breast-Cancer-Detection-in-Histopathological-Images-using-various-Convolutional-Neural-Networks

Subject: Deep Learning

Language used: Python

Department of Computer Engineering

Project Members:

Jasmine Batra - jb7854@nyu.edu, Sachin Karthik Kumar - sk9622@nyu.edu, Rudrasinh Nimeshkumar Ravalji - rnr8237@nyu.edu


***Abstract*** 
 
Cancer is one of the world's most frequent and deadly diseases. One in every eight women and one in every eight hundred males is diagnosed with breast cancer. As a result, our first goal should be early cancer detection, as early detection can aid in the effective treatment of cancer. As a result, we present a saliency detection method based on sophisticated deep learning techniques, in which the machine is taught to mimic pathologists' movements for the localisation of diagnostically relevant regions. We use a CNN to train to identify diagnostic types of breast cancer (VGG16, EfficientNet, ResNet architecture). To train our model, we used the BreakHis dataset. In histopathology imaging, we focus on both detecting and classifying malignant areas. The ‘salient' areas are those that are diagnostically important. Pathologists and medical institutions will be able to use the detection technology as an open-source online application.

  
*Introduction*

Cancer is one of the most frequent diseases that a person can contract, and it is also the second leading cause of mortality. Cancer cells divide and spread in the surrounding tissues in any part of the body and in all of its forms. Breast cancer is one of the most serious and prevalent cancers in women, and it can be caused by a variety of factors including lifestyle, screening, and family history. Breast cancer affects one in every eight women and one in every 800 males, according to statistics. Breast exams, mammograms, ultrasounds, collecting a sample of breast tissue (biopsy), and breast magnetic resonance imaging are just a few of the procedures to test and diagnose breast cancer. When compared to high-income countries, the number of cases of breast cancer is increasing by the day in low and middle-income countries. One of the major reasons for this is the high cost of getting yourself diagnosed, as well as the substandard treatment provided in low and middle-income countries. Many of the resources needed for mammography diagnosis are not available in low- and middle-income countries. Due to a lack of resources, the death rate from breast cancer has not decreased with diagnosis from clinical breast examination or self-examination. Even though we live in a time where digitization is prevalent in the field of diagnosis, pathology remains largely reliant on investigations conducted using microscopic examinations of tissues on glass slides. The advancement of machine vision and digitally scanning microscopes has enhanced the possibility of computer-assisted diagnosis in recent years, allowing slides of tissue histology to be recorded as digital images. As a result, image analysis and machine learning approaches have benefited from digital tissue histology.


*Literature Survey*

There hasn't been much research done in this area previously. Though there have been a lot of new approaches recently. This is a new field with a lot of room for advancement. As a result, our effort intends to make a substantial contribution to the discipline. 
Breast Cancer Diagnosis Using Deep Learning Algorithm is one of the few publications that uses the Wisconsin Breast Cancer Database. The database in question has 569 rows and 30 characteristics. For scaled datasets, they used pre-processing methods such as normalizers and label encoders. In 2018, their work was published. Breast Cancer Diagnosis from Histopathological Image based on Deep Learning is another work that leverages the BreakHis dataset. CNNs, or convolutional neural networks, were utilized to identify images as benign or cancerous. However, they employed the Inception V3 model, which has a poor initialization rate. A lot of computation time is wasted and adjusting is costly. 
In the year 2019, such material was published. Nuclear Atypia Grading in Histopathological Images of Breast Cancer is another paper. CNN-based approach was utilized with Convolutional Neural Networks. They graded mitotic cells, nuclear atypia, and tubule development using Nottingham’s technique, which has three elements. Their model is made up of two parts: a feature extraction section with convolutional layers and an activation function section with ReLu and pooling layers. 
 In 2018, the same was published. Another approach that leverages the BreakHis dataset is Histopathological Image Analysis for Breast Cancer Detection Using Cubic SVM.They examined six SVM variations, including support vector machine, random forest technique, and KNN (K Nearest Neighbors), for experimental purposes. They discovered that cubic SVM outperformed all of the other approaches. When the class size is big, however, SVM struggles to predict class labels. In the year 2020, this work was released. 
Another technique that employed BreakHis dataset was A Deep CNN Technique for Detection of Breast Cancer Using Histopathology Images.They also employ CNN, or convolutional neural networks, to differentiate between cancerous and benign images. Breast cancer is the most prevalent cancer diagnosed in women in the United States (excluding skin cancers), accounting for 30% of all new cancer diagnoses in women. Histopathology tries to differentiate between normal tissue, benign (benign) and malignant (carcinomas) abnormalities, as well as perform a prognosis evaluation. 

*Dataset*

There are a variety of histopathological imaging data sets that may be used with deep learning approaches to detect tumor tissue. We used the BreaKHis dataset, which consists of 9109 microscopic images of breast tumor tissue from 82 individuals using various magnification factors (40X,100X,200X,400X). It is divided into two types: benign and malignant. Tumor tissue is missing in benign tumors, but tumor tissue is present in malignant tumors. It shows the structure of the BreaKHis Dataset.  There are now 2,480 benign and 5,429 malignant samples in the database (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format). This database was created in partnership with Parana, Brazil's P&D Laboratory - Pathological Anatomy and Cytopathology.


*Methodology*

This section illustrates how breast images classify by integrating different convolution neural networks. Demonstrates proposed framework used in this classification. This framework starts with extracting images and loading labels from the dataset. After splitting the dataset, several Data augmentation techniques are performed. Finally, we train the model individual on Break-His dataset and evaluate the proposed Framework by doing validation on the test set. Now that the network is trained, we will save the model so we can load it later for making predictions. We will save the mapping of classes to indices which we get from one of the image datasets: image_datasets['train'].class_to_idx. We will attach this to the model as an attribute which makes inference easier later. We even included the checkpoints that will include all the information which we can use it later.  We have used predict function to use trained network for inference that take image and model then return the top K most likely classes along with the probabilities. After This method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes. We need to convert from these indices to actual class labels using class_to_idx which we added to Image Folder. This method takes a path an image and a model checkpoint then return the probabilities and classes. Using matplotlib, we plot the probabilities for the top 5 classes as a bar graph along with input image. The details of blocks related to the proposed framework are given in the following subsections.



Resnet 18

|		Optimizer	  | Learning Rate |		Accuracy	 |
| ------------- | ------------- |	------------ |			
| 		Adam 			|		 	0.01   		|			89.66		 |
| 	 SGD	   	  | 	 0.01   		|			90.465	 |
| 		Adam  		|		 	0.001	|			96.474		 |			
| 		SGD		|		  0.001   		|			96.955		 |

Resnet 50

|		Optimizer	  | Learning Rate |		Accuracy	 |
| ------------- | ------------- |	------------ |			
| 		Adam 			|		 	0.01   		|			96.38		 |
| 	 SGD	   	  | 	 0.01   		|			97	 |
| 		Adam  		|		 	0.001	|			94.792		 |			
| 		SGD		|		  0.001   		|			97.67	 |

Resnet 152

|		Optimizer	  | Learning Rate |		Accuracy	 |
| ------------- | ------------- |	------------ |			
| 		Adam 			|		 	0.01   		|			76.23		 |
| 	 SGD	   	  | 	 0.01   		|			86.45	 |
| 		Adam  		|		 	0.001	|			79.52	 |			
| 		SGD		|		  0.001   		|			87.33	 |

Parameter value used during Training!


|		Resnet Model	  | Parameter |		Value	 |
| ------------- | ------------- |	------------ |			
| 		18 			|		 	Optimizer   		|			SGD		 |
| 		   	  | 	 Learning Rate   		|			0.001	 |
| 		  		|		 	Loss Function		|			Negative Log Likelihood		 |			
| 				|		  Batch Size    		|			32		 |
| 			  | 		Epochs  		|			10		 |
| 		50			|		 	Optimizer   		|			SGD		 |
| 		   	  | 	 Learning Rate   		|			0.001	 |
| 		  		|		 	Loss Function		|			Negative Log Likelihood		 |			
| 				|		  Batch Size    		|			32		 |
| 			  | 		Epochs  		|			10		 |
| 		152 			|		 	Optimizer   		|			ADAM		 |
| 		   	  | 	 Learning Rate   		|			0.001	 |
| 		  		|		 	Loss Function		|			Cross Entropy Loss		 |			
| 				|		  Batch Size    		|			32		 |
| 			  | 		Epochs  		|			52	 |






*IV. Conclusion*

we present a Resnet 18, Resnet34, Resnet 152 framework for more accurately and reliably detecting breast cancer patients from microscope pictures. The suggested framework is based on a feature extraction and transfer learning technique that allows several pre-trained CNN models to extract features independently and then combine them for the classification job. The model has been trained and tested on a variety of image datasets, both small and large. It is built in such a way that it can operate efficiently on a wide range of datasets. The experimental results show that our framework for different Resnet models achieves percent and 98 percent classification accuracy in the BreakHis and respectively, beating both individual CNN pretrained architectures and all other state-of-the-art models discovered in the literature. Furthermore, it performs well in recognizing cancerous pictures, boosting the likelihood of survival. Based on these positive results, we feel that our model would be an excellent option for assisting doctors in quickly diagnosing and detecting breast cancer.



*Steps for Reproducing*

  1. Copy the repository link from GitHub Repository.
  2. Open terminal and Git Clone respository link. You will see the downloaded Resnet18.py, Resnet152.py, and Resnet34SGD.py in the folder.
  3. Download JSON file and Dataset[Output] from the below google link drive through NYU Mail account in the same folder.
  https://drive.google.com/drive/folders/1RbywUYfUEISZuGCyzBQNdS7Au4oyQ7Jr?usp=sharing
  
  4.Run the code in HPC.
