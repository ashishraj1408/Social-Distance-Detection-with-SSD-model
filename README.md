# Social-Distance-Detection-with-SSD-model
It an application which will help people to maintain social distancing in public places by detecting through Live feed of camera.

# Aim and Objectives
# Aim:
Social distancing aims to decrease or interrupt transmission of COVID-19 in a population by minimizing contact between potentially infected individuals and healthy individuals, or between population groups with high rates of transmission and population groups with no or low levels of transmission.

# Objective:
The main objective of the project is to create a program which can help in detecting Social Distance at crowded places with Live Camera feed for detection.

# Abstract:

➢This project uses Deep Learning based SSD(Single Shot Detection) Pretrained model for object Detection, OpenCV python library for image processing and Centroid Tracking Algorithm For object tracking. In this project, I am attaching the code for building a Social Distancing Detector to detect if a crowd is practicing Social Distancing or not, using a sample video.

➢Social Distancing is one such terminology that has gained popularity over the past few months, thanks to COVID-19. People are forced to maintain a sufficient amount of distance between each other to prevent the spread of this deadly virus. Amidst this crisis, I and My Project Team decided to build a simple Social Distancing Detector that could monitor the practice of social distancing in a crowd.

# Introduction:
➢The spread of COVID- 19 Pandemic complaint has created a most pivotal global health extremity of the world that has had a deep impact on humanity and the way we perceive our world and our everyday lives. In December 2019 the spread of severe acute respiratory pattern coronavirus 2 ( SARS- CoV- 2), a new severe contagious respiratory complaint surfaced in Wuhan, China and has infected 7,711 people and 170 reported deaths in China before coronavirus was declared as a global epidemic, was named by the World Health Organization as COVID- 19 ( coronavirus complaint 2019).
➢Vision- grounded object discovery and shadowing technology can be used to help covering social distancing, mollifying the spread of the complaint. With the development of artificial intelligence technology, vision- grounded object discovery and shadowing technology has gradationally entered into numerous aspects of people lives, similar as videotape surveillance, mortal- computer commerce, actions understanding, using deep neural networks numerous deep neural networks have demonstrated strong robustness in the field of object discovery by autonomously learning object features. 
➢The current real- time image discovery models grounded on deep literacy substantially use algorithm frame- workshop similar as SSD and YOLO. SSD was proposed by Wei Liu in 2016. It has presto recognition speed and high delicacy, and is suitable for multi-target recognition field. This composition attempts to expand its functions for Social distancing monitoring in the fight against the Covid- 19 and the delicacy attained was between 85 and 95.
# SSD:
➢SSD has two components: a backbone model and SSD head. Backbone model usually is a pre-trained image classification network as a feature extractor. This is typically a network like ResNet trained on ImageNet from which the final fully connected classification layer has been removed. 
➢We are thus left with a deep neural network that is able to extract semantic meaning from the input image while preserving the spatial structure of the image albeit at a lower resolution.
 For ResNet34, the backbone results in a 256 7x7 feature maps for an input image. ➢We will explain what feature and feature map are later on. The SSD head is just one or more convolutional layers added to this backbone and the outputs are interpreted as the bounding boxes and classes of objects in the spatial location of the final layers activations.
 
 # Methodology:
 
 <img width="617" alt="image" src="https://user-images.githubusercontent.com/88026146/232330294-e2dacc42-4693-4c31-8b8c-325e98db67af.png">
 
1) Data collection and pre-processing
2) Model development and training
3) Model testing
4) Model implementation

1. Data Collection and Pre-processing
The proposed system uses the existing background subtraction algorithm in a pre-processing step. The real-time automated detection of social distance maintenance or not which is performed by the SSD algorithm 

2. Model building and Training
Our proposed framework uses the transfer learning approach [20] and will fine-tune the MobileNetV2 model, which is a highly efficient architecture that can be applied to edge devices with limited computing power, such as raspberry pi4 to detect people in real time. We used 80% of our total custom data set to train our model with a single shot detector, which takes only one shot to detect multiple objects that are present in an image using multibox. The custom data set is loaded into the project directory and the algorithm is trained on the basis of the labeled images. In pre-processing steps, the image is resized to 224×224 pixels, converted to numpy array format and the corresponding labels are added to the images in the dataset before using our SSD model as input to build our custom model with MobileNetV2 as the backbone and train our model using the Object Detection API.
After downloading the pre-trained weights and creating a new fully-connected (FC) head, the SSD algorithm is trained with both the pre-trained ImageNet weights and the annotated images in the custom data set by tuning the head layer weights without updating weights of base layers. We trained our model for 1000 steps using the Adam optimizing algorithm, the learning decay rate for updating network weights, and the binary cross-entropy for mask type classification.
Parameters were initialized for the initial learning rate of INIT_LR = 1e-4, number of epoch EPOCHS = 20 and batch size BS = 32. We used webcam for social distance monitoring using cv2 and after a person has been identified, we start with bounding box coordinates and computing the midpoint between the top-left and the bottom-left along with the top-right and bottom-right points. We measure the Euclidean distance between the points in order to determine the distance between the people in the frame.

3. Model Testing
The proposed system operates in an automated way and helps to automatically perform the social distance inspection process. Once the model is trained with the custom data set and the pre-trained weights given, we check the accuracy of the model on the test dataset by showing the bounding box with the name of the tag and the confidence score at the top of the box. The proposed model first detects all persons in the range of cameras and shows a green bounding box around each person who is far from each other after that model conducts a test on the identification of social distances maintained in a public place, if persons breaching social distance norms bounding box color changes to red for those persons and if the social distance is not preserved, the system generates a warning and send alert to monitoring authorities with face image. The system detects the social distancing with a precision score of 91.7% with confidence score 0.7, precision value 0.91 and the recall value 0.91 with FPS = 28.07.


4. Model Implementation
The proposed system uses SSD with camera to automatically track public spaces in real-time to prevent the spread of Covid-19, and the software is attached camera. The camera feeds real-time videos of public places to the model, which continuously and automatically monitors public places and detects whether people keep safe social distances.
Our solution operates is, when the detection of a social distance violation by individuals is detected continuously in threshold time, there will be an alarm that instructs people to maintain social distance and a critical alert is sent to the control center of the State Police Headquarters for further action.
 
 
 # Calculating Distance:
 ➢First, detect the people in the image using the SSD model. This will give you the bounding boxes around each person.
➢Calculate the centroid of each bounding box. The centroid is the point at the center of the box.
➢Calculate the distance between each pair of centroids. You can use the Euclidean distance formula to calculate the distance between two points:
distance = sqrt((x2-x1)^2 + (y2-y1)^2).
➢where (x1, y1) and (x2, y2) are the coordinates of the two centroids.
Compare the distance between each pair of centroids to a threshold value. If the distance is less than the threshold value, then the people are too close to each other and violating social distancing guidelines.

# Demo:
<img width="369" alt="image" src="https://user-images.githubusercontent.com/88026146/232330429-2758c00d-e289-46da-877d-a58886525c82.png">
<img width="337" alt="image" src="https://user-images.githubusercontent.com/88026146/232330458-78314fa7-55bf-4cfc-967a-42028247b64f.png">

# Advantages:
➨It delays peak of epidemic and consecutively provide time to health authorities to slow down influenza transmission. Moreover it provides time to arrange for basic medical amenities such as Forehead Thermometer, Infrared Thermometer, FFP2 Mask, FFP3 Mask, Face Mask, Surgical Face Mask, KN95, N95, Protective Mask, beds, ventilators etc.
➨It reduces number of infections in the people.
➨It spreads number of infectious people over a longer period of time.
➨Social distancing reduces the rate of disease transmission and can stop an outbreak.
➨It is most effective technique when infection is being transmitted due to droplet contact such as coughing or sneezing.

# Future Scope:
The above-mentioned use cases are only some of the numerous features that were incorporated as part of this result. We assume there are several other cases of operation that can be included in this result to offer a more detailed sense of safety. Several of the presently under development features are listed below in brief:
1)  Coughing and Sneezing Detection:
 Chronic coughing and sneezing is one of the crucial symptoms of COVID- 19 infection as per WHO guidelines and also one of the major route of complaint spread to non-infected public. Deep literacy grounded approach can be proved handy then to discover & limit the  complaint spread by enhancing our  offered  result with body gesture analysis to  understand if an  existent is coughing and sneezing in public places while  violating facial mask and social distancing  guidelines and grounded on  outgrowth enforcement agencies can be advised.  
2)Temperature Screening:
 Elevated body temperature is another  crucial symptom of COVID- 19 infection, at present  script  thermal screening is done using handheld contactless IR thermometers where health worker need to come  by close  nearness  with the person need to be screened which makes the health workers vulnerable to get infected and also its  fair insolvable to capture temperature for each and every person in public places, the proposed use- case can be equipped with  thermal cameras grounded screening to analyze body temperature of the peoples in public places that can add another helping hand  to enforcement agencies to attack the pandemic effectively.

3)Alert System:
Alert System will have alarms whenever, social distancing is not followed the alarm system can ring the alert bell which  can help them to create distancing  to break the spread of the contagion.


# Coclusion:
➢Our work distinguishes the social distancing pattern and classifies them as a violation of social distancing or maintaining the social distancing norm. also, it also displays markers as per the object discovery.
 The classifier was also enforced for live videotape aqueducts and images also. This system can be used in CCTV for surveillance of people during afflictions. 
➢Mass screening is possible and hence can be used in crowded places like road stations, machine stops, requests, thoroughfares, boardwalk entrances,  seminaries,  sodalities, etc. 
➢By covering the distance between two individualities, we can make sure that an existent is maintaining social distancing in the right way which will enable us to check the contagion.
➢Social distancing detection is an important tool for preventing the spread of infectious diseases, such as COVID-19. By monitoring the distance between people in public spaces, we can help ensure that individuals maintain a safe distance from each other and reduce the risk of transmission.
➢Overall, social distancing detection is a critical component of public health measures to prevent the spread of infectious diseases. It can help to promote safe behavior and reduce the risk of transmission in public spaces, such as offices, schools, and shopping centers.

# Refrences:
1.	"Social Distancing Detection Using Computer Vision" by J. Ruiz-Ascencio et al., published in the IEEE Access journal in 2020.
2.	"Automated Social Distance Monitoring to Prevent COVID-19 Spread" by H. Hassanpour et al., published in the Journal of Medical Systems in 2020.
3.	"Social Distance Monitoring System Using Deep Learning" by T. Kim et al., published in the Journal of Healthcare Engineering in 2021.
4.	"Social Distance Monitoring Using Computer Vision and Deep Learning: A Review" by M. B. Khan et al., published in the IEEE Access journal in 2021.
5.	"Real-time Social Distancing Detection in COVID-19 Pandemic Using Computer Vision Techniques" by R. Poudel et al., published in the International Journal of Advanced Computer Science and Applications in 2020.
6.	Sneha Madane, Dnyanoba Chitre, Social Distancing Detection and Analysis through
a.	Computer Vision Department of Computer Engineering Terna College of Engineering,
b.	Nerul, Navi Mumbai
7.	 Afiq Harith Ahamad, Norliza Zaini, Mohd Fuad Abdul Latip Faculty of Electrical
a.	Engineering, Person Detection for SocialDistancing and Safety Violation Alert based on
b.	Segmented ROI Universiti Teknologi MARA (UiTM) Shah Alam Selangor, Malaysia
8.	Abdalla Gad* , Gasm ElBary* , Mohammad Alkhedher§ , Mohammed Ghazal* , SMIEEE  Vision-based Approach for Automated Social Distance ViolatorsDetection Department of Electrical and Computer Engineering* Department of Mechanical Engineering§ Abu Dhabi University.
9.	A. Bharade, S. Gaopande, and A. G. Keskar,“Statistical approach for distance estimation using inverse perspective mapping on embedded platform,” in 2014 Annual IEEE India Conference (INDICON), 2014
10.	S. Tuohy, D. O’Cualain, E. Jones, and M. Glavin, “Distance determination for an automobile environment using inverse perspective mapping in opencv,” in IET
11.	Neelavathy Pari, S.; Vasu, B.; Geetha, A.V. Monitoring Social Distancing by SmartPhone App in the effect of COVID-19. Glob. J.Computer Sci. Technol. 2020, 9, 946–953.
12.	Kobayashi, Y.; Taniguchi, Y.; Ochi, Y.; Iguchi, N. A System for Monitoring Social Distancing Using Microcomputer Modules on University Campuses. In Proceedings of the 2020 IEEE International Conference on Consumer Electronics Asia (ICCE-Asia), Busan, Korea, 1–3 November 2020; pp. 1–4.
13.	Munir, M.S.; Abedin, S.F.; Hong, C.S. Arisk-sensitive social distance recommendation system via Bluetooth to- wards the COVID-19 private safety. Proc. Natl. Inst. Inf. Sci.
14.	.David Oro ; Carles Fernández ; Javier Rodríguez Saeta ; Xavier Martorell ; Javier Hernando. Real-time GPU-based face detection in HD video streams. IEEE Int. Conf. Comp Vis 2011 
15.	Glass RJ, Glass LM, Beyeler WE, Min HJ. Targeted social distancing architecture for pandemic influenza. Emerging Infectious Diseases. 2006; 12:1671– 1681. 
16.	S. J. Pan and Q. Yang, "A Survey of Transfer Learning," in IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345- 1359, Oct. 2010. 
17.	S. S. Mohamed, N. M. Tahir and R. Adnan, "Background modelling and background subtraction efficiency for object detection," 2010 6th International Colloquium on Signal Processing & its Applications, Mallaca City, 2010, pp. 1-6, doi: 10.1109/CSPA.2010.5545291. 
18.	M. Piccardi, "Background subtraction techniques: a analysis," 2004 IEEE International Conference on Systems, Man and Cybernetics (IEEE Cat. No.04CH37583), The Hague, 2004, pp. 3099-3104 vol.4, doi: 10.1109/ICSMC.2004.1400815. 
19.	S. Ge, J. Li, Q. Ye and Z. Luo, "Detection of Masked Faces in the Wild with LLE-CNNs," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 426-434.
