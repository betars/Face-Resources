# Face-Resources
Following is a growing list of some of the materials I found on the web for research on face recognition algorithm.

## Papers

1. [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf).A work from Facebook.
2. [FaceNet](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf).A work from Google.
3. [ One Millisecond Face Alignment with an Ensemble of Regression Trees](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf). Dlib implements the algorithm.
4. [DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)
5. [DeepID2](http://arxiv.org/abs/1406.4773)
6. [DeepID3](http://arxiv.org/abs/1502.00873)
7. [Learning Face Representation from Scratch](http://arxiv.org/abs/1411.7923)
8. [Face Search at Scale: 80 Million Gallery](http://arxiv.org/abs/1507.07242)
9. [A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf)

10. [NormFace: L2 Hypersphere Embedding for Face Verification](https://arxiv.org/abs/1704.06369).* attention: model released !*
11. [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

## Datasets

1. [CASIA WebFace Database](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). 10,575 subjects and 494,414 images
2. [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).13,000 images and 5749 subjects
3. [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/) 202,599 images and 10,177 subjects. 5 landmark locations, 40 binary attributes.
4. [MSRA-CFW](http://research.microsoft.com/en-us/projects/msra-cfw/). 202,792 images and 1,583 subjects.
5. [MegaFace Dataset](http://megaface.cs.washington.edu/) 1 Million Faces for Recognition at Scale
690,572 unique people
6. [FaceScrub](http://vintage.winklerbros.net/facescrub.html). A Dataset With Over 100,000 Face Images of 530 People.
7. [FDDB](http://vis-www.cs.umass.edu/fddb/).Face Detection and Data Set Benchmark. 5k images.
8. [AFLW](https://lrs.icg.tugraz.at/research/aflw/).Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. 25k images.
9. [AFW](http://www.ics.uci.edu/~xzhu/face/). Annotated Faces in the Wild. ~1k images.
10.[3D Mask Attack Dataset](https://www.idiap.ch/dataset/3dmad). 76500 frames of 17 persons using Kinect RGBD with eye positions (Sebastien Marcel)
11. [Audio-visual database for face and speaker recognition](https://www.idiap.ch/dataset/mobio).Mobile Biometry MOBIO http://www.mobioproject.org/
12. [BANCA face and voice database](http://www.ee.surrey.ac.uk/CVSSP/banca/). Univ of Surrey
13. [Binghampton Univ 3D static and dynamic facial expression database](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html). (Lijun Yin, Peter Gerhardstein and teammates)
14. [The BioID Face Database](https://www.bioid.com/About/BioID-Face-Database). BioID group
15. [Biwi 3D Audiovisual Corpus of Affective Communication](http://www.vision.ee.ethz.ch/datasets/b3dac2.en.html).  1000 high quality, dynamic 3D scans of faces, recorded while pronouncing a set of English sentences.
16. [Cohn-Kanade AU-Coded Expression Database](http://www.pitt.edu/~emotion/ck-spread.htm).  500+ expression sequences of 100+ subjects, coded by activated Action Units (Affect Analysis Group, Univ. of Pittsburgh.
17. [CMU/MIT Frontal Faces ](http://cbcl.mit.edu/software-datasets/FaceData2.html). Training set:  2,429 faces, 4,548 non-faces; Test set: 472 faces, 23,573 non-faces.
18. [AT&T Database of Faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) 400 faces of 40 people (10 images per people)



## Trained Model

1. [openface](https://github.com/cmusatyalab/openface). Face recognition with Google's FaceNet deep neural network using Torch.
2. [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). VGG-Face CNN descriptor. Impressed embedding loss.
3. [SeetaFace Engine](https://github.com/seetaface/SeetaFaceEngine). SeetaFace Engine is an open source C++ face recognition engine, which can run on CPU with no third-party dependence. 
4. [Caffe-face](https://github.com/ydwen/caffe-face) - Caffe Face is developed for face recognition using deep neural networks. 

5. [Norm-Face](https://github.com/happynear/NormFace) - Norm Face, finetuned from  [center-face](https://github.com/ydwen/caffe-face) and [Light-CNN](https://github.com/AlfredXiangWu/face_verification_experiment)


## Tutorial

1. [Deep Learning for Face Recognition](http://valse.mmcheng.net/deep-learning-for-face-recognition/). Shiguan Shan, Xiaogang Wang, and Ming yang.

## Software

1. [OpenCV](http://opencv.org/). With some trained face detector models.
2. [dlib](http://dlib.net/ml.html). Dlib implements a state-of-the-art of face Alignment algorithm.
3. [ccv](https://github.com/liuliu/ccv).  With a state-of-the-art frontal face detector
4. [libfacedetection](https://github.com/ShiqiYu/libfacedetection). A binary library for face detection in images.
5. [SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine). An open source C++ face recognition engine.

##Frameworks

1. [Caffe](http://caffe.berkeleyvision.org/)
2. [Torch7](https://github.com/torch/torch7)
3. [Theano](http://deeplearning.net/software/theano/)
4. [cuda-convnet](https://code.google.com/p/cuda-convnet/)
5. [MXNET](https://github.com/dmlc/mxnet/)
6. [Tensorflow](https://github.com/tensorflow)
7. [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn)

## Miscellaneous

1. [faceswap](https://github.com/matthewearl/faceswap)  Face swapping with Python, dlib, and OpenCV
2. [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial) Competition on Kaggle.
3. [An implementation of Face Alignment at 3000fps via Local Binary Features](https://github.com/freesouls/face-alignment-at-3000fps)

---

>Created by betars on 27/10/2015.
