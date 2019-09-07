# Building change detection using deep learning techniques
Author Yinxia Cao

# Introduction

Structure of net: fully convolutional networks based on resnet-101 structures

Environment: matlab 2018b  windows

Theme: building change detection in remote sensing 

challenges: (1) unbalanced (skewed) proportion of change and no-change   (2) end-to-end learning

# Workflow: 

0. materials

image: two images aquired at different times

label: mannually delineated (time-consuming but important)


1.data preprocessing->

  image registration: resampled to the same spatial unit

  radiometric normalization


2.preparing training samples -> 

  training samples include (1) image pair (2) label (of the same size as the image: 1 change 0 no-change)
  
  key parameter: the size of image, which should be adjusted according to the image resolution and context

3.train a net -> 

  key parameter: pre-trained net parameters, learning rate...
  
  note that: visualize the feature map of a specific layer to ensure the net works 
  
4.prediction

  input: the learned net parameters & image pairs
  
  output: the building change map (end-to-end)
