 DGMwTensorFlow
======

[TOC]

*****

### Framework of deep galerkin model (DGM)
![](./Pictures/DGM_Framework.png)

* Each DGM layer

![](./Pictures/DGM_singlelayer.png)

### Training function and  Inference function

* Objective function

  ![](./Pictures/DGM_ObjectiveFunction.png)

* Target model

  ![](./Pictures/DGM_targetmodel.jpg)

* Calculation of inference accuracy

![](./Pictures/DGM_infformula.jpg)

### Training Process

![](./Pictures/DGM_TrainingProcess.png)

* First 10 training epoch

![](./Pictures/DGM_10ep_training.png)

### Inference accuracy and results

![](./Pictures/DGM_InferenceAccuracy.png)

* Inference results at 6 given moment in time

  ![](./Pictures/DGM_inferencePlot.png)

### Speedup strategies for the Model Inference

(1) **All CPU**

![](./Pictures/DGM_inf_allCPU.png)

(2) **All GPU**

![](./Pictures/DGM_inf_allGPU.png)

(3) **Layer 2 - layer 4 on GPU, layer 1 and layer 5  on CPU **

![](./Pictures/DGM_inf_CPUwGPU.png)

(4) **Results**
![](./Pictures/DGM_inf_CaseResult.png)

### Experimental environment

* **Hardware**

1. AMD Ryzen 5 2600 (6 cores 12 threads)
2. NVIDIA Geforce GTX 1070 (8 GB)
3. 16 GB system RAM

* **Software**

1. Python 3.7.1/ numpy 16.4
2. tensorflow 1.14.0/ tensorflow-gpu 1.14.0
3. cudatoolkit 10.0.13/ cudnn 7.6.0
4. Windows 10

### Reference

[1]  Al-Aradi et al., “Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning,” arXiv:1811.08782v1 [q-fin.CP], 21 Nov 2018.

[2] Justin Sirignano et al., “DGM: A deep learning algorithm for solving partial differential equations,” Journal of Computational Physics 375 (2018) 1339–1364 Contents.

[3] https://zhuanlan.zhihu.com/p/140343833

[4] https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs/python/tf (API doc)

[5] https://github.com/tensorflow/docs/tree/master/site/en/r1 (official tutorial)

[6] http://deeplearnphysics.org/Blog/index.html 
