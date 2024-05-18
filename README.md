SF(DA)<sup>2</sup>: Source-free Domain Adaptation Through the Lens of Data Augmentation
====
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sf-da-2-source-free-domain-adaptation-through/source-free-domain-adaptation-on-visda-2017)](https://paperswithcode.com/sota/source-free-domain-adaptation-on-visda-2017?p=sf-da-2-source-free-domain-adaptation-through) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sf-da-2-source-free-domain-adaptation-through/domain-adaptation-on-domainnet-1)](https://paperswithcode.com/sota/domain-adaptation-on-domainnet-1?p=sf-da-2-source-free-domain-adaptation-through)

![overview](https://github.com/shinyflight/SLOGAN/assets/25117385/5074be12-aca2-45fb-a90a-35768d4df5af)

**SF(DA)<sup>2</sup>: Source-free Domain Adaptation Through the Lens of Data Augmentation** (ICLR 2024) <br>
by [Uiwon Hwang](https://sites.google.com/view/uiwon-hwang), [Jonghyun Lee](https://scholar.google.com/citations?user=GPi5hw4AAAAJ&hl=ko), Juhyun Shin, [Sungroh Yoon](https://scholar.google.co.kr/citations?hl=ko&user=Bphl_fIAAAAJ) <br>

Paper: [https://openreview.net/forum?id=kUCgHbmO11](https://openreview.net/forum?id=kUCgHbmO11)

Abstract: In the face of the deep learning model's vulnerability to domain shift, source-free domain adaptation (SFDA) methods have been proposed to adapt models to new, unseen target domains without requiring access to source domain data. Although the potential benefits of applying data augmentation to SFDA are attractive, several challenges arise such as the dependence on prior knowledge of class-preserving transformations and the increase in memory and computational requirements. In this paper, we propose Source-free Domain Adaptation Through the Lens of Data Augmentation (SF(DA)<sup>2</sup>), a novel approach that leverages the benefits of data augmentation without suffering from these challenges. We construct an augmentation graph in the feature space of the pretrained model using the neighbor relationships between target features and propose spectral neighborhood clustering to identify partitions in the prediction space. Furthermore, we propose implicit feature augmentation and feature disentanglement as regularization loss functions that effectively utilize class semantic information within the feature space. These regularizers simulate the inclusion of an unlimited number of augmented target features into the augmentation graph while minimizing computational and memory demands. Our method shows superior adaptation performance in SFDA scenarios, including 2D image and 3D point cloud datasets and a highly imbalanced dataset.


Packages
----
We conducted experiments on the following versions of pakages:

- cudatoolkit == 11.3.1

- pytorch == 1.10.0

- python == 3.8

Data Preparation
----
- Please download the [VisDA dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and place in the path ``` ./data/visda-2017```.

Source Pretrained Model
----
- We used the source pretrained model parameters provided in the github repository of [SHOT](https://drive.google.com/drive/folders/1Hn3MXbwQF-A6UTBZG3L3ZBiwSrxctB35).

Adaptation
----

- VisDA: ```sh visda.sh```

  
## Citation 
    @inproceedings{
        hwang2024sfda,
        title={{SF}({DA})\${\textasciicircum}2\$: Source-free Domain Adaptation Through the Lens of Data Augmentation},
        author={Uiwon Hwang and Jonghyun Lee and Juhyeon Shin and Sungroh Yoon},
        booktitle={International Conference on Learning Representations},
        year={2024}
    }
