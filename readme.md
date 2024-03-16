Code for paper *"SF(DA)<sup>2</sup>: Source-free Domain Adaptation Through the Lens of Data Augmentation"*
====

Packages
----
We conducted experiments on the following versions of pakages:

- cudatoolkit == 11.3.1

- pytorch == 1.10.0

- python == 3.8

Data Preparation
----
- Please download the VisDA dataset from the official website and place in the path ``` ./data/visda-2017```.

Source Pretrained Model
----
- We used the source pretrained model parameters provided in the github repository of SHOT.

Adaptation
----

- VisDA: ```sh visda.sh```



**Note**: We also include log files (for both linux and windows) generated with three different seeds in ```./log/{linux or windows}```
More code will be included in our Github repository.