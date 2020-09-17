# e-motions

This repository contains a set of four CNNs (Convolutional Neural Networks) which can be trained to recognize actions in still images and videos based on optical flow.

## Getting Started

1. Download the [image](http://vision.stanford.edu/Datasets/40actions.html) and [video](https://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html) datasets and add them to the data folder
2. Run main.py -> this will preprocess the data as well as train the CNNs
3. Do your predictions!

[Note]: Every time you run main.py after the first time, it will recognize the data and CNNs that are already trained (and saved), and won't train them again.

### Prerequisites

Install the following prerequisites:

**Packages**:
* tensorflow == "1.8"
* matplotlib
* numpy
* h5py
* pandas
* seaborn
* opencv-python
* scikit-learn == "0.22.2"

**Requires**:
* python_version = "3.6"

## Authors

* **Lorenzo Bini**
* **Neele Dijkstra**

