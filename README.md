# BiCANet
<h1 align="center"> BiCANet: Bi-directional Contextual Aggregating Network for Image Semantic Segmentation
 </h1>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#setup">Setup</a> •
  <a href="#content">Content</a> •
  <a href="#results">Results</a>
</p>

## Description
This work is inspired by the article about <a href="https://arxiv.org/pdf/2003.09669.pdf" target="_blank"> BiCANet </a>, which aggregates contextual cues from a categorical perspective, which mainly consists of three parts: contextual condensed projection block (CCPB), bidirectional context interaction block (BCIB), and muti-scale contextual fusion block (MCFB).

The aforementioned architecture was implemeted from scratch, adapted towards <a href="https://disk.yandex.ru/d/1SSkfLh4WnEhmw" target="_blank"> photos of human faces </a>, tuned and tested.

## Dependencies
Install all the dependencies with

```
pip install -r ./pip-requirements-python-3.9.13.txt
```

## Setup

Clone GitHub repository:

```
git clone https://github.com/bcd8697/BiCANet
```

Make sure that all dependencies are installed and run the notebook, containing the experiments for the model.

## Content

This repo contains the codebase for the project, its structure is following:
* **model** folder contains codebase for the BiCANet model, as well as the detailed description and implementation details;
* **weights** folder contains checkpoint weights for the model, which were obtained after learning;
* **images** folder contains images used in the readme.

## Results

Model was trained on the dataset with human photos. Portrait Segmentation had been solving.

Results of the model perfomance could be seen below:

<p align="center">
  <img width="815" height="484" src="https://github.com/bcd8697/BiCANet/blob/main/images/result_bicanet.png">
</p>

