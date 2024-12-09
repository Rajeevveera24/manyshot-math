# MANYSHOT-MATH

A course project for course [10-623 Generative AI](https://www.cs.cmu.edu/~mgormley/courses/10423/) at Carnegie Mellon University (Fall 2024) exploring many-shot in-context learning for math reasoning and problem solving.

This project aims to replicate and extend some of the findings from Google DeepMind's "Many-Shot In-Context Learning" paper (NeurIPS 2024) - [https://arxiv.org/pdf/2404.11018](https://arxiv.org/pdf/2404.11018), investigating whether their results on Gemini models generalize to smaller open source models like Llama-8B. We explore how providing hundreds or thousands of examples during inference affects performance on math reasoning tasks like MATH and GSM8K.

Key aspects we investigate:

- Reinforced ICL: Testing if model-generated chain-of-thought rationales can replace human rationales
- Unsupervised ICL: Examining if providing only problem inputs helps models leverage pre-trained knowledge
- Comparing performance between Gemini and Llama-8B
- Evaluation on standard math reasoning benchmarks

Our goal is to validate whether many-shot learning's benefits, as demonstrated on large proprietary models, extend to more accessible open source models like Llama-8B.

## Set up

### Environments

We provide conda environment files for both macOS and Linux. To set up your environment, follow the instructions below for your operating system:

#### MacOS Setup

To create and/or update the conda environment on MacOS, use the following command:

```
conda env update --file conda_env_reqs_macos.yml --prune
```

Once you update the environment, you can export the new environment file using the following command:

```
conda env export > conda_env_reqs_macos.yml
```

This allows other developers to update their environments with the latest dependencies.

#### Linux Setup

To create and/or update the conda environment on Linux, use the following command:

```
conda env update --file conda_env_reqs_linux.yml --prune
```

Once you update the environment, you can export the new environment file using the following command:

```
conda env export > conda_env_reqs_linux.yml
```

This allows other developers to update their environments with the latest dependencies.

## Our Results

All our results are reported on questions sourced from MATH500 dataset and answers inferenced from Llama-3.1-8B-Instruct model

### MATH500 - Numeric Subset (Questions with Numeric Ground Truth Answers Only) 

####

| Experiment      | #Shots   | Accuracy       | Link with Date (To commit)         |
|-----------------|----------|----------------|----------------------------|
| Unsupervised | 4 | 32.82% | [12/8/24]() |
| Unsupervised | 5 | 33.75% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/f0acf7f1ca9e39d2c8fc05bf72cd45a6cd931b58/experiments/rveerara/5shot_unsupervised.json) |
| Unsupervised | 10 | 37.46%  | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/d35d295b97ff2239c190cacf493e69e349dcde20/experiments/rveerara/10shot_unsupervised.json)|
| Unsupervised | 25 | 41.80% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/7dede723501f83b65f277aab017bcad2440a15f6/experiments/rveerara/25shot_unsupervised.json) |
| Unsupervised | 50 | 43.34% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/94ac29e2e47996d463c6dd62187e21557aeb460c/experiments/rveerara/50shot_unsupervised.json) |
| Unsupervised | 75 | 45.20% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/437bb3be2af17042014dd8c6e94987d9c921d162/experiments/rveerara/75shot_unsupervised.json) |
| Unsupervised | 100 | 44.58% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/43466f90fc628ced114665881ee914bd8c59c262/experiments/rveerara/100shot_unsupervised.json) |
| Unsupervised | 125 | 41.49% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/f20e6c58e3ebea4a4981c63ce2fee29469a9f3a2/experiments/rveerara/125shot_unsupervised.json) |
| Unsupervised | 250 | 41.49% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/eb75b3784b59eb1b66ac89eae56d780fc62c23d2/experiments/rveerara/250shot_unsupervised.json) |
| Unsupervised | 500 | - | |
| Supervised | 4 | 31.89% | [12/9/24]() |
| Supervised | 5 | 36.84% | [12/9/24]() |
| Supervised | 10 |  | |
| Supervised | 25 |  | |
| Supervised | 50 |  | |
| Supervised | 75 |  | |
| Supervised | 100 |  | |
| Supervised | 125 |  | |
| Supervised | 250 |  | |
| Supervised | 500 |  | |
| Re-inforced | 4 |  | |
| Re-inforced | 5 |  | |
| Re-inforced | 10 |  | |
| Re-inforced | 25 |  | |
| Re-inforced | 50 |  | |
| Re-inforced | 75 |  | |
| Re-inforced | 100 |  | |
| Re-inforced | 125 |  | |
| Re-inforced | 250 |  | |
| Re-inforced | 500 |  | |

### MATH500 - Full Dataset Subset

####

| Experiment      | #Shots   | Accuracy       | Link with Date (To commit)         |
|-----------------|----------|----------------|----------------------------|
| Unsupervised | 4 |  | |
| Unsupervised | 5 |  | |
| Unsupervised | 10 |  | |
| Unsupervised | 25 |  | |
| Unsupervised | 50 |  | |
| Unsupervised | 75 |  | |
| Unsupervised | 100 |  | |
| Unsupervised | 125 |  | |
| Unsupervised | 250 |  | |
| Unsupervised | 500 |  | |
| Supervised | 4 |  | |
| Supervised | 5 |  | |
| Supervised | 10 |  | |
| Supervised | 25 |  | |
| Supervised | 50 |  | |
| Supervised | 75 |  | |
| Supervised | 100 |  | |
| Supervised | 125 |  | |
| Supervised | 250 |  | |
| Supervised | 500 |  | |
| Re-inforced | 4 |  | |
| Re-inforced | 5 |  | |
| Re-inforced | 10 |  | |
| Re-inforced | 25 |  | |
| Re-inforced | 50 |  | |
| Re-inforced | 75 |  | |
| Re-inforced | 100 |  | |
| Re-inforced | 125 |  | |
| Re-inforced | 250 |  | |
| Re-inforced | 500 |  | |