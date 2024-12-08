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

### 

| Experiment      | #Shots   | Accuracy       | Link with Date (With commit)         |
|-----------------|----------|----------------|----------------------------|
| Unsupervised | 4 |  | |
| Unsupervised | 5 | 33.75% | [12/7/24](https://github.com/Rajeevveera24/manyshot-math/blob/f0acf7f1ca9e39d2c8fc05bf72cd45a6cd931b58/experiments/rveerara/5shot_unsupervised.json) |
| Unsupervised | 10 | 37.46%  | [12/7/24]()|
| Unsupervised | 25 | 41.80% | [12/7/24]() |
| Unsupervised | 50 | - | |
| Unsupervised | 125 | - | |
| Unsupervised | 250 | - | |
| Unsupervised | 500 | - | |

