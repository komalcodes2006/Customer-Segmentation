# Parallel Customer Segmentation using K-Means

## Overview

This project implements customer segmentation using the K-Means clustering algorithm and compares a sequential implementation with a parallel implementation using OpenMP.

The objective is to study the performance and scalability of K-Means as the dataset size and number of threads increase.

## Dataset

The project uses the synthetic `customers.parquet` dataset from the Hugging Face dataset:

`ziadatalabs/FreeEcommerceTwin14M`

The dataset contains 1,000,000 customer records.

The raw dataset is not stored in this repository. It can be downloaded using:

```bash
python download_data.py