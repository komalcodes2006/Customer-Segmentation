# Customer Segmentation using Sequential and Parallel K-Means

## 1. Project Overview

This project implements customer segmentation using the K-Means clustering algorithm and compares a sequential implementation with a parallel implementation using OpenMP.

The main objective is to study how parallelization affects the execution time and scalability of K-Means as the dataset size and number of threads increase.

The project focuses on:

- Customer segmentation using K-Means
- Sequential K-Means implementation
- Parallel K-Means implementation using OpenMP
- Correctness comparison between sequential and parallel implementations
- Execution-time measurement
- Speedup and efficiency analysis
- Scalability with increasing dataset size
- Scalability with increasing thread count
- Identification of performance bottlenecks

The overall workflow is:

```text
Dataset
   ↓
Dataset Inspection
   ↓
Feature Selection
   ↓
Data Preprocessing
   ↓
Benchmark Dataset Generation
   ↓
Sequential K-Means
   ↓
Parallel OpenMP K-Means
   ↓
Correctness Comparison
   ↓
Performance Benchmarking
   ↓
Speedup / Efficiency Analysis
   ↓
Graphs and Conclusions
```

---

# 2. Initial Dataset Consideration

The first dataset considered was the UCI Online Retail dataset.

The dataset contains approximately 541,909 transaction records with columns including:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

During inspection, the following issues were identified:

- Approximately 4,372 identifiable customers
- 135,080 records with missing CustomerID
- 10,624 records with negative quantities
- 2,515 records with zero unit price
- The dataset is transaction-level rather than customer-level

Although the dataset is useful for customer analytics, it was not ideal for this project.

The project requires experiments involving datasets ranging from thousands to approximately one million customer records. The UCI dataset has only a few thousand identifiable customers.

Artificially duplicating or expanding the dataset to reach 100K, 500K, or 1M records would not provide a meaningful scalability experiment because the additional records would not represent genuinely different customers.

Therefore, the UCI Online Retail dataset was rejected for the final implementation.

---

# 3. Selected Dataset

The final dataset selected is:

**ziadatalabs/FreeEcommerceTwin14M**

The `customers.parquet` file is used.

The dataset contains:

- 1,000,000 customer records
- 7 columns

The columns are:

| Column | Description |
|---|---|
| `customer_id` | Customer identifier |
| `signup_date` | Customer signup date |
| `region` | Customer region |
| `segment` | Pre-existing customer segment |
| `lifetime_orders` | Number of lifetime orders |
| `lifetime_spend` | Lifetime customer spending |
| `email_domain` | Customer email domain |

## Important Dataset Note

The dataset is synthetic.

This should be explicitly disclosed in the final report. The main purpose of using this dataset is to provide a sufficiently large customer-level dataset for studying the computational behavior and scalability of K-Means.

The project should therefore not claim that the resulting clusters represent real-world customer populations.

---

# 4. Dataset Inspection

The dataset was inspected using:

```text
dataset_inspection.py
```

The original dataset has:

```text
Shape: 1,000,000 rows × 7 columns
```

The approximate data types are:

```text
customer_id        string
signup_date        string
region             string
segment            string
lifetime_orders    integer
lifetime_spend     float
email_domain       string
```

## 4.1 Customer Activity

A significant portion of the dataset consists of customers with no recorded orders.

Observed values:

```text
Total customers:    1,000,000
Active customers:     392,628
Inactive customers:   607,372
```

An active customer is defined as:

```text
lifetime_orders > 0
```

Therefore:

```text
1,000,000 - 392,628 = 607,372 inactive customers
```

The same number of customers also have zero lifetime spend.

---

# 5. Feature Selection

The goal is to cluster customers using behavioral information rather than identifiers or categorical metadata.

The following three features were selected:

```text
lifetime_orders
lifetime_spend
average_order_value
```

## 5.1 Average Order Value

Average Order Value (AOV) is calculated as:

```text
AOV = lifetime_spend / lifetime_orders
```

For customers with zero lifetime orders:

```text
AOV = 0
```

This provides a third behavioral feature describing the relationship between customer spending and order activity.

---

# 6. Features Excluded from Clustering

Several columns were deliberately excluded.

## customer_id

`customer_id` is an identifier and does not represent customer behavior.

Including it could introduce meaningless numerical relationships into the clustering process.

Therefore:

```text
customer_id → excluded
```

## segment

The dataset already contains a predefined `segment` column.

Using this column as an input feature would introduce an existing label into an unsupervised clustering task.

Therefore:

```text
segment → excluded
```

The existing segment column may still be useful for later analysis, but it is not used to determine the clusters.

## region

`region` is categorical.

Directly converting categories into integers would create artificial numerical relationships that do not represent meaningful Euclidean distances.

Therefore:

```text
region → excluded
```

## email_domain

`email_domain` is categorical and was not considered an appropriate initial behavioral feature.

Therefore:

```text
email_domain → excluded
```

## signup_date

The raw signup date was not directly used as a clustering feature.

Customer tenure could potentially be derived from this field in a future version, but it is outside the current feature set.

Therefore:

```text
signup_date → excluded
```

---

# 7. Preprocessing

The preprocessing pipeline is implemented in:

```text
preprocess.py
```

The preprocessing workflow is:

```text
Raw customer data
       ↓
Required-column validation
       ↓
Missing-value handling
       ↓
Average Order Value calculation
       ↓
Feature extraction
       ↓
log1p transformation
       ↓
StandardScaler
       ↓
Benchmark dataset generation
```

---

# 8. Log Transformation

The selected customer-behavior features are highly right-skewed.

Lifetime spend, in particular, contains many customers with low or zero spending and a smaller number of customers with substantially higher spending.

The preprocessing therefore applies:

```text
log1p(x)
```

which is equivalent to:

```text
log(1 + x)
```

The transformation helps reduce the influence of extremely large values while retaining zero-valued observations.

---

# 9. Standardization

After the log transformation, the features are standardized using `StandardScaler`.

Conceptually:

```text
z = (x - mean) / standard_deviation
```

This puts the features onto comparable scales.

Scaling is important because K-Means relies on Euclidean distance. Without scaling, a feature with a larger numerical range could dominate the distance calculation.

---

# 10. Full Dataset and Active Dataset

Two processed versions of the data are retained.

## Full Dataset

The full dataset contains:

```text
1,000,000 customers
```

This dataset is important for computational benchmarking and scalability experiments.

## Active Customer Dataset

The active dataset contains:

```text
392,628 customers
```

where:

```text
lifetime_orders > 0
```

The active dataset is retained for behavioral analysis because inactive customers form a very large population in the original dataset.

The inactive customers are therefore not silently deleted from the project. The full dataset remains available for benchmarking, while the active subset provides an additional analytical view.

---

# 11. Benchmark Dataset Sizes

The following benchmark sizes were generated:

```text
10,000
50,000
100,000
500,000
1,000,000
```

These datasets allow the project to investigate how execution time changes as the input size grows.

The generated datasets are stored locally under:

```text
data/processed/benchmarks/
```

They are intentionally excluded from GitHub because they can be recreated using the preprocessing pipeline.

---

# 12. K-Means Configuration

The current K-Means configuration is:

| Parameter | Value |
|---|---:|
| Number of clusters (K) | 4 |
| Maximum iterations | 100 |
| Convergence tolerance | 1e-4 |
| Distance metric | Squared Euclidean distance |
| Random seed | 42 |
| Number of features | 3 |

## Important Note About K = 4

`K = 4` is currently an experimental configuration.

It should **not** be interpreted as proof that four is the optimal number of customer segments.

The project is primarily focused on comparing sequential and parallel performance.

Additional cluster-selection analysis such as silhouette score or testing different values of K could be performed separately if required.

---

# 13. K-Means Algorithm

The implementation follows the standard K-Means procedure.

## Step 1 — Initialization

Four initial centroids are selected from the input data.

A fixed random seed:

```text
42
```

is used to improve reproducibility.

## Step 2 — Assignment

Each customer is assigned to the nearest centroid.

For a customer point `x` and centroid `c`, squared Euclidean distance is:

```text
d(x,c) = Σ(x_i - c_i)^2
```

The customer is assigned to the cluster with the smallest distance.

## Step 3 — Centroid Update

For every cluster, the centroid is recalculated as the mean of all points assigned to that cluster.

## Step 4 — Convergence

The algorithm checks how much the centroids moved.

If centroid movement is below:

```text
1e-4
```

the algorithm is considered converged.

Otherwise, another iteration begins.

---

# 14. Sequential Implementation

The sequential implementation is written in C++:

```text
src/kmeans_sequential.cpp
```

The implementation contains separate stages for:

- Loading CSV data
- Initializing centroids
- Calculating squared distances
- Assigning clusters
- Updating centroids
- Checking convergence
- Calculating final cluster sizes
- Calculating inertia
- Measuring execution time

The execution time measures the K-Means computation rather than CSV loading and preprocessing.

This makes the benchmark more focused on the computational cost of K-Means itself.

---

# 15. Sequential Benchmark Results

The following measurements were obtained from the sequential C++ implementation.

These are actual measurements from the development machine. They should not be treated as expected execution times on other machines.

| Dataset Size | Iterations | Time (seconds) | Inertia |
|---:|---:|---:|---:|
| 10,000 | 46 | 0.00376421 | 783.082 |
| 50,000 | 38 | 0.0153687 | 3,889.63 |
| 100,000 | 40 | 0.0311442 | 7,734.9 |
| 500,000 | 16 | 0.062535 | 39,002.3 |
| 1,000,000 | 38 | 0.314918 | 77,692.8 |

---

# 16. Cluster Sizes

The final cluster sizes obtained for each benchmark were:

## 10,000 customers

```text
Cluster 0: 654
Cluster 1: 1,302
Cluster 2: 1,956
Cluster 3: 6,088
```

## 50,000 customers

```text
Cluster 0: 9,966
Cluster 1: 30,269
Cluster 2: 6,521
Cluster 3: 3,244
```

## 100,000 customers

```text
Cluster 0: 12,871
Cluster 1: 19,697
Cluster 2: 60,842
Cluster 3: 6,590
```

## 500,000 customers

```text
Cluster 0: 303,712
Cluster 1: 99,645
Cluster 2: 32,431
Cluster 3: 64,212
```

## 1,000,000 customers

```text
Cluster 0: 199,088
Cluster 1: 607,842
Cluster 2: 64,871
Cluster 3: 128,199
```

Cluster numbers are arbitrary identifiers and do not have inherent semantic meaning.

For example, Cluster 0 in one run does not necessarily represent the same behavioral group as Cluster 0 in another run.

---

# 17. Initial Observations

## 17.1 Execution Time Increases with Dataset Size

The overall execution time increases as the number of customers increases.

Measured times range from approximately:

```text
0.0038 seconds for 10K
```

to:

```text
0.315 seconds for 1M
```

This demonstrates the increasing computational cost of processing larger datasets.

However, the relationship is not perfectly linear because total runtime depends on both:

```text
Number of data points
```

and:

```text
Number of K-Means iterations
```

---

# 18. Iteration Count Is Not Monotonic

The number of iterations does not consistently increase with dataset size.

Observed values:

```text
10K    → 46 iterations
50K    → 38 iterations
100K   → 40 iterations
500K   → 16 iterations
1M     → 38 iterations
```

Therefore, dataset size alone does not determine the number of iterations required for convergence.

The initialization and distribution of each dataset also influence convergence.

This is important when interpreting performance results.

---

# 19. Observation About Inactive Customers

The full dataset contains:

```text
607,372 inactive customers
```

The 1M-customer K-Means run produced a cluster containing:

```text
607,842 customers
```

This is very close to the number of inactive customers.

This suggests that K-Means is separating a large low-activity or zero-activity population from customers with higher behavioral activity.

However, this observation is a sanity check rather than proof that:

```text
K = 4
```

is the optimal number of clusters.

The cluster should be interpreted using its centroid and feature statistics rather than only its size.

---

# 20. Inertia

The implementation calculates K-Means inertia.

Inertia is the sum of squared distances between every data point and its assigned centroid.

Conceptually:

```text
Inertia = Σ distance(point, assigned_centroid)^2
```

Measured values:

```text
10K       → 783.082
50K       → 3,889.63
100K      → 7,734.9
500K      → 39,002.3
1M        → 77,692.8
```

Inertia naturally increases as the number of data points increases because more points contribute to the total.

Therefore, raw inertia should not be used alone to compare clustering quality across different dataset sizes.

---

# 21. Cluster Label Permutation

K-Means cluster numbers are arbitrary.

For example:

```text
Cluster 0
Cluster 1
Cluster 2
Cluster 3
```

are simply labels.

The same four centroids may receive different numerical labels in another run.

Therefore, when comparing sequential and parallel implementations, we should not require:

```text
Sequential Cluster 0 == Parallel Cluster 0
```

Instead, correctness should be evaluated using:

- Comparable final centroids
- Comparable inertia
- Comparable cluster membership after accounting for label permutation
- Similar convergence behavior

---

# 22. Why K-Means Is Suitable for Parallelization

A major computational component of K-Means is the assignment phase.

For each customer, the algorithm performs:

```text
Calculate distance to centroid 0
Calculate distance to centroid 1
Calculate distance to centroid 2
Calculate distance to centroid 3
Select nearest centroid
```

The calculation for one customer is independent of the calculation for another customer during this phase.

The centroids remain read-only during the assignment phase.

Therefore, customers can be divided among multiple threads.

Conceptually:

```text
Customers
──────────────────────────────────

Thread 1 → subset of customers
Thread 2 → subset of customers
Thread 3 → subset of customers
Thread 4 → subset of customers
```

This makes the assignment phase a natural candidate for OpenMP parallelization.

---

# 23. Centroid Update and Race Conditions

The centroid update phase requires more care.

Each cluster requires:

```text
Sum of feature values
Number of points
```

If multiple threads directly modify the same shared sums, race conditions can occur.

Conceptually:

```text
Thread 1 ──┐
Thread 2 ──┼──> Shared cluster sum
Thread 3 ──┤
Thread 4 ──┘
```

Multiple simultaneous writes could produce incorrect results.

Possible approaches include:

- Per-thread local sums followed by reduction
- OpenMP reduction where appropriate
- Thread-local accumulators followed by final aggregation

Correctness should be established before aggressive optimization.

---

# 24. Planned Parallel Performance Experiments

After the OpenMP implementation is complete, performance will be measured across:

## Dataset sizes

```text
10K
50K
100K
500K
1M
```

## Thread counts

For example:

```text
1
2
4
8
```

The exact thread counts should depend on the available hardware.

The 1-thread OpenMP result is particularly useful as a baseline for the parallel implementation.

---

# 25. Performance Metrics

## Execution Time

The primary measurement is:

```text
T
```

where `T` is the K-Means execution time.

## Speedup

Speedup is calculated as:

```text
Speedup = T_sequential / T_parallel
```

For example, if:

```text
Sequential = 1.0 seconds
Parallel = 0.5 seconds
```

then:

```text
Speedup = 2
```

## Efficiency

Parallel efficiency is:

```text
Efficiency = Speedup / Number_of_threads
```

As a percentage:

```text
Efficiency (%) =
Speedup / Number_of_threads × 100
```

All final performance values will be calculated from actual benchmark measurements.

No performance values should be assumed or invented before running the parallel implementation.

---

# 26. Expected Performance Considerations

Parallel speedup will not necessarily increase proportionally with the number of threads.

Potential limitations include:

- Serial portions of the algorithm
- Centroid update overhead
- Synchronization
- Thread scheduling overhead
- Memory bandwidth
- Cache behavior
- Small dataset sizes
- Differences in convergence iteration count

Therefore, the final analysis should be based on measured performance rather than assuming ideal linear speedup.

---

# 27. Current Project Status

## Completed

- [x] Dataset selection
- [x] Dataset inspection
- [x] Dataset quality analysis
- [x] Feature selection
- [x] Feature exclusion decisions
- [x] Average Order Value calculation
- [x] Log transformation
- [x] Standardization
- [x] Full dataset processing
- [x] Active dataset processing
- [x] Benchmark dataset generation
- [x] Sequential K-Means implementation
- [x] Sequential performance measurements
- [x] Initial interpretation of clustering behavior

## Remaining

- [ ] OpenMP parallel K-Means
- [ ] Parallel correctness validation
- [ ] Thread-count experiments
- [ ] Parallel performance measurements
- [ ] Speedup calculation
- [ ] Efficiency calculation
- [ ] Bottleneck analysis
- [ ] Performance graphs
- [ ] Final report
- [ ] LLM usage documentation
- [ ] Individual contribution documentation

---

# 28. Reproducibility

The project does not store the raw dataset or generated benchmark datasets in GitHub.

The repository contains the scripts required to recreate the data-processing pipeline.

The intended workflow is:

```bash
python download_data.py
python preprocess.py
```

This recreates the required processed datasets locally.

The raw dataset and generated benchmark data are excluded using `.gitignore`.

This keeps the repository lightweight while allowing other team members to reproduce the preprocessing pipeline.

---

# 29. Repository Structure

The intended repository structure is:

```text
Customer-Segmentation/
│
├── src/
│   ├── kmeans_sequential.cpp
│   └── kmeans_parallel.cpp
│
├── docs/
│   └── DEVELOPMENT.md
│
├── results/
│   ├── sequential_results.csv
│   ├── parallel_results.csv
│   └── graphs/
│
├── dataset_inspection.py
├── download_data.py
├── preprocess.py
├── README.md
├── .gitignore
└── requirements.txt
```

Generated datasets, raw datasets, compiled binaries, Python virtual environments, and cache files should remain outside version control.

---

# 30. Current Conclusions

At the current stage, the project has established a complete data-preparation and sequential K-Means baseline.

The selected dataset provides enough customer records to perform scalability experiments up to one million records without artificially duplicating a small real-world dataset.

The preprocessing pipeline converts the original customer information into three numerical behavioral features:

```text
Lifetime Orders
Lifetime Spend
Average Order Value
```

The use of `log1p` followed by standardization makes the features more suitable for Euclidean-distance-based clustering.

The sequential K-Means implementation successfully processes all five benchmark sizes.

The measured results show that execution time generally increases with input size, while the number of iterations varies depending on the dataset and initialization.

The clustering results also show a very large low-activity cluster in the full 1M-customer dataset, which is consistent with the large number of inactive customers present in the original data.

However, the current results do not establish that:

```text
K = 4
```

is the optimal number of customer segments.

They also do not yet establish the performance benefit of parallelization.

Those conclusions require the remaining OpenMP implementation and benchmarking experiments.

---

# 31. Next Immediate Step

The next implementation task is:

```text
Implement OpenMP parallel K-Means
```

Recommended development order:

```text
1. Use the sequential implementation as the baseline
2. Parallelize customer-to-centroid distance calculations
3. Implement safe parallel centroid accumulation
4. Test the parallel version with 1 thread
5. Compare the output against the sequential version
6. Test with multiple threads
7. Benchmark all dataset sizes
8. Calculate speedup and efficiency
9. Analyze bottlenecks
10. Generate performance graphs
```

The parallel implementation should prioritize correctness before optimization.

---

# 32. Important Limitations

The following limitations should be acknowledged in the final report:

1. The selected customer dataset is synthetic.
2. K = 4 is currently an experimental choice rather than a validated optimal value.
3. The clustering features are limited to three behavioral variables.
4. The initial experiments use a fixed random seed.
5. Execution times depend on the hardware and software environment.
6. Raw inertia increases naturally with dataset size and should not be used alone to compare clustering quality across different dataset sizes.
7. Cluster labels are arbitrary and may be permuted between implementations.
8. Parallel floating-point calculations may produce very small numerical differences from the sequential implementation.
9. The current benchmark measures the K-Means computation and does not include data loading or preprocessing time.

---

# 33. Development Principle

The project should clearly distinguish between:

```text
Observed Results
```

and:

```text
Interpretation
```

All performance values reported in the final report should come from actual benchmark runs.

Similarly, conclusions about customer segments should be based on the resulting cluster statistics rather than assumptions about what a cluster number represents.

The primary objective of the project is to demonstrate both:

```text
Correct customer segmentation
```

and:

```text
The computational and performance behavior of sequential
versus parallel K-Means.
```