# 🔬 Unsupervised Learning — PCA, K-Means, Hierarchical Clustering & Matrix Completion

> **Skills Demonstrated:** Principal Component Analysis (PCA) · K-Means Clustering · Hierarchical Clustering · Dendrogram Analysis · Matrix Completion (SVD & PCA) · Gene Expression Analysis · Correlation-Based Distance · StandardScaler · Python · Scikit-learn · SciPy

---

## 🎯 Project Overview

This project applies **Unsupervised Learning** techniques to discover hidden structure in data — without any labeled target variable. It answers real-world analytical questions:

> *"Can we group US states by crime patterns? Can we recover missing data? Which genes differ most between healthy and diseased patients?"*

Seven exercises are covered across five datasets:

1. **USArrests** — Proving equivalence of correlation-based vs Euclidean distance (Ex. 7)
2. **USArrests** — Verifying PVE calculation via two independent methods (Ex. 8)
3. **USArrests** — Hierarchical clustering with and without scaling (Ex. 9)
4. **Simulated Data** — PCA + K-Means clustering benchmark across K values (Ex. 10)
5. **Boston Dataset** — Matrix Completion via SVD Algorithm 12.1 (Ex. 11)
6. **Boston Dataset** — Matrix Completion reimplemented using PCA (Ex. 12)
7. **Gene Expression Data** — Hierarchical clustering + gene identification (Ex. 13)

---

## 📁 Datasets Used

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| **USArrests** | U.S. Census Bureau (Real) | 50 states, 4 features | State clustering by crime profile |
| **Simulated** | `numpy.random` | 60 obs, 50 features, 3 classes | K-Means & PCA benchmark |
| **Boston** | U.S. Census Bureau (Real) | 506 rows, 13 features | Matrix completion with missing data |
| **Ch12Ex13 (Gene Expression)** | [statlearning.com](https://www.statlearning.com) (Real) | 40 patients, 1,000 genes | Healthy vs diseased gene analysis |

---

## 🔧 Techniques & Tools Applied

| Technique | Library | Purpose |
|-----------|---------|---------|
| Principal Component Analysis (PCA) | `sklearn.decomposition.PCA` | Dimensionality reduction & variance explanation |
| PVE (Proportion of Variance Explained) | Manual + `explained_variance_ratio_` | Dual-method verification |
| K-Means Clustering (K=2,3,4) | `sklearn.cluster.KMeans` | Flat partitioning of observations |
| Agglomerative Hierarchical Clustering | `sklearn.cluster.AgglomerativeClustering` | Tree-based clustering with dendrogram |
| Correlation-Based Distance | `scipy.cluster.hierarchy.linkage` | Gene expression dissimilarity |
| Complete / Single / Average Linkage | `scipy.cluster.hierarchy` | Linkage method comparison |
| Matrix Completion (SVD) | `numpy.linalg.svd` | Recovering missing data entries |
| Matrix Completion (PCA) | `sklearn.decomposition.PCA` | Alternative implementation |
| StandardScaler | `sklearn.preprocessing.StandardScaler` | Feature standardization |
| Dendrogram Visualization | `scipy.cluster.hierarchy.dendrogram` | Cluster tree plotting |

**Libraries:** `numpy` · `pandas` · `scikit-learn` · `scipy` · `matplotlib` · `ISLP`

---

## 📊 Key Results

### Exercise 7 — Distance Equivalence Proof (USArrests)

**Claim:** After standardizing, `1 − r_ij` is proportional to squared Euclidean distance.

| Matrix | Values (off-diagonal) |
|--------|----------------------|
| Squared Euclidean Distance | Variable |
| 1 − Correlation | Variable |
| **Proportionality constant** | **0.01 (constant across all pairs)** |

```
prop_matrix = array([[ inf,  0.01, 0.01, 0.01],
                     [0.01,   nan, 0.01, 0.01],
                     [0.01,  0.01,  inf, 0.01],
                     [0.01,  0.01, 0.01,  nan]])
```
> **Proved ✅:** Proportionality constant = **0.01** for all feature pairs in USArrests. Diagonal entries are inf/nan (distance from a feature to itself).

---

### Exercise 8 — PVE Verification: Two Methods (USArrests)

**Method (a):** `explained_variance_ratio_` from fitted PCA
**Method (b):** Direct application of Equation 12.10 using `components_` loadings

| PC | PVE (Manual Formula) | Ratio (a/b) |
|----|---------------------|-------------|
| PC1 | **0.6201** | **1.0** ✅ |
| PC2 | **0.2474** | **1.0** ✅ |
| PC3 | **0.0891** | **1.0** ✅ |
| PC4 | **0.0434** | **1.0** ✅ |
| **Total** | **1.0000** | — |

> **Verified ✅:** Both methods produce identical results (`ratio = [1., 1., 1., 1.]`). PC1 alone explains **62%** of variance in US crime data — a single component captures most of the state-level crime variation.

---

### Exercise 9 — Hierarchical Clustering of US States (Complete Linkage)

#### Unscaled vs Scaled Clustering (3 clusters, Euclidean distance)

**Unscaled Cluster Sizes:**

| Cluster | Size | Character |
|---------|------|-----------|
| Cluster 0 | **16 states** | High crime (dominated by Assault magnitude) |
| Cluster 1 | **14 states** | Mid-level crime |
| Cluster 2 | remaining | Low crime / rural states |

**Scaled Cluster Sizes:**

| Cluster | Size | Character |
|---------|------|-----------|
| Cluster 0 | **8 states** | High crime across all metrics |
| Cluster 1 | **11 states** | Mid-level crime |
| Cluster 2 | remaining | Low crime states |

**State-level comparison (sample):**

| State | Unscaled Cluster | Scaled Cluster | Changed? |
|-------|-----------------|----------------|---------|
| Alabama | 0 | 0 | ❌ Same |
| Arizona | 0 | 1 | ✅ Changed |
| California | 0 | 1 | ✅ Changed |
| Arkansas | 1 | 2 | ✅ Changed |
| Connecticut | 2 | 2 | ❌ Same |

> **Key Finding:** Scaling significantly changes cluster assignments. Without scaling, `Assault` (values ~100–300) dominates distance calculations over `Murder` (values ~3–17) purely due to magnitude. **Variables must be scaled** before clustering when features are measured in different units — as is the case here (crime rates per 100,000 vs urban population percentage).

---

### Exercise 10 — K-Means & PCA on Simulated Data (n=60, p=50, 3 classes)

**Setup:** 3 classes of 20 observations each, with distinct mean shifts in feature subsets

#### K-Means with K=3 (Raw Data) — Perfect Recovery ✅

```
y           0   1   2
kmeans_3C
0           0  20   0   ← Class 1 perfectly captured
1          20   0   0   ← Class 0 perfectly captured
2           0   0  20   ← Class 2 perfectly captured
```

#### K-Means with K=2 — Classes 1 & 2 Merged

```
y           0   1   2
kmeans_2C
0          20   0   0   ← Class 0 alone
1           0  20  20   ← Classes 1 & 2 merged into one
```

#### K-Means with K=4 — Class 0 Artificially Split

```
y           0   1   2
kmeans_4C
0           0  20   0
1          10   0   0   ← Class 0 split into two halves
2           0   0  20
3          10   0   0   ← Other half of Class 0
```

#### K-Means on First 2 PCA Components — Perfect Recovery ✅

```
y               0   1   2
kmeans_3C_pca
0              20   0   0
1               0   0  20
2               0  20   0
```

#### K-Means on Scaled Data (StandardScaler) — Perfect Recovery ✅

```
y                0   1   2
kmeans_3C_stds
0                0   0  20
1               20   0   0
2                0  20   0
```

> **Key Findings:**
> - K=3 correctly recovers all 3 true classes on raw, PCA-reduced, and scaled data
> - K-Means on 2 PCA components gives identical results as full 50-feature data — confirming PCA captures the essential cluster structure
> - Scaling has no negative impact when classes are well-separated with mean shifts

---

### Exercise 11 — Matrix Completion via SVD (Boston Dataset)

**Setup:** 13 features standardized (excl. `chas` dummy), missing fractions 5%–30%, M=1–8 components, 10 repetitions averaged

**Averaged Approximation Error by Missing Fraction:**

| Missing Fraction | Avg. Approximation Error |
|-----------------|--------------------------|
| **5%** | **0.0695** |
| **10%** | **0.0885** |
| **15%** | **0.1075** |
| **20%** | **0.1222** |
| **25%** | ~0.135 |
| **30%** | ~0.150 |

> **Key Finding:** Approximation error increases monotonically with missing fraction — from **6.9% at 5% missing** to **~15% at 30% missing**. The number of principal components M (1–8) does **not** meaningfully affect error for the Boston dataset, suggesting a single component captures most of the recoverable structure.

---

### Exercise 12 — Matrix Completion via PCA (Boston Dataset)

Reimplemented Algorithm 12.1 using `sklearn.decomposition.PCA` instead of `numpy.linalg.svd`.

> **Result:** PCA-based implementation produces equivalent matrix completion results to the SVD approach, confirming the mathematical equivalence between SVD and PCA decomposition highlighted in Section 12.5.2.

---

### Exercise 13 — Gene Expression Analysis (40 Patients, 1,000 Genes)

**Setup:** 20 healthy vs 20 diseased patients, correlation-based hierarchical clustering

#### Linkage Method Comparison (2 clusters):

| Patient | Complete Linkage | Single Linkage | Average Linkage |
|---------|-----------------|----------------|-----------------|
| 0 | 0 | 0 | 0 |
| 1 | **1** | 0 | 0 |
| 4 | **1** | 0 | **1** |
| 5 | 0 | 0 | **1** |
| 9 | **1** | 0 | **1** |

> **Finding:** Results depend significantly on linkage type — **Single linkage** places almost all patients in one cluster (chaining effect), while **Complete and Average linkage** produce more balanced separation. **Results do depend on linkage method used.**

#### Top Genes Differentiating Healthy vs Diseased (PC1 Loadings):

| Rank | Gene Index | PC1 Loading | Abs Loading |
|------|-----------|-------------|-------------|
| 1 | **Gene 599** | **-0.1142** | **0.1142** |
| 2 | Gene 583 | -0.1104 | 0.1104 |
| 3 | Gene 548 | -0.1083 | 0.1083 |
| 4 | Gene 539 | -0.1081 | 0.1081 |
| 5 | Gene 501 | -0.1075 | 0.1075 |
| 6 | Gene 581 | -0.1065 | 0.1065 |
| 7 | Gene 564 | -0.1063 | 0.1063 |
| 8 | Gene 567 | -0.1062 | 0.1062 |
| 9 | Gene 528 | -0.1042 | 0.1042 |
| 10 | Gene 598 | -0.1038 | 0.1038 |

> **Gene 599** shows the highest absolute PC1 loading (**0.1142**) — making it the most differentially expressed gene between healthy and diseased patients. All top 10 genes have negative loadings, suggesting they are systematically downregulated in one patient group.

---

## 💡 Business Insights

1. **Always Scale Before Clustering:** Without scaling, features with large magnitudes (like Assault: 45–337) dominate distance calculations over features with small values (like Murder: 0.8–17.4). Scaled clustering reduces Cluster 0 from 16 to 8 states — a fundamentally different grouping with real policy implications.

2. **K-Means on PCA Components = Same Result, Much Faster:** Clustering on the first 2 PCA components (2 features) gives identical results to clustering on all 50 raw features — confirming PCA is a reliable pre-processing step that dramatically reduces computational cost for high-dimensional clustering tasks.

3. **Matrix Completion Scales Predictably:** Missing data error grows roughly linearly from ~7% (5% missing) to ~15% (30% missing) — giving data engineers a reliable estimate of imputation quality degradation for production ML pipelines.

4. **Gene Discovery via PCA Loadings:** The top 10 differentially expressed genes (by absolute PC1 loading) provide a ranked shortlist for experimental validation — a practical, computationally efficient approach to biomarker discovery without requiring labeled training data.

5. **Linkage Choice Matters in Biology:** Single linkage chaining effect groups 39/40 patients into one cluster — completely defeating the clustering purpose. Complete or Average linkage is required for meaningful separation in gene expression data.

---

## 🗂️ File Structure

```
Chapter_12_Applied_Exercise_Solutions/
│
├── Chapter_12.ipynb          ← Main analysis notebook (all exercises)
├── Chapter_12.html           ← Rendered HTML version (easy browser viewing)
├── Chapter_12.qmd            ← Quarto source file
├── Ch12Ex13.csv              ← Gene expression dataset (40×1000)
└── README.md                 ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP scikit-learn scipy pandas numpy matplotlib

# Launch notebook
jupyter notebook Chapter_12.ipynb
```

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
*An Introduction to Statistical Learning with Applications in Python.* Springer.
Chapter 12: Unsupervised Learning — Applied Exercises 7–13.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM) provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shadalishah)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*
