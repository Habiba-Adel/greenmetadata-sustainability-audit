# GreenMetaData Sustainability Audit

A sustainability audit of the scikit-learn classifier comparison workflow,

## Project Overview

This repository contains a reproducible sustainability audit of the
`plot_classifier_comparison.py` example from the scikit-learn library.
The audit measures the energy consumption of 10 machine learning classifiers
across 3 datasets, identifies energy hotspots, and proposes green software
redesign improvements.

**Audited project:** scikit-learn — https://github.com/scikit-learn/scikit-learn  
**Target script:** `examples/classification/plot_classifier_comparison.py`  
**Carbon measurement tools:** CodeCarbon + Green Algorithms Calculator  
**Green patterns reference:** https://patterns.greensoftware.foundation/catalog/ai/

---

## Repository Structure
```
greenmetadata-sustainability-audit/
├── audit/
│   ├── classifier_energy_audit.py   ← main instrumentation script
│   └── results/
│       └── emissions.csv            ← CodeCarbon measurement results
└── README.md
```

---

## How to Reproduce

### 1. Clone this repository
```bash
git clone https://github.com/Habiba-Adel/greenmetadata-sustainability-audit.git
cd greenmetadata-sustainability-audit
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install scikit-learn codecarbon matplotlib pandas
```

### 4. Run the audit script
```bash
python audit/classifier_energy_audit.py
```

Results will be saved to `audit/results/emissions.csv` and printed to the terminal.

---

## What the Script Does

The script measures the energy consumption of each classifier's `fit()` step
individually using CodeCarbon. It records:
- Training duration per classifier per dataset
- CO₂ emissions (kg) per classifier per dataset  
- Accuracy score per classifier per dataset

Results are averaged across 3 datasets and sorted by energy consumption
to reveal the actual energy hotspots.

---

## Key Findings

### Energy Hotspots Identified

| Classifier | Avg Duration | Energy Verdict |
|-----------|-------------|----------------|
| Neural Network | 0.116s | 🔴 Hotspot |
| Gaussian Process | 0.086s | 🔴 Hotspot |
| AdaBoost | 0.035s | 🔴 Hotspot |
| Random Forest | 0.011s | 🟡 Medium |
| Linear SVM | 0.002s | 🟢 Efficient |
| RBF SVM | 0.001s | 🟢 Efficient |
| Decision Tree | 0.001s | 🟢 Efficient |
| Naive Bayes | 0.001s | 🟢 Efficient |
| Nearest Neighbors | 0.001s | 🟢 Efficient |
| QDA | 0.001s | 🟢 Efficient |

> **Note:** CodeCarbon reported nearly identical CO₂ values for all classifiers
> due to its 15-second default sampling interval most classifiers complete
> in under 0.1s. Training duration was therefore used as the primary energy
> proxy. This is a documented CodeCarbon limitation for very short ML tasks.

### Key Insight
The most energy expensive classifiers are not the most accurate ones.
Nearest Neighbors (0.001s, accuracy 0.967) outperforms Neural Network
(0.116s, accuracy 0.925) in both energy efficiency and accuracy.

---

## Carbon Footprint Estimation

Two tools were used and compared:

| Tool | CO₂ Result | Method |
|------|-----------|--------|
| CodeCarbon | ~0.000004 kg | Real measurement during execution |
| Green Algorithms Calculator | ~0.00362 kg | Model-based estimation |

The large difference highlights the measurement standardization challenge
that GreenMetaData aims to address  two approved tools produce very
different results for the same workflow depending on methodology and
input assumptions.

> ML CO₂ Impact Calculator was attempted but ruled out: it supports GPU
> hardware only (no CPU option) and failed to return results after more than 2 hours.

---

## Green Software Patterns Applied

From https://patterns.greensoftware.foundation/catalog/ai/:

1. **Use energy efficient models** : replace hotspot classifiers with
   efficient alternatives when accuracy is equal or better
2. **Optimize model size** : add early stopping to Neural Network
   (`early_stopping=True`, `n_iter_no_change=10`)
3. **Select right hardware type** : match hardware to workload:
   AdaBoost is sequential, Random Forest parallelises naturally

---

## Redesign Proposal Summary

| Proposal | Estimated Energy Saving |
|---------|------------------------|
| Skip unnecessary classifiers via accuracy threshold | ~93% of total training time |
| Neural Network early stopping | ~40-60% of NN training time |
| Carbon-aware job scheduling | Up to 50% CO₂ reduction with same energy |

---

## Hardware & Environment

| Parameter | Value |
|-----------|-------|
| CPU | Intel Core i7-9850H @ 2.60GHz |
| Cores | 12 |
| RAM | 32 GB |
| OS | Linux |
| Python | 3.x (virtual environment) |
| Location | Cairo, Egypt |

---

## Dependencies
```
scikit-learn
codecarbon
matplotlib
pandas
```

---

## References

- Pedregosa et al. (2011) — scikit-learn: https://jmlr.org/papers/v12/pedregosa11a.html
- CodeCarbon documentation: https://docs.codecarbon.io/latest/
- Green Algorithms Calculator: https://calculator.green-algorithms.org/
- Green Software Patterns: https://patterns.greensoftware.foundation/catalog/ai/
- RSQKit Environmental Sustainability: https://everse.software/RSQKit/improving_environmental_sustainability
- ErUM-Data Workshop Report: https://arxiv.org/pdf/2602.24087
- AdaBoost vs Random Forest (UW CSE): https://courses.cs.washington.edu/courses/cse416/22su/lectures/8/lecture_8.pdf
