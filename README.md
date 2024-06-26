# Domain_Adaptation_GNN_HIV
Explainable Artificial Intelligence and Domain Adaptation for Predicting HIV Infection with Graph Neural Networks

# Methods
* Logistic Regression (LR)
* Random Forest (RF)
* Graph Attention Networks (GAT)

# Commands to Run Code

**Baseline Machine Learning (LR + RF) on individual cities:**

* run_chicago_convent.sh

* run_houston_convent.sh <br/><br/>

**Graph Attention Network (GAT) on individual cities and GNNExplainer:**

* Gnnexplainer.ipynb (plots for Figure 2 are generated in this file) <br/><br/>

**Graph Neural Network (GNN) Domain Adaptation** 

  **(Houston is source city and Chicago is target city):**

   * LR/RF: run_houston_transfer_lr_rf.sh

   * GAT: run_houston_transfer_gat.sh

  **(Chicago is source city and Houston is target city):**

   * LR/RF: run_chicago_transfer_lr_rf.sh

   * GAT: run_chicago_transfer_gat.sh

# Implementation

## Packages

**For GAT on individual cities and GNNExplainer**:

* Python 3.8+, torch 2.0.1+, and corresponding versions of scikit-learn, pandas, and numpy

**For others (baseline LR/RF and all domain adaptation models)**:

* Python 3.6+, Tensorflow 1.14+ and corresponding versions of scipy, scikit-learn, numpy,


