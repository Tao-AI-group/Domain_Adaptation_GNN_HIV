# Domain_Adaptation_GNN_HIV
Explainable Artificial Intelligence and Domain Adaptation for Predicting HIV Infection with Graph Neural Networks

# Methods:
* Logistic Regression (LR)
* Random Forest (RF)
* Graph Attention Networks (GAT)

# Commands to Run Code:

**Baseline Machine Learning (LR + RF) on individual cities:**

* run_chicago_convent.sh

* run_houston_convent.sh


**Graph Attention Network (GAT) on individual cities and GNNExplainer:**

* Gnnexplainer.ipynb (plots for Figure 2 are generated in this file)


**Graph Neural Network (GNN) Domain Adaptation (Houston is source city and Chicago is target city):**

* LR/RF: run_houston_transfer_lr_rf.sh

* GAT: run_houston_transfer_gat.sh

**(Chicago is source city and Houston is target city):**

* LR/RF: run_chicago_transfer_lr_rf.sh

* GAT: run_chicago_transfer_gat.sh


