# Machine-Learning-Workflow-for-Imbalanced-Data-Classification

Machine Learning Workflow 1: (1) apply the pre-processing methods: a) Oversampling, b) Polynomial feature construction, c) Information gain feature selection and d) Genetic algorithm feature selection; then, (2) optimizes the hyper-parameters of Random Forest (number of trees and max depth of each tree) and Logistic Regression (regularization) using grid search with K-fold Cross-Validation; finally, 3) trains Random Forest and Logistic Regression using the optimized hyper-parameters. The logistic regression and random forest classifers of Weka machine learning tool are called by workflow 1. Interface module: main.py

Machine Learning Workflow 2: 1) apply the pre-processing methods: a) Oversampling, b) Polynomial feature construction, c) Information gain feature selection and d) Genetic algorithm feature selection; then, (2) optimizes the hyper-parameters of Neual Networks, Gaussian Process, SVM, XGBoost or Random Forest; finally, 3) train a classifier and select an optimal classification threshold using the training set. The pre-processing methods of workflow 1 are called by workflow 2. The Neual Networks, Gaussian Process, SVM, XGBoost and Random Forest classifiers of sklearn tool are called by workflow 2. 
Interface module: pipeline.py

Machine Learning Workflow 3: 1) Train and validate multiple neural networks (MLPs and DNNs) of different topologies using Tensorflow in parallel on multiple CPU cores (a network is validated on a validation set 
while it is being trained in order to determine which epoch to stop training); 2) Select an optimal network among all the networks trained based on their validation performances; finally, 3) Evaluate the performance of the selected neural network in step 2 on a test set. Data pre-processing methods: normalization of input features, oversampling/undersampling (optional) and genetic algorithm feature selection (optional) can be performed on the training set before trainig neural networks to improve their performances. Interface modules: workflow_nn.py (to run on a PC with multiple CPU cores), slurm_script.sh (to run on a HPC using SLURM scheduler)

Programs to test the performance of the trained classifiers on a testset.

To handle an imbalanced training data (one class is heavily underrepresented compared to the other class) with irrelevant and/or redundant features, the pre-processing methods a), b), c) and d) output a dimensionality-reduced balanced training data with the most informative (relevant) polynomial features only; then, k-fold CV is used to find optimal hyper-parameters of the classifier(s); finally, the classifier(s) with their optimal hyper-parameters are trained. 


Reference:

D. Tian, ZQ. Lang, D. Zhang and D. O Anumba “A Filter-Predictor Polynomial Feature Based Machine Learning Approach for Preterm Birth Prediction from Electrical Impedance Spectroscopy”, Journal of Biomedical Signal Processing and Control, vol. 80, 2023

