# Machine-Learning-Workflow-for-Imbalanced-Data-Classification

Machine Learning Workflow 1: (1) apply the pre-processing methods: a) Oversampling, b) Polynomial feature construction, c) Information gain feature selection and d) Genetic algorithm feature selection; then, (2) optimizes the hyper-parameters of Random Forest (number of trees and max depth of each tree) and Logistic Regression (regularization) using grid search with K-fold Cross-Validation; finally, 3) trains Random Forest and Logistic Regression using the optimized hyper-parameters. The logistic regression and random forest classifers of Weka machine learning tool are called by workflow 1. 

Machine Learning Workflow 2: 1) apply the pre-processing methods: a) Oversampling, b) Polynomial feature construction, c) Information gain feature selection and d) Genetic algorithm feature selection; then, (2) optimizes the hyper-parameters of Neual Networks, Gaussian Process, SVM, XGBoost or Random Forest; finally, 3) train a classifier and select an optimal classification threshold using the training set. The pre-processing methods of workflow 1 are called by workflow 2. The Neual Networks, Gaussian Process, SVM, XGBoost and Random Forest classifiers of sklearn tool are called by workflow 2. 

Programs to test the performance of the trained classifiers on a testset.

To handle an imbalanced training data (one class is heavily underrepresented compared to the other class) with irrelevant and/or redundant features, the pre-processing methods a), b), c) and d) output a dimensionality-reduced balanced training data with the most informative (relevant) polynomial features only; then, k-fold CV is used to find optimal hyper-parameters of the classifier(s); finally, the classifier(s) with their optimal hyper-parameters are trained. 


