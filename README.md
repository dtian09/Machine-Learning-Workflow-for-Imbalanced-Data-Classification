# Machine-Learning-Workflow-for-Imbalanced-Data-Classification

Python programs to:

I) Build machine learning workflows that (1) apply the pre-processing methods: a) Oversampling, b) Polynomial feature construction, c) Information gain feature selection and d) Genetic algorithm feature selection; then, (2) optimizes the hyper-parameters of Random Forest and Logistic Regression using K-fold Cross-Validation; finally, 3) trains Random Forest and Logistic Regression using the optimized hyper-parameters.

II) Test the performance of the trained classifiers on a testset.

To handle an imbalanced training data (one class is heavily underrepresented compared to the other class) with irrelevant and/or redundant features, the pre-processing methods a), b), c) and d) output a dimensionality-reduced balanced training data with the most informative (relevant) polynomial features only; then, k-fold CV is used to find optimal hyper-parameters (number of trees and max depth of each tree) of random forest and optimal hyper-parameter (regularization) of logistic regression; finally, a random forest and a logistic regression with their optimal hyper-parameters are trained. 


