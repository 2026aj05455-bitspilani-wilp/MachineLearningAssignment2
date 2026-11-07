<<<<<<< HEAD
## a. Problem statement

Build and compare five supervised classification approaches for predicting the variety of a dry bean from measurements extracted from its image. The target has seven classes. This project scaffold is designed to help you implement Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine, evaluate each on one held-out test set, and present the results in an interactive Streamlit app.

## b. Dataset description

The **Dry Bean** dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean). UCI describes 13,611 observations, 16 numeric features, and seven bean varieties. The features are geometric measurements extracted from bean images. The target column is `Class`; the feature names are preserved as provided, including `AspectRation` and lower-case `roundness`. UCI reports no missing values. `training_data.csv` and `test_data.csv` are a stratified 80/20 split of the UCI workbook (seed 42).


## c. GitHub Repository Link




### d. Models Used.

                 Model  Accuracy  AUC (OvR macro)  Precision (macro)  Recall (macro)  F1 (macro)      MCC
   Logistic Regression  0.921410         0.994776           0.935383        0.932149    0.933538 0.905045
         Decision Tree  0.897907         0.957034           0.912674        0.912018    0.912228 0.876574
                   KNN  0.916636         0.985961           0.931825        0.927141    0.929237 0.899205
           Naive Bayes  0.763863         0.967193           0.774427        0.769417    0.767750 0.715406
Support Vector Machine  0.922145         0.994929           0.935937        0.932952    0.934297 0.905870


### e. Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved strong and balanced performance, with 92.14% accuracy and a macro F1 score of 0.9335. Its OvR macro AUC of 0.9948 indicates strong class ranking. |
| Decision Tree | Performed reasonably well, reaching 89.79% accuracy and a macro F1 score of 0.9122. Its scores were lower than those of Logistic Regression, KNN, and SVM on this test set. |
| KNN | Delivered strong results, with 91.66% accuracy and a macro F1 score of 0.9292. Its overall performance was close to Logistic Regression, but slightly lower across the reported metrics. |
| Naive Bayes | Had the lowest accuracy (76.39%) and macro F1 score (0.7678). Its OvR macro AUC was higher at 0.9672, suggesting stronger class ranking than its final class predictions. |
| Support Vector Machine | Achieved the highest scores across the reported metrics: 92.21% accuracy, 0.9949 OvR macro AUC, 0.9360 macro precision, 0.9330 macro recall, 0.9343 macro F1, and 0.9059 MCC. Its results were only slightly higher than Logistic Regression. |
| Overall, winner for your dataset? | Support Vector Machine was the top performer by a narrow margin. Logistic Regression was a close second. |


##### Data split, model settings, and evaluation

The experiments use the UCI Dry Bean dataset, which contains 13,611 rows, 16 input features, and the target column `Class`. The data was divided into a stratified 80/20 training and test split using `random_state=42`. This produced 10,888 training rows and 2,723 held-out test rows. The class proportions were preserved across the split. Models were fitted using `training_data.csv` and evaluated using `test_data.csv`; the test data was not used to fit models or preprocessing.

The following scikit-learn settings were used:

| Model | Settings and preprocessing |
|---|---|
| Logistic Regression | `max_iter=2000`, `random_state=42`; `StandardScaler` was fitted within the training pipeline. Other parameters use scikit-learn defaults. |
| Decision Tree | `max_depth=12`, `random_state=42`. Other parameters use scikit-learn defaults. |
| KNN | `n_neighbors=7`, `weights="distance"`; `StandardScaler` was fitted within the training pipeline. Other parameters use scikit-learn defaults. |
| Naive Bayes | Gaussian Naive Bayes (`GaussianNB`) with scikit-learn defaults. |
| Support Vector Machine | RBF kernel, `C=1.0`, `probability=True`, `random_state=42`; `StandardScaler` was fitted within the training pipeline. Other parameters use scikit-learn defaults. |
=======
# MachineLearningAssignment2
This is code repository for Machine learning Assignment 2.
>>>>>>> 0043ef9c6f6ef931a40110e9d354371989a9718f
