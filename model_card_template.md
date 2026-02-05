# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* **Person or organization developing model**: Andrew Picart
* **Model date**: February 2026
* **Model version**: 1.0.0
* **Model type**: Random Forest Classifier
* **Paper or other resource for more information**: Udacity Machine Learning DevOps Engineer Nanodegree
* **License**: MIT
* **Questions about the model**: Contact andrewpicart via GitHub

## Intended Use
This model is primarily designed to classify whether an individual earns more than $50,000 USD per year based on census data features. It is intended for use by researchers and data scientists interested in demographic income analysis. Please note that this model is out of scope for automated decision-making systems (such as loan approval or hiring) without further fairness and bias evaluations.

## Training Data
The model was trained on the Census Income Data Set obtained from the UCI Machine Learning Repository. The dataset contains 32,561 instances and includes attributes such as age, education, marital status, race, sex, and native country. Before training, we processed the data using a custom pipeline that applied One-Hot Encoding to categorical variables and Label Binarization to the target variable.
  
## Evaluation Data
For evaluation, we reserved a 20% split of the original dataset. This test data was processed using the same One-Hot Encoder and Label Binarizer pipelines fitted to the training data to ensure consistent feature transformation.
  
## Metrics
The model was evaluated using Precision, Recall, and F1-score. The performance on the test set is as follows:
* **Precision**: 0.7441
* **Recall**: 0.6218
* **F1 Score**: 0.6774

## Ethical Considerations
Users should be aware that the dataset, sourced from the 1994 Census, reflects the socioeconomic biases of that time. Because sensitive features like `race` and `sex` are included, the model risks propagating these historical biases. We strongly recommend analyzing performance disparities across demographic groups (as done in `slice_output.txt`) before any real-world application.

## Caveats and Recommendations
The most significant limitation is the data's age; economic shifts and inflation over the last 25 years mean the target income threshold may no longer be relevant. Furthermore, this model currently uses default hyperparameters. We recommend using techniques like GridSearch to tune these parameters for potentially better performance.