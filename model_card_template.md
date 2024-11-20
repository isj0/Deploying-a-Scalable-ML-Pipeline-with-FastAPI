# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Name**: Census Income prediction model
- **Developer**: Iqbal Jassal
- **Model Version**: 1.0
- **Date**: 11/19/2024
- **Model Type**: Random Forest Classifier
- **Training Algorithm**: Random Forest with 100 estimators, random state set to 42.
- **License**: This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.


## Intended Use

This model is intended to predict whether an individual's income exceeds $50K/yr based on census data.

## Training Data

The model was trained on the Census Income dataset downloaded from the UCI Machine Learning Repository.

## Evaluation Data

The evaluation data was available in the original Census income dataset. Also, 20% of the data is allocated to the test set.

## Metrics

The following are the results of metrics that were used to evaluate the performance of the model:

- **Precision**: 0.7419 
  - The model predicts an income greater than $50K approximately 74.19% of the time.
  
- **Recall**: 0.6384 
  - Shows that the model correctly identifies approximately 63.84% of individuals who earn more than $50K.
  
- **F1-score**: 0.6863 
  - The overall measure of the model's accuracy is 68.63%.

## Ethical Considerations

The attributes like race, sex, martial status used in the model can lead to biased predictions and discrepancies.

## Caveats and Recommendations

The dataset used for training this model was extracted by Barry Becker from the 1994 Census database. The demographic and socioeconomic conditions have changed significantly changed in the past 30 years and the model will not be applicable to current populations. Hence, this dataset must not be used to generalize today's population. 
