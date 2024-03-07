---
id: 7sh79cb52sq09lnckkvh55m
title: Linear Regression
desc: ''
updated: 1707797829360
created: 1707797824154
---

To build a strong foundation in regression analysis for interviews or general understanding, consider covering the following topics:

1. **Simple Linear Regression:**
   - Definition and concept
   - Assumptions of linear regression
   - Method of least squares
   - Interpretation of coefficients
   - Calculation of residuals
   - Evaluation of model fit (R-squared, adjusted R-squared)

2. **Multiple Linear Regression:**
   - Extension of simple linear regression to multiple predictors
   - Interpretation of coefficients in the multiple regression context
   - Adjusted R-squared and model fit assessment
   - Collinearity issues and remedies (e.g., variance inflation factor)

3. **Model Building and Variable Selection:**
   - Forward selection, backward elimination, and stepwise regression
   - Criteria for variable selection (e.g., AIC, BIC, Mallows' Cp)
   - Cross-validation techniques for model validation

4. **Assumptions and Diagnostics:**
   - Normality of residuals
   - Homoscedasticity (constant variance of residuals)
   - Independence of errors
   - Outlier detection and influential observations
   - Residual plots and diagnostics tests (e.g., Shapiro-Wilk test, Durbin-Watson statistic)

5. **Non-linear Regression:**
   - Polynomial regression
   - Generalized Additive Models (GAMs)
   - Transformation techniques (e.g., log transformation, Box-Cox transformation)

6. **Interaction and Dummy Variables:**
   - Incorporating interaction terms in regression models
   - Encoding categorical variables using dummy variables
   - Interpretation of coefficients in the presence of interaction terms

7. **Time Series Regression:**
   - Autocorrelation and its implications in time series data
   - Autoregressive (AR), Moving Average (MA), and Autoregressive Integrated Moving Average (ARIMA) models
   - Seasonal decomposition and forecasting techniques

8. **Robust Regression Methods:**
   - Robust regression techniques for handling outliers and influential observations (e.g., Huber regression, M-estimators)

9. **Generalized Linear Models (GLMs):**
   - Extension of linear regression to non-normal response variables
   - Logistic regression for binary outcomes
   - Poisson regression for count data

10. **Applications and Case Studies:**
    - Real-world applications of regression analysis in various fields (e.g., economics, healthcare, marketing)
    - Hands-on experience with regression analysis using statistical software (e.g., R, Python with libraries like Statsmodels, scikit-learn)

Understanding these topics will provide you with a comprehensive understanding of regression analysis, preparing you for interviews and enabling you to build robust regression models for data analysis and prediction tasks.


Algorithm from scatch:
---
---

```py
import numpy as np

def linear_regression(x, y):
  """
  This function performs simple linear regression on the given data.

  Args:
    x: A 1D NumPy array of independent variables.
    y: A 1D NumPy array of dependent variables.

  Returns:
    A tuple containing the slope (m) and intercept (b) of the regression line.
  """
  # Calculate mean of x and y
  mean_x = np.mean(x)
  mean_y = np.mean(y)

  # Calculate numerator and denominator for slope
  numerator = np.sum((x - mean_x) * (y - mean_y))
  denominator = np.sum((x - mean_x) ** 2)

  # Calculate slope and intercept
  slope = numerator / denominator
  intercept = mean_y - slope * mean_x

  return slope, intercept

# Example usage
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m, b = linear_regression(x, y)

print(f"Slope: {m:.4f}, Intercept: {b:.4f}")
```

R-squared value calculation
---

R-squared, also known as the coefficient of determination, is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. To calculate the R-squared value in linear regression, you can follow these steps:

1. **Calculate the mean** of the observed data (y-values).
2. **Calculate the total sum of squares (SST)**, which is the sum of the squared differences between each observed value and the mean of the observed values.
3. **Calculate the sum of squares of residuals (SSR)**, which is the sum of the squared differences between the observed values and the values predicted by the linear regression model.
4. **Calculate the sum of squares explained by the regression (SSE)**, which is the difference between SST and SSR.
5. **Calculate the R-squared value** using the formula:

   $[ R^2 = 1 - \frac{SSR}{SST} ]$

   or, equivalently,

   $[ R^2 = \frac{SSE}{SST} ]$

Here's the step-by-step calculation in more detail:

1. **Mean of observed data (y-values):**
   
   $[ \bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i ]$

   where \( n \) is the number of observations and \( y_i \) is each individual observed value.

2. **Total sum of squares (SST):**
   
   $[ SST = \sum_{i=1}^{n} (y_i - \bar{y})^2 ]$

3. **Sum of squares of residuals (SSR):**
   
   $[ SSR = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 ]$

   where \( \hat{y}_i \) is the predicted value from the regression line for the ith observation.

4. **Sum of squares explained by regression (SSE):**
   
   $[ SSE = SST - SSR ]$

5. **R-squared value:**

   $[ R^2 = 1 - \frac{SSR}{SST} ]$

   or

   $[ R^2 = \frac{SSE}{SST} ]$

A higher R-squared value means a better fit between your regression model and the observed data. An R-squared value of 1 indicates a perfect fit, while an R-squared of 0 indicates that the model does not explain any of the variability of the response data around its mean.

It's also important to note that while a higher R-squared value generally indicates a better model fit, it doesn't necessarily mean the model is good. It's possible to have a high R-squared for a model that doesn't truly represent the underlying data structure, particularly if the model is overfitted. Therefore, R-squared should be one of several statistics you use to evaluate the quality of your regression model.


Adjuste R-Square
---

Adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. It is calculated using the following formula:

$Adj(R^2) = 1 - [\frac{(1 - R^2) * (n - 1)} {(n - k - 1)}]$

where:
- R-squared is the coefficient of determination
- n is the sample size
- k is the number of predictors in the model

To calculate the adjusted R-squared, you first need to calculate the R-squared value for your regression model. Then, you can use the above formula to adjust the R-squared value for the number of predictors in your model. The resulting value will be the adjusted R-squared.


Dummy Coding
---



Dummy coding is a method used in regression analysis to handle categorical variables. Categorical variables are those that can be divided into multiple categories but have no order or priority. Examples include gender (male, female), marital status (single, married, divorced), or education level (high school, college, graduate). In regression analysis, these categorical variables need to be converted into a form that can be used to estimate the model parameters. Dummy coding is one of the most common methods for this conversion.

### How Dummy Coding Works

1. **Creation of Dummy Variables**: For each category of the categorical variable, a new binary (0/1) variable is created. Each of these binary variables is known as a dummy variable. The number of dummy variables created is equal to the number of categories minus one.

2. **Assignment of Values**: Each dummy variable is assigned a value of 1 for the category it represents and 0 for all other categories. For example, if you have a categorical variable "Gender" with categories "Male" and "Female", you would create two dummy variables: "Gender_Male" and "Gender_Female". "Gender_Male" would be 1 for all male observations and 0 for all female observations, and vice versa for "Gender_Female".

3. **Inclusion in the Model**: The dummy variables are included in the regression model as independent variables. The interpretation of the coefficients for these dummy variables is that they represent the difference in the dependent variable's expected value between the category represented by the dummy variable and the reference category (usually the first category in the list).

### Example

Suppose you have a dataset with a categorical variable "Education" with three categories: "High School", "College", and "Graduate". You would create two dummy variables:

- **Education_HighSchool**: 1 for "High School" and 0 for "College" and "Graduate".
- **Education_College**: 1 for "College" and 0 for "High School" and "Graduate".

If you include these dummy variables in your regression model, the coefficient for "Education_HighSchool" would represent the difference in the dependent variable's expected value between "High School" and "College" (assuming "High School" is the reference category).

### Choosing a Reference Category

The choice of the reference category (the category that is represented by 0 in the dummy variables) is arbitrary but can affect the interpretation of the coefficients. It's common to choose the reference category based on the order of the categories or the most common category.

### Advantages of Dummy Coding

- **Simplicity**: It simplifies the model by converting categorical variables into a form that can be easily included in regression analysis.
- **Interpretability**: It allows for the interpretation of the effect of each category on the dependent variable.

### Limitations

- **Dummy Variable Trap**: If not handled correctly, including all categories as dummy variables can lead to perfect multicollinearity, making the model unstable and the estimates unreliable. This is often addressed by dropping one dummy variable as the reference category.
- **Assumption of Independence**: Like all regression models, dummy coding assumes that the independent variables are independent of each other, which may not always be the case.

Dummy coding is a fundamental technique in regression analysis for handling categorical variables, enabling the inclusion of such variables in the model and facilitating the interpretation of the model's results.