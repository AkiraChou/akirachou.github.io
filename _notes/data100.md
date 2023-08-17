---
title: "Notes DATA100"
collection: notes
permalink: /notes/data100
date: 2023-05-15
---

DATA100: Principles and Techniques of Data Science. I took this class during spring 2023 with Lisa Yan and Narges Norouzi. The following notes are taken from the class [course notes](https://ds100.org/sp23/).

## Examining Datasets: the Data Science Lifecycle, Programming Tools, Visualization Techniques, and Data Cleaning Methods

### **Data Science Lifecycle**

- **Ask a Question:** Data scientists start by asking relevant questions, such as predicting business profits or analyzing medical treatment benefits. Clear questions ensure project clarity and define success metrics.
- **Obtain Data:** Obtaining necessary data is crucial, either from existing sources or by collecting it. Considerations include defining data units, necessary features, sampling methods, and ensuring data represents the target population.
- **Understand the Data:** Data analysis involves investigating data patterns and relationships. Understanding data organization, relevance, biases, anomalies, and transforming data for effective analysis are important tasks. Exploratory data analysis and visualization aid in this stage.
- **Understand the World:** Using observed data patterns, data scientists answer questions through prediction or inference. The reliability of conclusions and predictions is crucial, requiring robust models. If needed, further analysis may lead to new questions and data collection.

### **Pandas**

- Pandas is a data analysis library in Python.
- It's designed for working with tabular data.
- Three fundamental pandas data structures: Series, DataFrame, and Index.
    - **Series**: 1D labeled array, similar to a column in DataFrame.
    - **DataFrame**: 2D tabular data with rows and columns.
    - **Index**: Sequence of row/column labels shared by Series and DataFrame.

### Data Cleaning and EDA

**File Format**

- File formats define how data is encoded and stored. Common formats include CSV (Comma-Separated Values) and TSV (Tab-Separated Values). JSON (JavaScript Object Notation) files are also used.

**Variable Types**

- Variables can be quantitative (numeric) or qualitative (categorical). Quantitative variables can be continuous (measured on a continuous scale) or discrete (taking finite values). Qualitative variables include ordinal (ordered levels) and nominal (no specific order) categories.

**Granularity, Scope, and Temporality**

- Granularity refers to the level of detail represented by a single data row. Scope is the subset of the population covered, and temporality refers to the time period during which the data was collected.

**Faithfulness**

- Data used for analysis should accurately represent the real world. Signs of data issues include unrealistic values, violations of dependencies, errors in data entry, and more. Missing data can be addressed through imputation techniques, but the reason for missing values should be considered.

### Regular Expressions

**Why Work with Text?**

1. Canonicalization: Convert data that has multiple formats into a standard form.
    - By manipulating text, we can join tables with mismatched string labels
2. Extract information into a new feature.
    - For example, we can extract date and time features from text

**Regex Basics**

- Sequence of characters that specifies a search pattern
- Regular expressions are used to extract specific information from text.

**Limitations of Regular Expressions**

- They can be difficult to write, read, and debug.
- Unsuitable for parsing hierarchical structures.
- Unsuitable for complex features and properties

### Visualizations

**Goals of Visualization**

1. Broaden understanding of data through exploratory data analysis.
2. Communicate results and conclusions effectively, where visualization theory is important

**Overview of Distributions**

- **Distributions**: A distribution represents the frequency of unique values in a variable. Distributions must fulfill two properties: each data point belongs to only one category, and the total frequency equals 100% or the number of values considered.
- **Bar Plots**: Bar plots are common for displaying the distribution of qualitative variables
- **Histograms**: Histograms visualize the distribution of quantitative data. Helps analyze distribution shape, skewness, tails, outliers, and modes.
- **Density Curves**: Density curves are smooth curves that approximate continuous distributions in histograms.
- **Box Plots and Violin Plots**: Box plots visualize numerical distribution characteristics such as quartiles, IQR, whiskers, and outliers. Violin plots combine box plots with density curves to show relative frequency.
- **Ridge Plots**: Ridge plots display multiple density curves offset from each other to minimize overlap.

**Kernel Density Functions**

- KDE estimates a density curve from data using kernels.
1. Place a kernel at each data point.
2. Normalize kernels to have total area of 1.
3. Sum kernels together

**Transformations**

- Transformations manipulate data to find relationships.
- Logarithmic transformation spreads data.
- Linear transformations linearize relationships

**Visualization Theory**

- The following are powerful tools in visualization: axis, color, markings, conditioning (process of comparing data that belong to separate groups), context (informative titles, axis labels, and descriptive captions)

### Sampling

**Censuses and Surveys**

- A census is an official count or survey of a population, recording details of individuals.
- Surveys involve asking questions to a subset of a population, used to make inferences about the population
- Population: The group that you want to learn something about
- Sampling Frame: The list from which the sample is drawn
- Sample: Who you actually end up sampling.

**Bias**

- **Selection bias**: systematically excludes (or favors) particular groups.
- **Response bias**: occurs because people don’t always respond truthfully. Survey designers pay special detail to the nature and wording of questions to avoid this type of bias.
- **Non-response bias**: occurs because people don’t always respond to survey requests, which can skew responses.
- A convenience sample is not necessarily representative and can be biased.

**Probability Samples**

- In a probability sample, we know the chance any given set of individuals will be in the sample.
- **random sample with replacement**: a sample drawn uniformly at random with replacement
- **simple random sample (SRS)**: a sample drawn uniformly at random without replacement
- **stratified random sample**: random sampling is performed on strata (specific groups), and the groups together compose a sample

**Approximating Simple Random Sampling**

- In situations with a large population and small sample, random sampling with and without replacement are similar.
- Multinomial probabilities are used for sampling a categorical distribution at random *with replacement*.

## Using Data to Better Understand the World: Predictive Modeling

### Introduction to Modeling

**What is a Model?**

- A model is an idealized representation of a system. A system is a set of principles or procedures according to which something functions
- Reasons for building models
    - we care about creating models that are simple and interpretable, allowing us to understand what the relationships between our variables are
    - we care more about making extremely accurate predictions, at the cost of having an uninterpretable model
- models can be split into two categories:
    - Deterministic physical (mechanistic) models represent laws governing how the world works.
    - Probabilistic models attempt to understand how random processes evolve, often making simplifying assumptions

**Simple Linear Regression**

- the unique straight line that minimizes the mean squared error of estimation among all straight lines
    - slope: $$r \cdot \frac{\text{Standard Deviation of y}}{\text{Standard Deviation of x}}$$
    - y-intercept: $$\text{average of y} - \text{slope}\cdot\text{average of x}$$
    - $$r = \frac{1}{n} \sum_1^n (\frac{x_i - \bar{x}}{\sigma_x})(\frac{y_i - \bar{y}}{\sigma_y})$$
        - Correlation: $$r$$
        - Mean: $$\bar{x}$$
        - Standard Deviation: $$\sigma_x$$
        - Predicted value: $$\hat{x}$$

**The Modeling Process**

1. **Choose a model**: how should we represent the world?
    - $$\hat{y}_i = \theta_0 + \theta_1 x_i$$
        - features: $$x_i$$, predictions: $$\hat{y}_i$$, intercept term: $$\theta_0$$, slope term: $$\theta_1$$
2. **Choose a loss function**: how do we quantify prediction error?
    - loss function: characterizes the cost, error, or fit resulting from a particular choice of model or model parameters
    - *Squared loss**, also known as **L2 loss**, computes loss as the square of the difference between the observed $$y_i$$ and predicted $$\hat{y}_i$$:
        - $$L(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2$$
    - *Absolute loss**, also known as **L1 loss**, computes loss as the absolute difference between the observed $$y_i$$ and predicted $$\hat{y}_i$$:
        - $$L(y_i, \hat{y}_i) = |y_i - \hat{y}_i|$$
    - The **Mean Squared Error (MSE)** is the average squared loss across a dataset:
        - $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
    - The **Mean Absolute Error (MAE)** is the average absolute loss across a dataset:
        - $$\text{MAE}= \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
3. **Fit the model**: how do we choose the best parameters of our model given our data?
    - To find the optimal model parameter, we:
        1. Take the derivative of the cost function with respect to that parameter
        2. Set the derivative equal to 0
        3. Solve for the parameter
    - Results:
        - $$r \sigma_x \sigma_y = \hat{\theta}_1 \times \sigma_x^2$$
        - $$\hat{\theta}_1 = r \frac{\sigma_y}{\sigma_x}$$
4. **Evaluate model performance**: how do we evaluate whether this process gave rise to a good model?
    - three main ways: statistics, performance metrics, visualizations

### Constant Model, Loss, and Transformations**

**Constant Model**

- the constant model *predicts the same constant number every time,* $$\theta$$
- MSE: L2 (squared) loss
    - the best choice of $$\theta$$ is the **mean of the observed** $$y$$ **values**. In other words, $$\hat{\theta} = \bar{y}$$
- MAE: L1 (absolute loss)
    - the best choice of $$\theta$$ is the **median of the observed** $$y$$ **values**. In other words, $$\hat{\theta} = \text{median}(y)$$

**Comparing Loss Functions**

- Changing the loss function leads to different optimal model parameters.
- MAE is robust to outliers, while MSE is sensitive
- MSE has a unique solution for $$\hat{\theta}$$, MAE is not guaranteed to have a single unique solution
- RMSE (Root Mean Squared Error) is a common performance metric.
- Residual plots show the difference between observed and predicted values

**Linear Transformations**

- Linear transformations help handle non-linear data relationships.
- Applying a logarithmic transformation can linearize relationships.
- The Tukey-Mosteller Bulge Diagram is a useful tool for summarizing what transformations can linearize the relationship between two variables


### OLS

**Linearity**

- An expression is linear in a set of parameters if it is a linear combination of the elements of the set.
- To check if an expression is linear, see if it can separate into a matrix product of two terms: a vector of parameters and a matrix/vector not involving parameters
- Multiple Linear Regression extends simple linear regression to include multiple features
    - $$\hat{y}i = \theta_0\:+\:\theta_1x{i1}\:+\:\theta_2 x_{i2}\:+\:...\:+\:\theta_p x_{ip}$$
    - equivalent to the dot (scalar) product of the observation vector and parameter vector

**OLS Properties**

1. When using the optimal parameter vector, our residuals $$e = \mathbb{Y} - \hat{\mathbb{Y}}$$ are orthogonal to $$span(\mathbb{X})$$
2. For all linear models *with an intercept term*, the sum of residuals is zero.
3. The Least Squares estimate $$\hat{\theta}$$ is unique if and only if $$\mathbb{X}$$ is full column rank

### **Gradient Descent**

- Gradient descent is used to minimize (or sometimes maximize) an objective function, also known as a loss or cost function. The objective function measures how well a model's predictions match the actual data.
- Update Rule / Learning Rate: The learning rate (often denoted as $$\alpha$$) is a hyperparameter that determines the step size taken in each iteration of gradient descent. It controls the size of the update to the parameters
- Convergence: occurs when updates become smaller, and the algorithm reaches a point where further updates don't significantly change the parameter values.
- Batch Gradient Descent: In batch gradient descent, the gradient is calculated using the entire dataset in each iteration. This can be computationally expensive for large datasets.
- Stochastic Gradient Descent (SGD): In stochastic gradient descent, the gradient is calculated using only a single randomly chosen data point in each iteration. It's faster but can result in noisy updates.
- Mini-Batch Gradient Descent: Mini-batch gradient descent is a compromise between batch and stochastic gradient descent. It uses a small batch of data points for gradient calculation in each iteration.
- Convexity: Convex functions have a single global minimum, making gradient descent more reliable. However, non-convex functions may have multiple local minima, leading gradient descent to converge to different solutions depending on the starting point

### **Feature Engineering**

- Feature Engineering is the process of transforming the raw features into more informative features that can be used in modeling or EDA tasks.
- Capabilities:
    - Capture domain knowledge (e.g. periodicity or relationships between features).
    - Express non-linear relationships using simple linear models.
    - Encode non-numeric features to be used as inputs to models.
- Feature Function: takes a $$d$$ dimensional input and transforms it to a $$p$$ dimensional input. This results in a new, feature-engineered design matrix that we rename $$\Phi$$.
    - $$\mathbb{X} \in \mathbb{R}^{n \times d} \longrightarrow \Phi \in \mathbb{R}^{n \times p}$$
- One Hot Encoding: feature engineering technique that generates numeric features from categorical data, allowing us to use our usual methods to fit a regression model on the data
- Variance and Training Error:
    - training error: model’s error when generating predictions from the same data that was used for training purposes
    - model variance: sensitivity of the model to the data used to train
- Overfitting: Complex models may memorize training data and perform poorly on new data. Striking a balance between model complexity is important to avoid overfitting
    
    

### Cross Validation and Regularization

**The Holdout Method**

- Overfitting can occur when a model is too complex and fits the training data too closely. To avoid overfitting, the holdout method is used.
- The holdout method involves splitting the data into a training set and a validation set. The training set is used to train models, while the validation set is used to measure overfitting and choose the best model.

**K-Fold Cross Validation**

- K-Fold Cross Validation is a technique to assess the quality of a model's hyperparameters.
- The data is divided into k equally sized folds. A model is trained on k-1 folds and validated on the remaining fold, repeating for all k folds.
- The average validation error is used to evaluate the hyperparameters.
- It provides a more robust alternative to the holdout method and helps mitigate issues related to data variability.

**Test Sets**

- Test sets are used to evaluate the final performance of a chosen model. Test sets help provide an unbiased estimate of the model's performance on unseen data.
- A test set is separate from the training and validation sets and is never used during the model development process.

**Constrained gradient descent**

- Constrained gradient descent restricts the model's parameter space to a specific region. It helps control the complexity of the model and prevents overfitting.
- **L2 Regularization (Ridge Regression)**
    - L2 regularization constrains the sum of squared coefficients to a certain limit.
    - L2 regularization penalizes large coefficients..
    - Standardizing the data before applying L2 regularization helps ensure fair treatment of features with different scales.
    - Ridge Regression maintains non-zero coefficients for all features
- **L1 Regularization (Lasso Regression)**
    - L1 regularization constrains the sum of absolute coefficients to a certain limit.
    - Lasso Regression is effective at implicit feature selection, often reducing the importance of less relevant features. It does so by setting their respective feature weights to 0
    - Unlike Ridge Regression, Lasso Regression does not have a closed-form solution due to non-differentiability


## Quantitative Analysis: The B**ias-Variance Tradeoff**

### Random Variables

**Random Variables and Distributions**

- The distribution of a random variable $$X$$ describes how the total probability of 100% is split over all possible values that $$X$$ could take
- For any two random variables $$X$$ and $$Y$$
    - $$X$$ and $$Y$$ are equal if $$X(s) = Y(s)$$ for every sample $$s$$
    - $$X$$ and $$Y$$ are identically distributed if the distribution of $$X$$ is equal to the distribution of $$Y$$
    - $$X$$ and $$Y$$ are independent and identically distributed (IID) if 1) the variables are identically distributed and 2) knowing the outcome of one variable does not influence our belief of the outcome of the other

**Expectation and Variance**

- The expectation of a random variable $$X$$ is the weighted average of the values of
$$X$$, where the weights are the probabilities of each value occurring
- linearity of expectation: the expected value of a sum of random variables is the sum of the expected values of the variables
- variance of a random variable is a measure of its chance error
- Unlike expectation, variance is non-linear
- covariance of two random variables:  the expected product of deviations from expectation
- the variance of a sum of independent random variables is the sum of their variances:

**Standard Deviation**

- the standard deviation is equal to the square root of the variance

### **Estimators, Bias, and Variance**

- Distribution of a Population and Sample:
    - The distribution of a **population** describes the behavior of a random variable across all individuals of interest.
    - The distribution of a **sample** describes the behavior of a random variable in a specific sample from the population
- **Central Limit Theorem**: if an IID sample of size $$n$$ is large, then the probability distribution of the sample mean is roughly normal with mean $$\mu$$ and SD $$\sigma/\sqrt{n}$$.
- Prediction vs. Inference:
    - **Prediction** involves using models to predict outcomes for unseen data.
    - **Inference** involves using models to understand the underlying relationships between features and responses in the population.
- Bias and Variance:
    - **Bias** measures how far an estimator deviates from the parameter on average.
    - **Variance** measures the extent to which an estimator varies from its mean.
    Mean squared error (MSE) combines bias and variance to evaluate an estimator's "goodness."
- Observed data includes random errors (ε) due to data collection imperfections
    - $$\text{True relationship: }Y = g(x)$$
    - $$\text{Observed relationship: }Y = g(x) + \epsilon$$

### Bias, Variance, and Inference

**Bias-Variance Tradeoff**

- Model risk is the mean square prediction error of a model and consists of observation variance, model bias, and model variance.
    - $$\mathbb{E}\left[(Y(x)-\hat{Y}(x))^2\right]$$
- Observation variance is the variance of the random noise in the observations.
    - $$\sigma^2$$
- Model bias measures how much the model's predictions deviate from the true underlying relationship.
    - $$\mathbb{E}\left[\hat{Y}(x)\right]-g(x)$$
- Model variance describes how much the model's predictions vary when trained on different samples.
    - $$\text{Var}(\hat{Y}(x))$$
- The bias-variance tradeoff states that there is a balance between model complexity and model performance
    - $$\text{Model risk = observation variance + (model bias)}^2 \text{+ model variance}$$
    - Increasing model complexity reduces model bias but increases model variance, and vice versa
    - The goal is to find a complexity level that minimizes overall model risk.
        
       

**Inference for Regression**

- Inference involves drawing conclusions about population parameters based on sample data.
- Inference can help us understand the relationship between predictor variables and the response variable in a regression model.
- Hypothesis testing is a common technique in inference. It involves forming null and alternative hypotheses and assessing evidence in the data.
- Bootstrapping is a resampling technique used to estimate the sampling distribution of a statistic.
- Confidence intervals provide a range of values within which the true population parameter is likely to fall with a certain level of confidence.
- Colinearity occurs when one feature can be predicted fairly accurately by a linear combination of the other features, which happens when one feature is highly correlated with the others

**Correlation vs Causation**

- When interpreting model parameters, it's important to distinguish between correlation and causation
- Causal questions involve understanding the effects of interventions and counterfactuals.
- A confounder is a variable that affects both predictor and response, distorting the correlation between them
- The Neyman-Rubin Causal Model defines potential outcomes for each individual and helps formulate causal questions.
- The Average Treatment Effect (ATE) is the difference in outcomes between treated and control groups.
- Randomized experiments and observational studies are approaches used to estimate causal effects.

**Covariate Adjustment**

- Covariate adjustment is a method to control for confounding variables in estimating causal effects.
- Ignorability assumption is a key concept in covariate adjustment. It assumes that all important confounders are included in the model

## Different Tools: SQL, Logistic Regression, PCA, and Clustering

### SQL

- Databases are organized collections of data managed by Database Management Systems (DBMS).
- Databases offer advantages over CSV files for large datasets, including reliability, optimized computation, and access control.
- SQL (Structured Query Language) is a programming language used to communicate with databases.
- SQL is declarative, specifying what outcome is desired rather than detailing the process to achieve it.
- SQL commands are written in uppercase, but lowercase is also acceptable.
- The basic structure of a SQL query is: **`SELECT <columns> FROM <table> [additional clauses]`**.
- Clauses can be added to refine the query, such as **`WHERE`**, **`ORDER BY`**, **`LIMIT`**, and more.
- The **`WHERE`** keyword filters rows based on conditions.
- The **`ORDER BY`** keyword sorts rows in ascending order by default, using **`DESC`** for descending order.
- The **`LIMIT`** keyword restricts the number of rows in the output.
- The **`OFFSET`** keyword shifts the starting point of **`LIMIT`**.
- The **`GROUP BY`** clause groups data based on specified columns.
- Aggregation functions like **`MAX`**, **`MIN`**, **`SUM`**, and **`COUNT`** can be used with **`GROUP BY`**.
- The **`HAVING`** clause filters groups based on aggregate conditions.
- **`CAST`** is used to convert data from one type to another. It generates a new column with the converted values.
- **`CASE`** is used for conditional operations in SQL, similar to **`if`** statements in Python.
- **`LIKE`** operator is used to match strings with a specific pattern. % is a wildcard that matches any character or characters.
- **`JOIN`** is used to combine data from multiple tables based on common columns. Types of joins: cross join, inner join, full outer join, left outer join, right outer join.

### Logistic Regression (the following notes aren’t as great, I suggest visiting the course notes instead)

**Regression vs. Classification**

- Regression and classification are supervised learning problems.
- Classification models categorize data into groups, while regression models predict continuous values.
- binary classification: a classification problem where data may only belong to one of two groups

**Logistic Regression Model**

- Logistic regression returns the probability of data belonging to binary classes.
- Model output > 0.5 implies class 1, otherwise class 0.
- Odds ratio is the ratio of the probability of an event happening to not happening.
- It's used to measure likelihood and express probabilities in odds form.
- The logit function transforms odds ratio to an unbounded real number. It maps [0, 1] to (-∞, ∞).
- Logistic Regression Model Summary
    1. Optimize the model to find parameters $$\vec{\hat\theta}$$
    2. Use parameters to calculate classification activation: $$z=\vec{\hat\theta^T}\vec{x}$$
    3. Apply the logistic function to get the probability of a given datapoint belonging to class 1
- Logistic regression predicts probabilities using a sigmoid function.
- Linear regression predicts continuous values. Logistic regression predicts probabilities.

**Parameter Estimation**

- MSE loss isn't suitable for logistic regression.
- Non-convex, bounded, and conceptually questionable.

**Cross-Entropy Loss**

- Cross-entropy loss is a more intuitive alternative.
- Convex, strong error penalization, conceptually sound.
- It addresses the shortcomings of MSE loss.

**Maximum Likelihood Estimation**

- Maximum likelihood estimation (MLE) is a method to find model parameters that maximize the likelihood of the observed data.
- In the context of logistic regression, MLE helps us find the optimal parameter that maximizes the likelihood of observing the given data.
- Likelihood represents how probable the observed data is given a particular parameter value.

**Linear Separability and Regularization**

- Linear separability refers to the ability to separate classes using a linear boundary.
- In the case of logistic regression, linear separability can lead to diverging weights and overconfident predictions.
- Regularization is used to prevent overfitting and control the magnitudes of weights.
- Regularized logistic regression includes a regularization term in the loss function to balance between fitting the data and controlling weights.

**Performance Metrics**

- Accuracy is a basic metric to evaluate classification models, but it's not always suitable, especially with imbalanced data.
- Confusion matrix summarizes the classification results by counting true positives, true negatives, false positives, and false negatives.
- Precision and recall are two important metrics that provide insights into the model's performance on positive class prediction.

**Accuracy, Precision, and Recall**

- Precision measures the proportion of true positive predictions among all positive predictions.
- Recall measures the proportion of true positive predictions among all actual positive instances.
- Precision and recall are inversely related; optimizing one may come at the cost of the other.
- The choice between precision and recall depends on the specific context and priorities of the problem.

**Adjusting the Classification Threshold**

- The classification threshold determines the point at which the model classifies data into one of the classes.
- Adjusting the threshold can impact precision and recall.
- A higher threshold increases specificity and decreases sensitivity, while a lower threshold has the opposite effect.
- The ROC curve and AUC provide a graphical way to analyze the trade-off between true positive rate (recall) and false positive rate.

**Two More Metrics**

- True Positive Rate (TPR) corresponds to recall and measures the proportion of true positive predictions.
- False Positive Rate (FPR) measures the proportion of false positive predictions.
- TPR and FPR are related and change as the classification threshold varies.

**The ROC Curve**

- The ROC curve shows the trade-off between TPR (recall) and FPR as the threshold changes.
- A perfect classifier has TPR of 1 and FPR of 0, resulting in an AUC of 1.
- The closer the ROC curve is to the top-left corner, the better the classifier's performance.
- AUC quantifies the overall performance of the classifier, with a perfect classifier having an AUC of 1 and a random one having an AUC of 0.5.

### PCA

- In supervised learning, models are trained using labeled input-output pairs.
- Unsupervised learning focuses on finding patterns and relationships in unlabeled data.

**Dimensionality Reduction**

- Dimensionality reduction is a technique used to reduce the number of features while retaining important information. Dimensionality reduction is useful for visualizing high-dimensional data.

**Principle Component Analysis (PCA)**

- PCA is a linear technique for dimensionality reduction that transforms data to a lower-dimensional space.
- It involves finding the principal components that capture the most variability in the data. The primary goal of PCA is to capture the most important aspects of data variability.
- PCA vs. SVD: PCA uses Singular Value Decomposition (SVD) as one of its steps. SVD is a linear algebra algorithm that decomposes a matrix into three component parts.
- PCA Procedure
    - Center the data matrix by subtracting the mean of each attribute.
    - Use SVD to find all $$v_i$$, the principal components that satisfy the following criteria:
        - $$v_i$$ is a unit vector that linearly combines the attributes.
        - $$v_i$$ gives a one-dimensional projection of the data.
        - $$v_i$$ is chosen to minimize the sum of squared distances between each point and its projection onto v.
        - $$v_i$$ is orthogonal to all previous principal components.
- Data Variance and Centering; The i-th singular value represents how much variance the i-th principal component captures. The sum of the squared singular values is equivalent to the total variance of the data.

**Scree Plots**

- A scree plot shows the magnitude of singular values (diagonal values of Σ) in descending order.
- Scree plots help determine the number of principal components needed to describe the data sufficiently.
- A drop in singular value magnitude indicates a potential cutoff for the number of principal components.

**Biplots**

- Biplots overlay feature vectors on a scatter plot of principal component 2 vs. principal component 1.
- Each arrow represents a feature's direction and magnitude of contribution to the principal components.
- The angle and direction of the arrows provide insights into the correlations between features and principal components.

### Clustering

- Clustering is an unsupervised learning technique that aims to group similar observations together based on patterns in the data.
- An example involves using PCA to reduce data dimensions and then clustering the reduced data points.

**K-Means Clustering**

- K-Means is a popular clustering algorithm that follows these steps:
    1. Initialize K cluster centers randomly.
    2. Iterate:
        - Assign each data point to the nearest cluster center.
        - Update cluster centers to the mean of data points in the cluster.
    3. Converge when cluster assignments no longer change.
- K-Means aims to minimize inertia (sum of squared distances from data points to cluster centers). Inertia is a metric used to evaluate the quality of K-Means clustering results.
- The "elbow method" involves plotting inertia against the number of clusters and choosing the "elbow" point as the optimal number of clusters (K).
- Choosing the appropriate number of clusters (K) is subjective and can impact the results.

**Hierarchical Agglomerative Clustering**

- Hierarchical agglomerative clustering is a bottom-up approach.
- It starts with each data point in a separate cluster and then merges clusters iteratively.
- Different linkage criteria (e.g., single, complete, average) determine how to merge clusters.

**Clustering, Dendrograms, and Intuition**

- Agglomerative clustering creates a hierarchy of clusters.
- Each cluster is a tree, and merging history is tracked.
- Dendrograms visually represent the merging process.

**Silhouette Scores/Plot**

- Silhouette scores measure how well a data point is clustered relative to its own cluster compared to other clusters.
- Higher silhouette scores indicate better clustering.
- Silhouette scores can be plotted for all data points to visualize clustering quality.
- Points with high silhouette widths are well-clustered.