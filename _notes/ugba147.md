---
title: "Notes UGBA147"
collection: notes
permalink: /notes/ugba147
date: 2023-05-15
---

UGBA147: Advanced Business Analytics. I took UGBA147 during summer 2023 with Richard Huntsinger. The following notes are taken from Richard Huntsinger’s textbook: *Data Science for Business Decisions*. I have kept my notes largely free of code to maintain brevity and readability, so please check out Huntsinger’s textbook for wonderful real-life examples on how to apply the concepts described in my notes.

# Data & Decisions

### **Data Landscape**

- The field of data science offers various methods to understand data for better business decisions.
- Decision modeling involves exploring the effects of decisions informed by data.
    - Decision Modeling (effects of decisions)
- Intuition relies on patterns recognized from personal experience.
    - Intuition (patterns discerned from experience)
- Statistical analysis is about identifying patterns within data and includes:
    - Descriptive Statistics (summarizing data)
    - Statistical Inference (inferring implications from data samples)
    - Other statistical methods
- Data science encompasses several components:
    - Data management involves curating data.
    - Data engineering focuses on designing data pipelines.
        - Cluster Computing (parallel processing)
        - Stream Computing (incremental availability)
        - Cloud Computing (remote access)
    - Statistical analysis for data patterns.
    - Computational statistics combines statistics and computer science for intensive analysis.
        - Also known as Data Analytics or Data Mining.
    - Data analysis includes:
        - Descriptive Statistics (summarizing with numbers)
        - Data Visualization (summarizing with graphics)
        - Other data representation methods.
    - Data analytic modeling includes:
        - Machine Learning (Statistical Learning)
        - Descriptive Modeling (Unsupervised Learning or Cluster Analysis)
        - Predictive Modeling (Supervised Learning)
        - Social Network Analysis and other methods.
- Business Analytics applies data science and other sciences to business problems.
    - Descriptive Statistics, Data Visualization, Descriptive Modeling, Predictive Modeling, Stochastic Modeling (Simulation), Prescriptive Analytics (Optimization), and more.
- Business Intelligence applies data analysis to business situational awareness.
    - Descriptive Statistics, Data Visualization, and other methods.
- Artificial Intelligence includes methods executed by machines for tasks historically done by humans.
    - Machine Learning is a sub-field of AI.
    - Rule-Based Systems (Expert Systems), Machine Learning, and more.
- Terminology in data science varies across fields and contexts, leading to diverse meanings.

### D**ata-to-decision process model**

- data-to-decision process model prescribes a way to make data-informed decisions by
iteratively working through four stages: decision modeling, data retrieval, data analysis, and
data analytic modeling
- Decision modeling involves exploring the effects of data-informed decisions.
- Data retrieval involves retrieving data for use in decision making.
- Data analysis involves preparing, exploring, and transforming data in various ways to expose
patterns that could lead to non-obvious insights useful in decision making.
- Data analytic modeling involves constructing models that further expose patterns in data by
estimating the underlying processes responsible for generating the data

### Decision Model

**Elements of Decision Model**

- A decision model is a construct that connects a decision or decision method to an estimated business outcome through inter-dependent calculations. It can be thought of as a formula or a sequence of calculations represented as a feed-forward network.
- A robust decision model is built from several key elements:
    - **Decision**: Specify the decision to be made, such as choosing a product or determining investment.
    - **Decision Method**: If studying the effect of a decision method, indicate so.
    - **Decision Method Performance**: Identify calculated values that measure the performance of the decision method.
    - **Business Parameters**: Specify values determined by data analysis or provided by business management.
    - **Dependencies**: Define which calculated values depend on decisions, business parameters, or other calculated values.
    - **Calculated Values**: Specify intermediary calculations needed to estimate the business outcome.
    - **Business Result**: Indicate which calculated value represents the desired business outcome.

### **Sensitivity Analysis**

- Sensitivity analysis involves examining a decision model to understand how variations or adjustments in its inputs, such as decision method performance and business parameter values, can impact the estimated business outcome.

# Data Preparation

### Selection

**About Selection**

- Data selection is akin to slicing and dicing in cooking.
- Different data analysis methods work on different portions of datasets, based on analysis objectives.
- Data selection entails choosing a subset of data from a larger dataset.

**Anatomy of a Dataset**

- Datasets are collections of observations described by variables, organized as grids.
- Rows represent observations, columns represent variables.
- Different terms are used across fields to describe dataset elements (e.g., observation, row, record, instance, example).
- Various terms are used for variable (e.g., column, field, feature, attribute).

**Selection Methods**

- Index-Based Selection: Rows, Columns, Rows & Columns
- Name-Based Selection: Column, Columns
- Selection with Reorder
- Selection with Random Reorder: Rows
- Criterion-Based Selection: Rows, Columns

### Amalgamation

- Amalgamation is accomplished by concatenation or join
    - Row-wise concatenation is useful for appending or inserting when datasets have
    corresponding variables.
    - Column-wise concatenation is useful for appending or inserting when datasets have
    corresponding observations ordered in the same way.
    - Join is useful when datasets describe the same kind of events, but employ different variables
    or include observations about different specific events.

### Synthetic Variables

- Synthetic variables are defined in terms of other already existing variables
    - Unit conversion: Define the synthetic variable as cx + d, where x is an already existing
    variable, c is a conversion rate, and d is a zero or non-zero conversion shift.
    - Linear recombination: Define a synthetic variable as c1x1 + c2x2 + · · · + cnxn + d, where
    x1, x2, . . ., xn are already existing variables, c1, c2, . . ., cn are coefficients not expressed in terms of variables, and d is a zero or non-zero constant not expressed in terms of variables.
    - Non-linear recombination: Define a synthetic variable as a formula involving already
    existing variables. Linear combination is a special case of non-linear recombination.
    - Descriptive statistic: Define a synthetic variable as a descriptive statistic function.
    Descriptive statistic is a special case of linear or non-linear recombination.
    - Recombination with timestep offset: Define a synthetic variable such that the value at any
    observation is a formula involving already existing variable values at other observations.

### Dummy Variables

- For each value of the categorical variable's domain, create a new numerical variable, called
a dummy variable.
- For each observation, assign dummy variables
- For each observation, assign each dummy variable to be 1 if it corresponds to the categorical
variable value, otherwise assign it to be 0.
- Discard the categorical variable.
- Discard one of the dummy variables, because the remaining dummy variables are enough
to preserve all information that was in the categorical variable.

# Data Exploration

### Descriptive Statistics

- A descriptive statistic summarizes a dataset in just a few numbers, often just one number

### Cross-Tabulation

- Cross-tabulation is a summarization technique using a small table for convenience and pattern recognition.
- An aggregate table is constructed to summarize data using aggregation methods.
- Steps for creating an aggregate table:
    - Choose group-by variables to categorize observations.
    - Select an aggregate variable.
    - Choose an aggregation function.
    - Group observations based on group-by variables.
    - Apply the aggregation function to the aggregate variable for each group.
    - Compile results into a table with rows for each group and columns for components of group names and aggregated values.
- Observations include identification and measurement variables (e.g., (date=2023-01-01, sales=1,000,000)).
- Long table representation retains identifier variables but replaces measure variables with variable names and values.
- Converting to long table is useful for constructing cross-tables.

**Cross-Tabulation Method**

- The cross-tabulation method is a generalized version of the aggregation method. It constructs a small table, called a cross-table, that summarizes a dataset. The cross-table's rows and columns are named after the dataset's variable values and variable names.
- Steps for creating a cross-table:
    - Convert dataset to long table representation.
    - Identify variables suitable for row and column names in the cross-table (identifier and measure variables).
    - Specify rows and columns for the cross-table.
    - Choose aggregation function for measures.
    - Construct the table as specified.
    - Populate the table with appropriate aggregated measure values.
- Cross-tables provide a comprehensive view of relationships between variables.
- Suitable for visualizing patterns, associations, and distributions in data.

### Data Visualization

- Data visualization is about summarizing a dataset with a graphic. Here are some popular data
visualization methods:
- 1-axis scatterplot, 2-axis scatterplot, trellis of scatterplots, 3-axis scatterplot projection
- lineplot, stepplot, pathplot, bar chart, histogram, pie chart, violinplot, boxplot, heat map, conditional format

### Kernel Density Estimation

- Kernel Density Estimation (KDE) is a method used to estimate the underlying process that generates a dataset.
- It constructs a probability density function (PDF) that approximates the underlying process.
- A probabilistic model simplifies the underlying process, assigning various values with probabilities.
- A probability distribution is a type of probabilistic model, providing probabilities for possible values.

**Kernel Density Estimation Method**

- Steps of the KDE method:
    1. Choose a kernel and one or more bandwidths.
    2. For each observation in space:
        - Place the kernel in space, centered around the observation's value(s).
    3. For every point in space:
        - Sum the contributions from all kernels at that point.
    4. Interpret the cumulative result as a probability density function (PDF).

**Kernel and Bandwidth**

- Kernel: A defined shape, often a bell-shaped curve, used to estimate the underlying distribution.
- Bandwidth(s): Control(s) determining the width and shape of kernels.
    - Higher bandwidth: Increased overlap of kernels, resulting in a smoother, flatter PDF.
    - Lower bandwidth: Decreased overlap of kernels, leading to a spikier, more detailed PDF.

# Data Transformation

### Normalization

- Normalization transforms a dataset measured in any units into a similar dataset with all
variables scaled to standard units that account for mean and standard deviation of variables
- Normalization does not cause loss or addition of information to the dataset, except for
information specific to the units, which can often facilitate useful data analysis and analytic
modeling

### Balancement

- A balanced dataset contains roughly equal observations for each possible class value.
- An unbalanced dataset lacks this equality in class distribution.
- Balancing transforms an unbalanced dataset into a similar, balanced one.

**Binary Class Variable in Unbalanced Dataset**

- In an unbalanced dataset with a binary class variable (two possible class values):
    - Majority Class: The value observed in the majority of instances.
    - Minority Class: The less frequent value in the dataset.

**Class Assignment**

- An observation is classified as the majority class if its class value is the majority class.
- An observation is classified as the minority class if its class value is the minority class.

**Balancing Methods**

- Various methods aim to balance a dataset:
    - Balancing by Downsample: Removing observations from the majority class.
    - Balancing by Bootstrap: Adding observations by duplicating instances from the minority class.
    - Balancing by Downsample and Bootstrap: Combining both downsampling and bootstrap techniques.

### Imputation

- Imputation transforms a dataset with missing values into a similar dataset without any missing values. It does so by replacing missing values with new synthesized values.
- There are several popular trimming and imputation methods, including these:
    - Remove Observations with Missing Values
    - Remove Variables with Missing Values
    - Impute by Variable Mean: Within a variable, replace missing values with the variable
    mean.
    - Impute by Neighbor Mean: Within a variable, replace missing values with the mean of
    the nearest non-missing values.
    - Impute by Linear Interpolation: Within a variable, replace missing values with the linear
    interpolation of the nearest non-missing values.

### Alignment

- Alignment transforms multiple time series datasets measured at different resolutions into one
dataset.
- There are 2 general approaches to alignment:
    - Alignment by Contraction: Aggregate observations of a fine resolution dataset down to
    the size of a coarse resolution dataset, and then join the datasets.
    - Alignment by Expansion: Duplicate observations of a coarse resolution dataset up to the
    size of a fine resolution dataset, disaggregate the duplicated observations, and then join
    the datasets

### **PCA**

- Principal Component Analysis (PCA) transforms a dataset represented by a set of variables into a new dataset represented by a different set of variables.
- In the transformed dataset, all the information and variance from the original dataset is preserved, but much of the variance is concentrated in the first few new variables (principal components).
- These new variables are orthogonal axes in the data space and are derived from linear combinations of the original variables.

**Principal Components**

- Principal components are new variables resulting from PCA.
- They capture most of the variance in the dataset, allowing for dimensionality reduction.

**PCA Process**

1. Optional normalization of the dataset.
2. Calculate the centroid of the dataset (mean of each variable).
3. Calculate the weight matrix using a PCA function, which defines lines of optimal variance in the multi-variable space.
4. For each observation, apply the PCA formula to calculate principal component values using the centroid, weight matrix, and observation's variable values.
5. Principal component values represent the observation in the new transformed space.

**PCA Formula**

- Principal component values for an observation with 'm' native variables and 'm' principal components:
PC1 = (wx1,PC1 * (x1 - cx1)) + (wx2,PC1 * (x2 - cx2)) + ... + (wxm,PC1 * (xm - cxm))
PC2 = (wx1,PC2 * (x1 - cx1)) + (wx2,PC2 * (x2 - cx2)) + ... + (wxm,PC2 * (xm - cxm))
...
PCm = (wx1,PCm * (x1 - cx1)) + (wx2,PCm * (x2 - cx2)) + ... + (wxm,PCm * (xm - cxm))
    - xi is the observation's i-th variable value
    - P Cj is the observation's j-th principal component value
    - cxi is the dataset centroid's i-th variable value
    - wxiP Cj is the weight matrix's value for the i-th variable and the j-th principal component

**Normalization**

- Normalizing the dataset is often beneficial for PCA, as it prevents variable units from distorting the weight matrix.

**Purpose of PCA**

- Dimensionality reduction while preserving most of the variance.
- Uncover patterns in data by highlighting the directions of maximum variability.
- Feature extraction to remove noise and redundancies in data

# Classification

### Classification Methodology

- The classification methodology consists of three main components: construction of a classifier, prediction using the classifier, and evaluation of the classifier's performance.

**Construction**

- A classifier construction method takes a reference classified dataset and hyper-parameter settings as input.
- The classifier construction method identifies patterns in the dataset that distinguish different classes and creates a classifier incorporating this knowledge.
- Hyper-parameters configure the behavior of the classifier construction method.
- Parameters are components set by the classifier construction method to fine-tune the classifier.
- The result is a classifier that estimates class probabilities for predictions, with its form and parameters defined by the construction method.
- Hyper-parameters are like switches and dials to set the classifier's behavior.
- Parameters are switches and dials that determine the classifier's operation conditions.

**Prediction**

- A classifier takes an unclassified dataset as input and estimates probabilities of class values using patterns learned during construction.
- Each observation in the new dataset receives a class value probability.
- The classifier can use a cutoff value to make a binary classification decision based on probabilities.
- Setting the cutoff influences the classifier's certainty threshold for making predictions.
- Prediction is also known as classification.

**Evaluation**

- Evaluation estimates the quality of a classifier's predictions by comparing them with other evaluation classifiers.
- Evaluation classifiers are constructed using the same method and hyper-parameter settings as the prime classifier but on subsets of the dataset.
- Performance metrics for evaluation classifiers provide an estimate of the prime classifier's performance.
- Evaluation helps determine how well the classifier will generalize to new, unseen data.
- Evaluation is also known as validation or testing.

### Classification Evaluation

**Confusion Matrix**

- A confusion matrix is a tool used to summarize and evaluate the performance of a classifier.
- It provides insight into how a classifier's predictions align with the actual known classes.
- The matrix is typically structured as a 2x2 table, with rows representing predicted class values and columns representing known class values.

**Interpreting a Confusion Matrix**

- The confusion matrix is divided into four cells, each representing different prediction outcomes.
- Row 1, Column 1: True Positive (TP) - predicted positive and known positive
- Row 2, Column 1: False Positive (FP) - predicted positive but known negative
- Row 1, Column 2: False Negative (FN) - predicted negative but known positive
- Row 2, Column 2: True Negative (TN) - predicted negative and known negative

**Performance Metrics from Confusion Matrix**

- Essential metrics include accuracy, true positive rate (sensitivity/recall), true negative rate (specificity), false positive rate, false negative rate, positive predictive value (precision), negative predictive value, and F1 score.
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - True Positive Rate (Sensitivity/Recall): TP / (TP + FN)
    - True Negative Rate (Specificity): TN / (TN + FP)
    - False Positive Rate: FP / (FP + TN)
    - False Negative Rate: FN / (FN + TP)
    - Positive Predictive Value (Precision): TP / (TP + FP)
    - Negative Predictive Value: TN / (TN + FN)
    - F1 Score: (2 * TP) / (2 * TP + FP + FN)

**Evaluation Methods**

- Evaluation by In-Sample Performance: Evaluates a classifier using the same dataset it was trained on. Might lead to overly optimistic estimates.
- Evaluation by Out-of-Sample Performance (Holdout): Evaluates a classifier using a separate testing dataset from the same reference dataset. Offers a more realistic estimate of performance.
- Evaluation by Cross-Validation Performance: Divides the dataset into folds and trains and tests the classifier multiple times, rotating the folds. Provides a good estimate of performance while utilizing the entire dataset for both training and testing.

**Advantages and Limitations**

- Cross-validation is generally considered more reliable for performance estimation than in-sample and out-of-sample evaluations.
- Each evaluation method has its own set of limitations, such as overly optimistic estimates in in-sample and potentially high variance in out-of-sample evaluation.

### k-Nearest Neighbors

**Method Overview**

- The k-nearest neighbors (k-NN) method involves computing distances (or dissimilarities) between observations to classify new, unclassified data points.

**Constructing a Classifier**

- Constructing a k-NN classifier based on a reference classified dataset is simple.
- The reference dataset itself is the classifier, essentially serving as a lookup table.

**Predicting with k-NN**

- To predict the class of a new unclassified observation using k-NN:
    1. Set the number of neighbors to consider (k).
    2. Choose a distance metric (e.g., Euclidean distance).
    3. Normalize the data (optional) to avoid distortion due to differing scales of predictor variables.
    4. Calculate distances from the new observation to all observations in the reference dataset.
    5. Identify the k observations with the shortest distances.
    6. Calculate the mean of the class labels of the k identified observations (positive class = 1, negative class = 0).
    7. Interpret the mean as the probability that the new observation belongs to the positive class.
    8. Predict the new observation's class based on probability.

**Measures of Distance**

- Different distance measures can be used, such as Manhattan distance, Euclidean distance, or customized approaches.
- Euclidean distance is commonly used in the absence of a specific reason to choose another measure.

**Other Considerations**

- The number of neighbors (k) is typically set within the range of 1 to 20.
- Normalization of data is recommended when predictor variables have significantly different scales.
- Constructing a k-NN classifier requires minimal computation time, but predicting can be computationally intensive for large datasets.

**Applicability**

- The k-NN method is suitable for datasets with multiple predictor variables and where distances between observations can be computed

### Logistic Regression

**Method Overview**

- Logistic regression involves finding an appropriate vector sigmoid function.
- The classifier constructed through logistic regression is determined by the coefficients of this vector sigmoid function.

**Constructing a Classifier**

- To construct a logistic regression classifier:
    1. Begin with a reference classified dataset.
    2. Treat predictor variables as inputs to a vector sigmoid function.
    3. Use the class dummy variable as the outcome of the vector sigmoid function (0 for the positive class and 1 for the negative class).
    4. Identify the vector sigmoid function that best approximates the dataset using a search algorithm and best-fit criterion.

**Predicting with Logistic Regression**

- To predict the class of a new unclassified observation using logistic regression:
    1. Apply the vector sigmoid function to the predictor variable values of the new observation.
    2. Interpret the result as the probability that the new observation belongs to the negative class.
    3. Predict the new observation's class based on probability.

**Sigmoid and Vector Sigmoid Functions**

- A sigmoid function maps numeric values to the range between 0 and 1.
- The vector sigmoid function rescales a vector of numeric values into the 0 to 1 range.
- The sigmoid (logistic) function: sigmoid(x) = 1 / (1 + e^(-x)).

### Decision Tree

**Predicting Using Full Dataset Probabilities**

- Predicting new observations' classes based on probabilities from a full dataset.
- Choose the class with a probability exceeding a certain cutoff.

**Predicting Using Split Dataset Probabilities**

- Alternatively, split the dataset into subsets (left and right) using a split criterion.
- Predict using probabilities from one of the splits based on the split criterion applied to the new observation.
- Choose the class with a probability exceeding the cutoff.

**Predicting Using Recursively Split Dataset Probabilities**

- Extend the concept of splitting recursively until terminal splits (leaf nodes) are reached.
- Construct a decision tree classifier.
- To predict, traverse the tree based on the split criteria for each node until a terminal node is reached.
- Choose the class with the highest probability in the terminal node exceeding the cutoff.

**Finding Best Splits**

**Entropy**

- Entropy measures uncertainty in a distribution.
- Define entropy function H for class labels A, B, ... as: H = -Σ(Probability(label) × log2(Probability(label))).
- Maximum entropy corresponds to maximum uncertainty (50% probability for each class).

**Method to Find Best Split Criterion**

1. Calculate the entropy of the dataset's class distribution.
2. For each candidate split criterion:
    - Determine the two splits based on the criterion.
    - Calculate the entropy of each split's class distribution.
    - Calculate the weighted average of split entropies.
    - Calculate information gain (dataset entropy - weighted average of split entropies).
3. Pick the candidate split criterion associated with the largest information gain.

**Pruning**

**Overfitting and Pruning**

- Fully recursively splitting a decision tree can lead to overfitting.
- Pruning restricts splitting using hyper-parameters to prevent overfitting.

**Pruning Methods**

- Pruning by Maximum Depth: Restrict the maximum levels of splits.
- Pruning by Minimum Number of Observations per Node: Restrict the minimum number of observations in a split.
- Pruning reduces the complexity of the tree, avoiding overfitting.

### Classification by Naive Bayes

- Naive Bayes is a classification method that uses probability and Bayes' theorem to classify new observations into predefined classes. It's a simple and effective approach that involves calculating likelihoods, prior probabilities, and posterior scores to make predictions.

**Naive Bayes with 1 Numeric Predictor Variable**

- Likelihood Prediction: Using probability density functions based on kernel density estimation to build likelihoods for numeric predictor variables.
- Prediction by Likelihood: Classifying new observations based on the likelihood of their proximity to existing observations of each class.
- Prediction by Prior Probability: Considering prior probabilities to classify new observations based on the dominant class in the dataset.
- Prediction by Relative Posterior Score: Combining likelihoods and prior probabilities to create a posterior score, then scaling them to relative posterior scores. Comparing these scores to a cutoff to make predictions.

**Naive Bayes and 1 Categorical Predictor Variable**

- Categorical Likelihoods: For categorical predictor variables, using relative frequency tables to calculate likelihoods.
- Prediction by Relative Posterior Score: Combining likelihoods and prior probabilities for both predictor variables to create relative posterior scores. Comparing these scores to a cutoff to make predictions.

**Naive Bayes with 2 Predictor Variables**

- Likelihoods for Multiple Variables: Calculating likelihoods for each variable separately using Gaussian density estimation or kernel density estimation.
- Prediction by Relative Posterior Score: Multiplying all likelihoods and the prior probability to calculate the posterior score. Scaling these scores to relative posterior scores for prediction.

**Naive Bayes with Gaussian Density Estimation**

- Gaussian Likelihoods: Using Gaussian density estimation to build likelihoods for numeric predictor variables.
- Predictive Accuracy: Despite its simplicity, the method can produce accurate classifiers that capture general patterns.

**Naive Bayes with Laplace Smoothing**

- Undesirable Situation: Demonstrating the problem of zero likelihoods for specific categorical values not present in the dataset.
- Laplace Smoothing: Adjusting relative frequency tables by adding a small value to each count to avoid zero probabilities.
- Enhanced Likelihoods: Using smoothed likelihoods to calculate posterior scores for better prediction.
- Connection to LaPlace Smoothing: Named after Pierre-Simon LaPlace's approach to the Sunrise Problem, where probabilities are smoothed to avoid extremes.

### Classification by Support Vector Machine

**Introduction to SVM:**

- SVM is a classification method used to separate data into classes using a hyperplane.
- Linearly separable datasets can be divided by a point (1D), line (2D), plane (3D), or hyperplane (>3D).

**Linearly Separable Data with 1 Predictor Variable:**

- Linearly separable datasets can be partitioned by predictor variables.
- Class A and B observations can be separated by predictor variable x1.
- New observations between x1=6 and x1=14 can partition the dataset.
- Classification of a new observation is determined by its position relative to the classes.

**Finding Support Vectors and Boundary:**

- Support vectors are observations at the edges of the margin between classes.
- Boundary is the middle of the margin, defined by support vectors.
- Boundary determination helps in classifying new observations.

**Scoring and Prediction by Score Sign:**

- Observations are scored based on their distances from the boundary.
- The sign of the score determines the predicted class.

**Scoring and Prediction by Probability:**

- Scores are transformed into probabilities using the sigmoid function.
- Observations are classified based on probabilities and a cutoff value.

**SVM with Penalties for Not Linearly Separable Data:**

- Introduces penalties to handle non-linearly separable data.
- The penalty accounts for misclassified observations.
- Penalty and cost influence support vector determination and boundary.

**SVM with Kernel Trick for Non-Linear Data:**

- Kernel trick adds synthetic variables to make data linearly separable.
- Kernel function transforms data to a higher-dimensional space.
- Observations can be separated using higher-dimensional boundaries.

**Support Vector Machine with Many Predictor Variables:**

- SVM can handle datasets with many predictor variables.

**Balanced vs. Unbalanced Data:**

- SVM performs better with balanced datasets.
- Balanced data results in fairer classification.

### Neural Network

- Neural network method (deep learning) is inspired by biological neural networks.
- Online learning: Iteratively update a model using sequential reference data.
- Activation threshold: Only large values propagate; smaller ones are blocked.
- Perceptron model: Simplest, with input nodes to an output node.
- 2-layer neural network: Input nodes to hidden nodes to output node.
- Multi-layer neural network: Input nodes to hidden layers to output node.
- Connections have weights, activation functions for hidden and output nodes.
- Predictions involve applying the function to new data and interpreting class probabilities.
- Neural network method finds weights for good predictions.
- 2-layer neural network can approximate any function closely.
- Multi-layer model can approximate 2-layer model using fewer nodes.

**Model Form:**

- **Perceptron Model:**
    - Input nodes for bias and predictors.
    - Output node.
    - Connections from input to output.
    - Weights on connections (one per connection).
    - Activation function on output node.
- **2-Layer Neural Network Model:**
    - Input nodes for bias and predictors.
    - Hidden nodes (plus bias node).
    - Output node.
    - Connections from input to hidden, hidden to output.
    - Weights on connections (one per connection).
    - Activation function on hidden and output nodes.
- **Multi-Layer Neural Network Model:**
    - Input nodes for bias and predictors.
    - Hidden nodes (across layers, including bias nodes).
    - Output node.
    - Connections from input to hidden, between hidden layers, last hidden to output.
    - Weights on connections (one per connection).
    - Activation function on hidden and output nodes.

**Activation Functions:**

- Sigmoid (logistic): Rescales to 0-1.
- Hyperbolic tangent (tanh): Rescales to -1 to 1.
- Softplus: Continuous approximation of ReLU.
- Rectified linear unit (ReLU): Non-continuous, common choice.

**Construct Model**

- **Perceptron Method:**
    - Initialize: Cutoff, class dummy, random weights.
    - Iterate:
        - Randomize observation order.
        - For each observation:
            - Compute output, error, prediction.
            - If incorrect prediction, adjust weights.
    - Converges for separable data; may not for non-separable.
- **Neural Network Method:**
    - Initialize: Error function, class dummy, random weights.
    - Iterate:
        - Randomize observation order.
        - For each observation:
            - Compute output, error.
            - Adjust weights using back-propagation.
    - Output computed by propagating values, applying activation functions.
- **Predict:**
    - Apply model to new observation's variables to compute output.
    - Scale output if necessary into 0-1 range, interpret as probability.
    - Apply cutoff to probability to determine class value.

# Regression

### Regression Methodology

- Regression and classification methodologies are similar but not identical.
- The discourse on regression methodology parallels classification methodology.
- Three main parts of regression methodology: Construction, Prediction, Evaluation.

**Construction:**

- Regressor construction method uses known-outcome data and hyper-parameter settings.
- Goal: Detect patterns in known-outcome data to build a regressor.
- Regressor is defined by form and parameters set by the construction method.
- Hyper-parameters configure the behavior of the construction method.
- Parameters are components of the regressor set by the construction method.
- Construction is also known as supervised learning, training, modeling, etc.
- Reference known-outcome dataset with predictor and outcome variables.

**Prediction:**

- Regressor predicts outcomes for new unknown-outcome data.
- Regressor uses built-in knowledge from known-outcome data patterns.
- Each observation in the new dataset gets a predicted outcome.
- Prediction is also known as regression.

**Evaluation:**

- Evaluation estimates regressor's quality by examining other similar regressors.
- Prime regressor is the one being evaluated.
- Evaluation regressors are constructed using the same method and settings as the prime regressor.
- They use subsets of known-outcome data.
- Assumes evaluation regressors' performance is similar to the prime regressor's.
- Evaluation provides estimates of the prime regressor's performance.
- Evaluation is also known as validation or testing.
- Process involves partitioning data, constructing evaluation regressors, and calculating performance metrics.

### Regression Evaluation

**Error Table and Performance Metrics:**

- Error table summarizes how well a regressor back-predicts outcome values.
- Provides a convenient way to calculate performance metrics.
- Error table consists of known outcomes, predicted outcomes, and errors.
- Essential performance metrics:
    - MAE (Mean Absolute Error): Average of absolute errors.
    - MSE (Mean Square Error): Average of squared errors.
    - RMSE (Root Mean Square Error): Square root of MSE.
    - MAPE (Mean Absolute Percent Error): Average of percent errors.

**In-Sample Evaluation:**

- All reference known-outcome dataset used for training and testing.
- Prime regressor constructed and tested using the same dataset.
- Performance metric (e.g., RMSE) calculated for the prime regressor.

**Out-of-Sample Evaluation:**

- Dataset partitioned into training and testing subsets (holdin and holdout).
- Prime regressor constructed on holdin data and evaluated on holdout data.
- Performance metric calculated for prime regressor's performance.

**Cross-Validation Evaluation:**

- Dataset divided into multiple equally-sized folds.
- Each fold serves as a testing set while the others are used for training.
- Multiple evaluation regressors constructed, each trained on different subset.
- Performance metrics calculated for each evaluation regressor.
- Average of performance metrics used as an estimate of the prime regressor's performance.

### Linear Regression

- Linear regression is a method for constructing a regressor model.
- Different versions of linear regression use optimization techniques to find the best-fit line, plane, or hyper-plane through a set of reference observations.

**Simple Linear Regression:**

- Simple linear regression creates a predictive model with one numerical predictor and one numerical outcome.
- The model is represented by an equation for a line defined by an intercept and coefficient.
- The goal is to find the best-fit line that minimizes the sum of squared errors between the observations and the line.
- The error is the vertical distance between an observation and the line.

**Multiple Linear Regression:**

- Multiple linear regression constructs a predictive model with multiple numerical predictors and one numerical outcome.
- The model is represented by an equation for a plane (with 2 predictors) or a hyper-plane (with 3 or more predictors), defined by intercept and coefficients.
- Similar to simple linear regression, the aim is to find the best-fit plane or hyper-plane that minimizes the sum of squared errors.

**Interpretation and Evaluation:**

- Every set of observations has a corresponding best-fit line, plane, or hyper-plane.
- The quality of the best-fit can vary; larger sum of squared errors indicates poorer fit.
- Linear regression's strength lies in its ability to model various shapes beyond just lines, planes, or hyper-planes. This is achieved through cleverly prepared datasets with synthetic variables.
- Linear regression can identify non-linear patterns as well.

# Ensemble Assembly

### Bagging

- Ensemble learning aims to combine multiple individual models to create a more powerful and accurate predictor.
- Bagging (Bootstrap Aggregating) is a specific ensemble assembly method that utilizes multiple models constructed using the same method but different bootstrap samples of the reference dataset.

**Bagging Process:**

1. **Reference Dataset:** Begin with a reference dataset containing observations and variables.
2. **Bootstrap Samples:** Create several bootstrap samples from the reference dataset. Bootstrap samples involve randomly selecting observations with replacement, maintaining the same number of observations as the original dataset.
3. **Component Models:** For each bootstrap sample, build a model using the same construction method and hyper-parameter values.
4. **Predictions:** Use the component models to make predictions on new observations.
5. **Voting:** Compare the predictions of the component models and predict the class that receives the majority of votes.

**Benefits of Bagging:**

- Bagging reduces the variance and overfitting that can occur with single models.
- It leverages the diversity of the models trained on different bootstrap samples.
- The ensemble's prediction tends to be more robust and accurate than individual model predictions.

### Boosting

- Boosting is a specific ensemble assembly method that combines models constructed using a single method and hyper-parameter values, but with different weighted bootstrap samples of a reference dataset. It focuses on observations predicted incorrectly by other models.

**Boosting Process:**

1. **Reference Dataset:** Start with a reference dataset containing observations and variables.
2. **Component Models:** Build a model using the same construction method and hyper-parameter values on the original dataset. Then assess its prediction errors.
3. **Weighted Bootstrap Sampling:** Create a new dataset by selecting observations with prediction errors with higher weights, emphasizing misclassified instances.
4. **Component Model Enhancement:** Build another model on the new dataset, aiming to correct the errors of the previous model.
5. **Repeat and Combine:** Continue building models iteratively, each focusing on the observations misclassified by the previous models.
6. **Predictions and Voting:** Combine the predictions of the component models and predict the class that receives the majority of votes.

**Benefits of Boosting:**

- Boosting improves model accuracy by iteratively focusing on challenging observations.
- The ensemble is more robust and tends to achieve high accuracy on complex datasets.
- Boosting addresses the weaknesses of individual models and captures nuanced patterns.

### Stacking

- Stacking is a specific ensemble assembly method that combines predictive models built using different model construction methods, utilizing a single reference dataset.

**Stacking Process:**

1. **Reference Dataset:** Begin with a reference dataset containing observations and variables.
2. **Component Models:** Build several models using various methods or hyper-parameter values but based on the same reference dataset.
3. **Transformed Data:** Collect the predictions made by the component models on the original data to create a new dataset.
4. **Stacked Model Construction:** Build a new model (stacked model) using the transformed data as the reference dataset.
5. **Prediction:** Make predictions using the stacked model on new observations by expressing the new observation's class in terms of how the component models would predict it.

**Benefits of Stacking:**

- Stacking allows for the combination of diverse modeling techniques, leveraging their strengths.
- It captures the collective wisdom of multiple models, potentially yielding more accurate predictions.
- Stacking can be particularly effective when different models excel in different aspects of prediction.

# Cluster Analysis

### Cluster Analysis Methodology

- Cluster analysis is used to determine logical groupings within a dataset when there are many features but only a few clusters are feasible.
- It's an unsupervised machine learning technique, meaning it doesn't require labeled data.

**Cluster Analysis Methods:**

- Popular cluster analysis methods include:
    - Hierarchical Agglomeration
    - K-Means
    - Gaussian Mixture

**Methodology:**

- Cluster analysis involves two main phases: construction and evaluation.
- **Construction:** Build a cluster model based on an unclassified dataset.
- **Evaluation:** Assess the quality of the resulting cluster model.

**Construction Phase:**

- Use a cluster analysis method that takes an unclassified dataset as input and generates a cluster model as output.
- The cluster model assigns each observation to a class (cluster), effectively extending the dataset with an additional class variable.
- The cluster model construction can be referred to as training, modeling, building a model, or fitting a model.

**Evaluation Phase:**

- Evaluate the quality of the cluster model using evaluation methods that generate a quality metric score.
- These methods often involve measuring the dissimilarity between observations within a cluster and between clusters.
- The quality score provides insight into how well the cluster model represents the data's inherent patterns.

**Measures of Dissimilarity:**

- Both construction and evaluation involve measuring dissimilarity between observations and clusters.
- Inter-observation distance: A specification for measuring the distance between individual observations, often using methods like Euclidean distance, Manhattan distance, Maximum distance, etc.
- Inter-cluster distance: Measuring the distance between clusters involves considering inter-observation distances between the clusters' constituent observations, often using centroid linkage, median linkage, single linkage, complete linkage, etc.
- Normalized data is often used for distance calculations to ensure uniformity in units.

**Benefits of Cluster Analysis:**

- Helps uncover hidden patterns and groupings in data.
- Useful in exploratory data analysis and forming hypotheses.
- Can be applied to a wide range of fields, from marketing segmentation to biology and more.

### Cluster Model Evaluation

**Evaluation Metrics:**

- To assess the usefulness of a cluster model, evaluate it using a method that generates a quality metric score.
- This method typically involves measuring the dissimilarity of observations within a cluster and between clusters.

**Quality Metrics for Scoring Cluster Models:**

- Mean Mean Intra-Cluster Dispersion: Average distance between observations within a cluster, averaged across all clusters.
- Mean Inter-Cluster Dispersion: Average distance between clusters.
- Dispersion Ratio: Average distance between clusters compared to average distance within a cluster, across all clusters. Accounts for dissimilarity within and between clusters.
- Akaike Information Criterion (AIC): Measures the likelihood that the cluster model represents the underlying process compared to other possible models, considering model complexity.
- Bayesian Information Criterion (BIC): Similar to AIC but accounts for model complexity differently.
- Mallow's Cp: A special case of AIC.

**Distance Matrices:**

- Some evaluation methods and construction methods use distance matrices for calculations.
- Distance matrices are square matrices where columns and rows represent observation numbers, and entries denote the distance between pairs of observations.

**Dispersion Ratio:**

- Dispersion ratio is a key metric in cluster model evaluation.
- It quantifies how similar observations are within a cluster compared to how dissimilar clusters are from each other.
- Calculated as the mean inter-cluster dispersion divided by the mean mean intra-cluster dispersion.

**Inter-Cluster Distance Calculation:**

- Calculate the distance between clusters by determining the centroids (mean values of variables) for each cluster.
- Compute the distance between centroids using a specified linkage method (e.g., centroid linkage) and inter-observation distance (e.g., Euclidean distance).
- Calculate the mean of the distances between all pairs of clusters to obtain the mean inter-cluster dispersion.

**Interpreting Dispersion Ratio:**

- A higher dispersion ratio indicates that observations within a cluster are more similar to each other than to observations in other clusters, which is often desired.
- In contrast, a lower dispersion ratio suggests that observations within a cluster aren't much more similar to each other than to observations in other clusters.

**Comparing Cluster Models:**

- Use dispersion ratios to compare the quality of different cluster models and select the one that best suits the intended purpose.