#!/usr/bin/env python
# coding: utf-8

# # Author: Sadaf Shaikh

# ![fight cancer](Image.335.png)

# **Background:**
# 
#    Breast cancer is the most common cancer among women in the world. It account for 25% of all cancer cases, and affected over 2.1 Million people in 2015. It starts when cells in the breast begin to grow out of control. These cells usually form a tumor that can often be seen on an X-ray or felt as a lump.
#     
#    Early diagnosis significantly increases the chances of surviver. The key challenges against it's detection is how to classify tumors into malignant(Cancer) or benign(not cancer). A tumor is considered malignant (Cancer) if the cells can grow into surrounding tissues or spread to distant areas of the body. A benign tumor does not invade nearby tissue or spread to other parts of the body the way cancer can. But benign tumors can be serious if they press on vital structures such as blood vessel or nerves.
#     
#    Machine Learning technique can dramatically improve the level of diagnosis in breast cancer. Research shows that experience physicians can detect cancer by 79% accuracy, while 91%(up to 97%) accuracy can be achieved using Machine Learning techniques.
# 
# 
# 
# 
#     

# **Project Task**
# 
#    In this study, my task is to classify tumors into malignant (cancer) or benign using features obtained from several cell images.
#    
#    Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
#    
#    
# **Attribute Information:**
#    
# 1.  ID number 
# 2.  Diagnosis (M = malignant, B = benign) 
# 
# **Ten real-valued features are computed for each cell nucleus:**
# 
# 1. Radius (mean of distances from center to points on the perimeter) 
# 2. Texture (standard deviation of gray-scale values) 
# 3. Perimeter 
# 4. Area 
# 5. Smoothness (local variation in radius lengths) 
# 6. Compactness (perimeter^2 / area - 1.0) 
# 7. Concavity (severity of concave portions of the contour) 
# 8. Concave points (number of concave portions of the contour) 
# 9. Symmetry 
# 10. Fractal dimension ("coastline approximation" - 1)
# 
# 
# 

# # Loading packages and data

# # Import needed Python Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[2]:


cancer


# As we can see above, not much can be done in the current form of the dataset. We need to view the data in a better format.

# # Let's view the data in a dataframe.

# In[3]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()


# - mean radius = mean of distances from center to points on the perimeter
# - mean texture = standard deviation of gray-scale values
# - mean perimeter = mean size of the core tumor
# - mean area = 
# - mean smoothness = mean of local variation in radius lengths
# - mean compactness = mean of perimeter^2 / area - 1.0
# - mean concavity = mean of severity of concave portions of the contour
# - mean concave points = mean for number of concave portions of the contour
# - mean symmetry =
# - mean fractal dimension = mean for "coastline approximation" - 1
# - radius error = standard error for the mean of distances from center to points on the perimeter
# - texture error = standard error for standard deviation of gray-scale values
# - perimeter error =
# - area error =
# - smoothness error = standard error for local variation in radius lengths
# - compactness error = standard error for perimeter^2 / area - 1.0
# - concavity error = standard error for severity of concave portions of the contour
# - concave points error = standard error for number of concave portions of the contour
# - symmetry error =
# - fractal dimension error = standard error for "coastline approximation" - 1
# - worst radius = "worst" or largest mean value for mean of distances from center to points on the perimeter
# - worst texture = "worst" or largest mean value for standard deviation of gray-scale values
# - worst perimeter =
# - worst smoothness = "worst" or largest mean value for local variation in radius lengths
# - worst compactness = "worst" or largest mean value for perimeter^2 / area - 1.0
# - worst concavity = "worst" or largest mean value for severity of concave portions of the contour
# - worst concave points = "worst" or largest mean value for number of concave portions of the contour
# - worst fractal dimension = "worst" or largest mean value for "coastline approximation" - 1

# # Let's Explore Our Dataset

# In[4]:


df_cancer.shape


# As we can see,we have 596 rows (Instances) and 31 columns(Features)

# In[5]:


df_cancer.columns


# Above is the name of each columns in our dataframe.

# # The next step is to Visualize our data

# In[6]:


# Let's plot out just the first 5 variables (features)
sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness'] )


# The above plots shows the relationship between our features. But the only problem with them is that they do not show us which of the "dots" is Malignant and which is Benign. 
# 
# This issue will be addressed below by using "target" variable as the "hue" for the plots.

# In[7]:


# Let's plot out just the first 5 variables (features)
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness'] )


# **Note:** 
#     
#   1.0 (Orange) = Benign (No Cancer)
#   
#   0.0 (Blue) = Malignant (Cancer)

# # How many Benign and Malignant do we have in our dataset?

# In[8]:


df_cancer['target'].value_counts()


# As we can see, we have 212 - Malignant, and 357 - Benign

#  Let's visulaize our counts

# In[9]:


sns.countplot(df_cancer['target'], label = "Count") 


# # Let's check the correlation between our features 

# In[10]:


plt.figure(figsize=(20,12)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# There is a strong correlation between the mean radius and mean perimeter, mean area and mean primeter

# # Introduction to Classification Modeling: Suport Vector Maching (SVM)¶

# # Modeling

# Depending on how long we've lived in a particular place and traveled to a location, we probably have a good understanding of commute times in our area. For example, we've traveled to work/school using some combination of the Metro, buses, trains, Ubers, taxis, carpools, walking, biking, etc.
# 
# **All humans naturally model the world around them.**
# 
# Over time, our observations about transportation have built up a mental dataset and a mental model that helps us predict what traffic will be like at various times and locations. We probably use this mental model to help plan our days, predict arrival times, and many other tasks.
# 
# - As data scientists we attempt to make our understanding of relationships between different quantities more precise through using data and mathematical/statistical structures.
# - This process is called modeling.
# - Models are simplifications of reality that help us to better understand that which we observe.
# - In a data science setting, models generally consist of an independent variable (or output) of interest and one or more dependent variables (or inputs) believed to have an effect on the independent variable.
# - Linear regression is an extremely common and critically important modeling tool.

# # Model-based inference
# 
# - We can use models to conduct inference.
# - Given a model, we can better understand relationships between an independent variable and the dependent variable or between multiple independent variables.
# 
# **An examples of where inference from a mental model would be valuable is:**
# 
# Determining what times of the day we work best or get tired.

# # Prediction

# - We can use a model to make predictions, or to estimate an dependent variable's value given at least one independent variable's value.
# - Predictions can be valuable even if they are not exactly right.
# - Good predictions are extremely valuable for a wide variety of purposes.
# 
# **An examples of where prediction from a mental model could be valuable:**
# 
# Predicting how long it will take to get from point A to point B.

# # What is the difference between model prediction and inference?

# - Inference is judging what the reationship, if any, there is between the data and the output.
# - Prediction is making guesses about future scenarios based on data and a model constructed on that data.

# **In this project, we will be talking about a particular Machine Learning Model called Support Vector Machine (SVM)**

# # Introduction to SVM

# # What is a Support Vector Machine (SVM)?
# 
# A Support Vector Machine (SVM) is a binary linear classification whose decision boundary is explicitly constructed to minimize generalization error. It is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression and even outlier detection. 
# 
# SVM is well suited for classification of complex but small or medium sized datasets.

# # How does SVM classify?

# It's important to start with the intuition for SVM with the **special linearly separable** classification case.
# 
# If classification of observations is **"linearly separable"**, SVM fits the **"decision boundary"** that is defined by the largest margin between the closest points for each class. This is commonly called the **"maximum margin hyperplane (MMH)"**.

# ![linearly separable SVM](linear_separability_vs_not.png)

# # The advantages of support vector machines are:
# 
# - Effective in high dimensional spaces.
# - Still effective in cases where number of dimensions is greater than the number of samples.
# - Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
# - Versatile: different [Kernel](http://scikit-learn.org/stable/modules/svm.html#svm-kernels) functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

# # The disadvantages of support vector machines include:
# 
# - If the number of features is much greater than the number of samples, avoid over-fitting in choosing [Kernel functions](http://scikit-learn.org/stable/modules/svm.html#svm-kernels) and regularization term is crucial.
# - SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see [Scores and probabilities](http://scikit-learn.org/stable/modules/svm.html#scores-probabilities), below).

# # Model Training

# **From our dataset, let's create the target and predictor matrix**
# 
# - "y" = Is the feature we are trying to predict (Output). In this case we are trying to predict wheither our "target" is Cancer (Malignant) or not (Benign). I.e. we are going to use the "target" feature here.
# - "X" = The predictors which are the remaining columns (mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc)

# In[11]:


X = df_cancer.drop(['target'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
X.head()


# In[12]:


y = df_cancer['target']
y.head()


# # Create the training and testing data

# Now that we've assigned values to our "X" and "y", the next step is to import the python library that will help us to split our dataset into training and testing data.

# - Training data = Is the subset of our data used to train our model.
# - Testing data =  Is the subset of our data that the model hasn't seen before. This is used to test the performance of our model.

# In[13]:


from sklearn.model_selection import train_test_split


# Let's split our data using 80% for training and the remaining 20% for testing.

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)


# Let now check the size our training and testing data.

# In[15]:


print ('The size of our training "X" (input features) is', X_train.shape)
print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)


# # Import Support Vector Machine (SVM) Model 

# In[16]:


from sklearn.svm import SVC


# In[17]:


svc_model = SVC()


# # Now, let's train our SVM model with our "training" dataset.

# In[18]:


svc_model.fit(X_train, y_train)


# # Let's use our trained model to make a prediction using our testing data

# In[19]:


y_predict = svc_model.predict(X_test)


# **Next step is to check the accuracy of our prediction by comparing it to the output we already have (y_test). We are going to use confusion matrix for this comparison**

# <a id='confusion-matrix'></a>
# 
# ## The confusion matrix
# 
# ---
# 
# The confusion matrix is a table representing the performance of your model to classify labels correctly.
# 
# **A confusion matrix for a binary classification task:**
# 
# |   |Predicted Negative | Predicted Positive |   
# |---|---|---|
# |**Actual Negative**  | True Negative (TN)  | False Positive (FP)  | 
# |**Actual Positive** | False Negative (FN)  | True Positive (TP)  |  
# 
# In a binary classifier, the "true" class is typically labeled with 1 and the "false" class is labeled with 0. 
# 
# > **True Positive**: A positive class observation (1) is correctly classified as positive by the model.
# 
# > **False Positive**: A negative class observation (0) is incorrectly classified as positive.
# 
# > **True Negative**: A negative class observation is correctly classified as negative.
# 
# > **False Negative**: A positive class observation is incorrectly classified as negative.
# 
# Columns of the confusion matrix sum to the predictions by class. Rows of the matrix sum to the actual values within each class. You may encounter confusion matrices where the actual is in columns and the predicted is in the rows: the meaning is the same but the table will be reoriented.
# 
# > **Note:** Remembering what the cells in the confusion matrix represents can be a little tricky. The first word (True or False) indicates whether or not the model was correct. The second word (Positive or Negative) indicates the *model's guess* (not the actual label!).

# ** Let's create a confusion matrix for our classfier's performance on the test dataset.**

# In[20]:


# Import metric libraries

from sklearn.metrics import classification_report, confusion_matrix


# In[21]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[22]:


sns.heatmap(confusion, annot=True)


# In[23]:


print(classification_report(y_test, y_predict))


# **As we can see, our model did not do a very good job in its predictions. It predicted that 48 healthy patients have cancer.**
# 
# **Let's explore ways to improve the performance of our model.**

# # Improving our Model

# The first process we will try is by Normalizing our data

# Data Normalization is a feature scaling process that brings all values into range [0,1]

# X' = (X-X_min) / (X_max - X_min)-----> X_range

# # Normalize Training Data

# In[24]:


X_train_min = X_train.min()
X_train_min


# In[25]:


X_train_max = X_train.max()
X_train_max


# In[26]:


X_train_range = (X_train_max- X_train_min)
X_train_range


# In[27]:


X_train_scaled = (X_train - X_train_min)/(X_train_range)
X_train_scaled.head()


# # Normalize Training Data

# In[28]:


X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range


# In[31]:


svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[32]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)


# # SVM with Normalized data

# In[33]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[34]:


sns.heatmap(confusion,annot=True,fmt="d")


# In[35]:


print(classification_report(y_test,y_predict))


# **Awesome performance! We only have 1 false prediction.**

# # Further Model Improvement

# ** The search for the optimal set of hyperparameters is called gridsearching.**
# 
# Gridsearching gets its name from the fact that we are searching over a "grid" of parameters. For example, imagine the alpha hyperparameters on the x-axis and fit_intercept on the y-axis, and we need to test all points on the grid.

# <a id='searching'></a>
# 
# ## Searching for the best hyperparameters
# 
# ---
# 
# Let's see if we can improve on our model by searching for the best hyperparameters.
# 
# We would need to evaluate on the training data the set of hyperparameters that perform best, and then use this set of hyperparameters to fit the final model and score on the testing set.
# 

# # Gridsearch Model

# In[36]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[37]:


from sklearn.model_selection import GridSearchCV


# In[38]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[39]:


grid.fit(X_train_scaled,y_train)


# **Let's print out the "grid" with the best parameter**

# In[40]:


print (grid.best_params_)
print ('\n')
print (grid.best_estimator_)


# **As we can see, the best parameters are "C" = 100, "gamma" = "0.01" and "kernel" = 'rbf'**

# In[41]:


grid_predictions = grid.predict(X_test_scaled)


# In[42]:


cm = np.array(confusion_matrix(y_test, grid_predictions, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[43]:


sns.heatmap(confusion, annot=True)


# In[44]:


print(classification_report(y_test,grid_predictions))


# **As we can see, our best model is SVM with Normalized data, followed by our Gridsearch model**

# ---
# 
# ## Reference table of common classification metric terms and definitions
# 
# <br><br>
# 
# |  TERM | DESCRIPTION  |
# |---|---|
# |**TRUE POSITIVES** | The number of "true" classes correctly predicted to be true by the model. <br><br> `TP = Sum of observations predicted to be 1 that are actually 1`<br><br>The true class in a binary classifier is labeled with 1.|
# |**TRUE NEGATIVES** | The number of "false" classes correctly predicted to be false by the model. <br><br> `TN = Sum of observations predicted to be 0 that are actually 0`<br><br>The false class in a binary classifier is labeled with 0.|
# |**FALSE POSITIVES** | The number of "false" classes incorrectly predicted to be true by the model. This is the measure of **Type I error**.<br><br> `FP = Sum of observations predicted to be 1 that are actually 0`<br><br>Remember that the "true" and "false" refer to the veracity of your guess, and the "positive" and "negative" component refer to the guessed label.|
# |**FALSE NEGATIVES** | The number of "true" classes incorrectly predicted to be false by the model. This is the measure of **Type II error.**<br><br> `FN = Sum of observations predicted to be 0 that are actually 1`<br><br>|
# |**TOTAL POPULATION** | In the context of the confusion matrix, the sum of the cells. <br><br> `total population = tp + tn + fp + fn`<br><br>|
# |**SUPPORT** | The marginal sum of rows in the confusion matrix, or in other words the total number of observations belonging to a class regardless of prediction. <br><br>|
# |**ACCURACY** | The number of correct predictions by the model out of the total number of observations. <br><br> `accuracy = (tp + tn) / total_population`<br><br>|
# |**PRECISION** | The ability of the classifier to avoid labeling a class as a member of another class. <br><br> `Precision = True Positives / (True Positives + False Positives)`<br><br>_A precision score of 1 indicates that the classifier never mistakenly classified the current class as another class.  precision score of 0 would mean that the classifier misclassified every instance of the current class_ |
# |**RECALL/SENSITIVITY**    | The ability of the classifier to correctly identify the current class. <br><br>`Recall = True Positives / (True Positives + False Negatives)`<br><br>A recall of 1 indicates that the classifier correctly predicted all observations of the class.  0 means the classifier predicted all observations of the current class incorrectly.|
# |**SPECIFICITY** | Percent of times the classifier predicted 0 out of all the times the class was 0.<br><br> `specificity = tn / (tn + fp)`<br><br>|
# |**FALSE POSITIVE RATE** | Percent of times model predicts 1 when the class is 0.<br><br> `fpr = fp / (tn + fp)`<br><br>|
# |**F1-SCORE** | The harmonic mean of the precision and recall. The harmonic mean is used here rather than the more conventional arithmetic mean because the harmonic mean is more appropriate for averaging rates. <br><br>`F1-Score = 2 * (Precision * Recall) / (Precision + Recall)` <br><br>_The f1-score's best value is 1 and worst value is 0, like the precision and recall scores. It is a useful metric for taking into account both measures at once._ |

# # Sources:

# 1. http://scikit-learn.org/stable/modules/svm.html

# In[ ]:





# In[ ]:




