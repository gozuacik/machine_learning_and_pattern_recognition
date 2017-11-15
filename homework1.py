# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:53:28 2017

@author: necip
"""

# Importing numpy library
import numpy as np

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Import library for plotting
import matplotlib.pyplot as plt

#Test case sample settings
test_case=np.array([100,1000,2000,100,1000,2000])

#Test case result matrix [6X2]. This will be including success ratio
#for both training and validation set
test_result=np.array([[0.0,0.0],[0.0,0.0],[0.0, 0.0],[0.0,0.0],[0.0 ,0.0],[0.0 ,0.0]])


# This function converts character value to float type
convert = lambda x: float(ord(x)/100.)

# Input data consists of Sex,Length,Diameter,Height,Whole weight,
# Shucked weight,Viscera weight,Shell weight
# Convertion is performed for first column 'Sex'
# This data will be used for Test Case 4,5 and 6.
input_array=np.genfromtxt('abalone_dataset.txt',dtype=float, usecols=[0,1,2,3,4,5,6,7],
                          converters={0: convert})
input_array_length=len(input_array)


# Subset input data consists of Sex,Length,Diameter
# Convertion is performed for first column 'Sex'
# This data will be used for Test Case 1,2 and 3.
subset_input_array=np.genfromtxt('abalone_dataset.txt',dtype=float, usecols=[0,1,2],
                          converters={0: convert})
subset_input_array_length=len(subset_input_array)


# Output data
output_array=np.genfromtxt('abalone_dataset.txt', dtype=float, usecols=[8])


# Case 1 
# Using 3 features as input
# 100 samples for training, and rest for validation set
# Splitting data for training set and display statistic data
print("Case 1 Test Results")
print("-------------------")
training_array=subset_input_array[0:test_case[0]]

# Splitting data for validation set and display statistic data
validation_array=subset_input_array[test_case[0]:(subset_input_array_length+1)]

# Set up an array to include actual output values corresponding to training_array
output_training_array=output_array[0:test_case[0]]

# Set up an array to include actual output values corresponding to validation_array
output_validating_array=output_array[test_case[0]:(subset_input_array_length+1)]

#Create a Gaussian Naive BayesClassifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(training_array, output_training_array)

# Predict Output for Training Array
predicted_training_array= model.predict(training_array)

# Confusion Matrix for predicting training_array
confusion_matrix_training_array=np.array([[0,0,0],[0,0,0],[0,0,0]])

# Below loop compares predicted_training_array and output_training_array
for num in range(0,test_case[0]):  
    x=int(predicted_training_array[num])
    y=int(output_training_array[num])
    confusion_matrix_training_array[x-1][y-1]=confusion_matrix_training_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_training_array[i][j]
        total=total+confusion_matrix_training_array[i][j]

           
success=total-error;
success_ratio=(success*100)/total
test_result[0][0]=success_ratio;
print("Number of Successful Predictions for Training") 
print(success)
print("Number of Erroneous Predictions for Training") 
print(error)
print("Success Ratio for Training Set")
print(success_ratio)

print("Confusion Matrix for Training Array") 
print(confusion_matrix_training_array)

print("\n")
 
# Predict Output for Validation Array
predicted_validation_array= model.predict(validation_array)

# Confusion Matrix for predicting validation_array
confusion_matrix_validating_array=np.array([[0,0,0],[0,0,0],[0,0,0]])


for num in range(0,(subset_input_array_length-test_case[0])):  
    x=int(predicted_validation_array[num])
    y=int(output_validating_array[num])
    confusion_matrix_validating_array[x-1][y-1]=confusion_matrix_validating_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_validating_array[i][j]
        total=total+confusion_matrix_validating_array[i][j]
            
success=total-error;
success_ratio=(success*100)/total
test_result[0][1]=success_ratio;
print("Number of Successful Predictions for Validating Array") 
print(success)
print("Number of Erroneous Predictions for Validating Array") 
print(error)
print("Success Ratio for Validating Set")
print(success_ratio)

print("Confusion Matrix for Validating Array") 
print(confusion_matrix_validating_array)

print("\n")


# Case 2 
# Using 3 features as input
# 1000 samples for training, and rest for validation set
# Splitting data for training set and display statistic data
print("Case 2 Test Results")
print("-------------------")
training_array=subset_input_array[0:test_case[1]]

# Splitting data for validation set and display statistic data
validation_array=subset_input_array[test_case[1]:(subset_input_array_length+1)]

# Set up an array to include actual output values corresponding to training_array
output_training_array=output_array[0:test_case[1]]

# Set up an array to include actual output values corresponding to validation_array
output_validating_array=output_array[test_case[1]:(subset_input_array_length+1)]

#Create a Gaussian Naive BayesClassifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(training_array, output_training_array)


# Predict Output for Training Array
predicted_training_array= model.predict(training_array)

# Confusion Matrix for predicting training_array
confusion_matrix_training_array=np.array([[0,0,0],[0,0,0],[0,0,0]])


for num in range(0,test_case[1]):  
    x=int(predicted_training_array[num])
    y=int(output_training_array[num])
    confusion_matrix_training_array[x-1][y-1]=confusion_matrix_training_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_training_array[i][j]
        total=total+confusion_matrix_training_array[i][j]
            

success=total-error;
success_ratio=(success*100)/total
test_result[1][0]=success_ratio;
print("Number of Successful Predictions for Training") 
print(success)
print("Number of Erroneous Predictions for Training") 
print(error)
print("Success Ratio for Training Set")
print(success_ratio)

print("Confusion Matrix for Training Array") 
print(confusion_matrix_training_array)

print("\n")
 
# Predict Output for Validation Array
predicted_validation_array= model.predict(validation_array)

# Confusion Matrix for predicting validation_array
confusion_matrix_validating_array=np.array([[0,0,0],[0,0,0],[0,0,0]])

for num in range(0,(subset_input_array_length-test_case[1])):  
    x=int(predicted_validation_array[num])
    y=int(output_validating_array[num])
    confusion_matrix_validating_array[x-1][y-1]=confusion_matrix_validating_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_validating_array[i][j]
        total=total+confusion_matrix_validating_array[i][j]
            
success=total-error;
success_ratio=(success*100)/total
test_result[1][1]=success_ratio;
print("Number of Successful Predictions for Validating Array") 
print(success)
print("Number of Erroneous Predictions for Validating Array") 
print(error)
print("Success Ratio for Validating Set")
print(success_ratio)

print("Confusion Matrix for Validating Array") 
print(confusion_matrix_validating_array)

print("\n")

# Case 3 
# Using 3 features as input
# 2000 samples for training, and rest for validation set
# Splitting data for training set and display statistic data
print("Case 3 Test Results")
print("-------------------")
training_array=subset_input_array[0:test_case[2]]

# Splitting data for validation set and display statistic data
validation_array=subset_input_array[test_case[2]:(subset_input_array_length+1)]

# Set up an array to include actual output values corresponding to training_array
output_training_array=output_array[0:test_case[2]]

# Set up an array to include actual output values corresponding to validation_array
output_validating_array=output_array[test_case[2]:(subset_input_array_length+1)]

#Create a Gaussian Naive BayesClassifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(training_array, output_training_array)


# Predict Output for Training Array
predicted_training_array= model.predict(training_array)

# Confusion Matrix for predicting training_array
confusion_matrix_training_array=np.array([[0,0,0],[0,0,0],[0,0,0]])


for num in range(0,test_case[2]):  
    x=int(predicted_training_array[num])
    y=int(output_training_array[num])
    confusion_matrix_training_array[x-1][y-1]=confusion_matrix_training_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_training_array[i][j]
        total=total+confusion_matrix_training_array[i][j]
            

success=total-error;
success_ratio=(success*100)/total
test_result[2][0]=success_ratio;
print("Number of Successful Predictions for Training") 
print(success)
print("Number of Erroneous Predictions for Training") 
print(error)
print("Success Ratio for Training Set")
print(success_ratio)

print("Confusion Matrix for Training Array") 
print(confusion_matrix_training_array)

print("\n")
 
# Predict Output for Validation Array
predicted_validation_array= model.predict(validation_array)

# Confusion Matrix for predicting validation_array
confusion_matrix_validating_array=np.array([[0,0,0],[0,0,0],[0,0,0]])

for num in range(0,(subset_input_array_length-test_case[2])):  
    x=int(predicted_validation_array[num])
    y=int(output_validating_array[num])
    confusion_matrix_validating_array[x-1][y-1]=confusion_matrix_validating_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_validating_array[i][j]
        total=total+confusion_matrix_validating_array[i][j]
            
success=total-error;
success_ratio=(success*100)/total
test_result[2][1]=success_ratio;
print("Number of Successful Predictions for Validating Array") 
print(success)
print("Number of Erroneous Predictions for Validating Array") 
print(error)
print("Success Ratio for Validating Set")
print(success_ratio)

print("Confusion Matrix for Validating Array") 
print(confusion_matrix_validating_array)

print("\n")


# Case 4 
# Using all features as input
# 100 samples for training, and rest for validation set
# Splitting data for training set and display statistic data
print("Case 4 Test Results")
print("-------------------")
training_array=input_array[0:test_case[3]]

# Splitting data for validation set and display statistic data
validation_array=input_array[test_case[3]:(input_array_length+1)]

# Set up an array to include actual output values corresponding to training_array
output_training_array=output_array[0:test_case[3]]

# Set up an array to include actual output values corresponding to validation_array
output_validating_array=output_array[test_case[3]:(input_array_length+1)]

#Create a Gaussian Naive BayesClassifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(training_array, output_training_array)

# Predict Output for Training Array
predicted_training_array= model.predict(training_array)

# Confusion Matrix for predicting training_array
confusion_matrix_training_array=np.array([[0,0,0],[0,0,0],[0,0,0]])


for num in range(0,test_case[3]):  
    x=int(predicted_training_array[num])
    y=int(output_training_array[num])
    confusion_matrix_training_array[x-1][y-1]=confusion_matrix_training_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_training_array[i][j]
        total=total+confusion_matrix_training_array[i][j]

success=total-error;
success_ratio=(success*100)/total
test_result[3][0]=success_ratio;
print("Number of Successful Predictions for Training") 
print(success)
print("Number of Erroneous Predictions for Training") 
print(error)
print("Success Ratio for Training Set")
print(success_ratio)

print("Confusion Matrix for Training Array") 
print(confusion_matrix_training_array)

print("\n")
 
# Predict Output for Validation Array
predicted_validation_array= model.predict(validation_array)

# Confusion Matrix for predicting validation_array
confusion_matrix_validating_array=np.array([[0,0,0],[0,0,0],[0,0,0]])

for num in range(0,(input_array_length-test_case[3])):  
    x=int(predicted_validation_array[num])
    y=int(output_validating_array[num])
    confusion_matrix_validating_array[x-1][y-1]=confusion_matrix_validating_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_validating_array[i][j]
        total=total+confusion_matrix_validating_array[i][j]
            
success=total-error;
success_ratio=(success*100)/total
test_result[3][1]=success_ratio;
print("Number of Successful Predictions for Validating Array") 
print(success)
print("Number of Erroneous Predictions for Validating Array") 
print(error)
print("Success Ratio for Validating Set")
print(success_ratio)

print("Confusion Matrix for Validating Array") 
print(confusion_matrix_validating_array)

print("\n")

# Case 5 
# Using all features as input
# 1000 samples for training, and rest for validation set
# Splitting data for training set and display statistic data
print("Case 5 Test Results")
print("-------------------")
training_array=input_array[0:test_case[4]]

# Splitting data for validation set and display statistic data
validation_array=input_array[test_case[4]:(input_array_length+1)]

# Set up an array to include actual output values corresponding to training_array
output_training_array=output_array[0:test_case[4]]

# Set up an array to include actual output values corresponding to validation_array
output_validating_array=output_array[test_case[4]:(input_array_length+1)]

#Create a Gaussian Naive BayesClassifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(training_array, output_training_array)

# Predict Output for Training Array
predicted_training_array= model.predict(training_array)

# Confusion Matrix for predicting training_array
confusion_matrix_training_array=np.array([[0,0,0],[0,0,0],[0,0,0]])


for num in range(0,test_case[4]):  
    x=int(predicted_training_array[num])
    y=int(output_training_array[num])
    confusion_matrix_training_array[x-1][y-1]=confusion_matrix_training_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_training_array[i][j]
        total=total+confusion_matrix_training_array[i][j]
            

success=total-error;
success_ratio=(success*100)/total
test_result[4][0]=success_ratio;
print("Number of Successful Predictions for Training") 
print(success)
print("Number of Erroneous Predictions for Training") 
print(error)
print("Success Ratio for Training Set")
print(success_ratio)

print("Confusion Matrix for Training Array") 
print(confusion_matrix_training_array)

print("\n")
 
# Predict Output for Validation Array
predicted_validation_array= model.predict(validation_array)

# Confusion Matrix for predicting validation_array
confusion_matrix_validating_array=np.array([[0,0,0],[0,0,0],[0,0,0]])

for num in range(0,(input_array_length-test_case[4])):  
    x=int(predicted_validation_array[num])
    y=int(output_validating_array[num])
    confusion_matrix_validating_array[x-1][y-1]=confusion_matrix_validating_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_validating_array[i][j]
        total=total+confusion_matrix_validating_array[i][j]
            
success=total-error;
success_ratio=(success*100)/total
test_result[4][1]=success_ratio;
print("Number of Successful Predictions for Validating Array") 
print(success)
print("Number of Erroneous Predictions for Validating Array") 
print(error)
print("Success Ratio for Validating Set")
print(success_ratio)

print("Confusion Matrix for Validating Array") 
print(confusion_matrix_validating_array)

print("\n")

# Case 6 
# Using all features as input
# 2000 samples for training, and rest for validation set
# Splitting data for training set and display statistic data
print("Case 6 Test Results")
print("-------------------")
training_array=input_array[0:test_case[5]]

# Splitting data for validation set and display statistic data
validation_array=input_array[test_case[5]:(input_array_length+1)]

# Set up an array to include actual output values corresponding to training_array
output_training_array=output_array[0:test_case[5]]

# Set up an array to include actual output values corresponding to validation_array
output_validating_array=output_array[test_case[5]:(input_array_length+1)]

#Create a Gaussian Naive BayesClassifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(training_array, output_training_array)

# Predict Output for Training Array
predicted_training_array= model.predict(training_array)

# Confusion Matrix for predicting training_array
confusion_matrix_training_array=np.array([[0,0,0],[0,0,0],[0,0,0]])


for num in range(0,test_case[5]):  
    x=int(predicted_training_array[num])
    y=int(output_training_array[num])
    confusion_matrix_training_array[x-1][y-1]=confusion_matrix_training_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_training_array[i][j]
        total=total+confusion_matrix_training_array[i][j]
            

success=total-error;
success_ratio=(success*100)/total
test_result[5][0]=success_ratio;
print("Number of Successful Predictions for Training") 
print(success)
print("Number of Erroneous Predictions for Training") 
print(error)
print("Success Ratio for Training Set")
print(success_ratio)

print("Confusion Matrix for Training Array") 
print(confusion_matrix_training_array)

print("\n")
 
# Predict Output for Validation Array
predicted_validation_array= model.predict(validation_array)

# Confusion Matrix for predicting validation_array
confusion_matrix_validating_array=np.array([[0,0,0],[0,0,0],[0,0,0]])


for num in range(0,(input_array_length-test_case[5])):  
    x=int(predicted_validation_array[num])
    y=int(output_validating_array[num])
    confusion_matrix_validating_array[x-1][y-1]=confusion_matrix_validating_array[x-1][y-1] + 1

error=0;
total=0;
for i in range(0,3):
    for j in range(0,3):
        if i != j:
            error=error + confusion_matrix_validating_array[i][j]
        total=total+confusion_matrix_validating_array[i][j]
            

success=total-error;
success_ratio=(success*100)/total
test_result[5][1]=success_ratio;
print("Number of Successful Predictions for Validating Array") 
print(success)
print("Number of Erroneous Predictions for Validating Array") 
print(error)
print("Success Ratio for Validating Set")
print(success_ratio)

print("Confusion Matrix for Validating Array") 
print(confusion_matrix_validating_array)
 
 
# Plotting data
n_groups = 6
ratio_train = (test_result[0][0], test_result[1][0], test_result[2][0],
               test_result[3][0],test_result[4][0],test_result[5][0])
ration_valid = (test_result[0][1], test_result[1][1], test_result[2][1],
               test_result[3][1],test_result[4][1],test_result[5][1])
 
# Below code creates plot based on success ratio calculated
# above for each test case including Training and Validation
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, ratio_train, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Training Set')
 
rects2 = plt.bar(index + bar_width, ration_valid, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Validating Set')

plt.ylim(0,100)
plt.xlabel('Test Scenarios')
plt.ylabel('Percentage')
plt.title('Success Ratio')
plt.xticks(index + bar_width, ('Case 1', 'Case 2', 'Case 3', 'Case 4','Case 5','Case 6'))
plt.legend()
 
plt.tight_layout()
plt.show()