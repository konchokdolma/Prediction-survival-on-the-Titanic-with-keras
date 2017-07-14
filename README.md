## Prediction survival on the Titanic

We take data from the kaggle competition: https://www.kaggle.com/c/titanic In the file with train data we are given the information about passengers in Titanic. For the input we have their name, age, gender, ticket class, siblings/spouses and parents/children aboard, ticket number, fare, cabin number and port of embarkation. As an output we have if person survived or not.

First of all code writes the list of data into DataFrame and drops columns with names and ticket numbers. Then it converts columns with strings to integers, i.e. in the column with genders 'male' is '0' and 'female' is '1'.

In the 'train' function we train our model for prediction. We fit the model to evaluate it first for the full data, and then separatly for train(67% of data) and test(33% of data).

To call the function we drop rows with nan values and then convert all data into integer.

### prediction-survival-titanic-dense.py

In this example to build the model we use five dense layers with 270 number of epochs and 27 for the batch size.

Example output:

`
32/712 [>.............................] - ETA: 0s
acc: 86.24%
loss: 5.24%

train accuracy: 87.61% 
train loss: 4.88%

test accuracy: 83.47% 
test loss: 5.96%
`

### prediction-survival-titanic-lstm.py

In this example for the model we use two lstm layers with 30 of the epochs and batch size 10.

Example output:

`
32/712 [>.............................] - ETA: 0s
acc: 83.15%
loss: 5.38%

train accuracy: 83.19% 
train loss: 5.24%

test accuracy: 83.05% 
test loss: 5.66%
`
