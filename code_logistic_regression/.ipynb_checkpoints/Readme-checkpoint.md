# Implement and evaluate the logistic regression with L2 regularization and stochastic gradient descent (SGD)

All coding happens in `logistic_regression.py`.

## One real-world data set in CSV format: `ionosphere`

See the `data_description` file within the data set folder for the description of the source of the data.

The data set includes:

- X_train, y_train: the training data set samples and labels.

- X_test, y_test: the validation data set samples and labels.

Reminder and suggestion: 

- When you read in the data set into numpy array, do not misalign/mess up the samples and their labels. 

- When you randomly shuffle the each data set (in sub-project 2), do not misalign/mess up the samples and their labels.

- An easy way to read in CSV files is to use the `Pandas` library. 


## First paste your code of the `z_transform` function into the `code_misc/utils.py` module.


## Sub-project 1.1: Finish `fit()` for batch GD without regularization.   (50 points)

See the instruction in the docstring and comment of the functions in the source code file.

In this subproject, you do not need to worry about and touch the `lam`, `SGD`, and `mini_batch_size` parameters. 


## Sub-project 1.2: Finish `predict()`, `error()`, `_sigmoid()`, and `_v_sigmoid()`.   (20, 20, 5, 5 points)

See the instruction in the docstring and comment of the functions in the source code file.



## Sub-project 2: Add regularization into `fit()`. (20 points)

In this subproject, you will revise your Subproject 1.1's code for the `fit()` function, so that the function will use the parameter `lam` for regularization in the gradient descent search. 



## Sub-project 3: Add SGD into `fit()`. (50 points)

In this subproject, you will revise your Sub-project 2's code for the `fit()` function, so that the function will use the parameters `SGD` and `mini_batch_size` for SGD search. If `SGD == True`, your `fit()` function will use SGD search for logistic regression and the size of each minibatch is `mini_batch_size`.

Remind that before the training starts, you will first randomly shuffle your entire training set. Then, you will **logically** cut the entire training set into multiple mini batches of each sized `mini_batch_size`. By **logically**, it means that you are not going to make copies of any of the training samples, but instead you only need to specify the starting row and ending row of the minibatch in the entire training set. Your model's training will repeatedly iterate through these mini batches and use each minibatch as a training set to update the `w` vector in one gradient descent walk step. 

To get a better  design for the program/code, I suggest in this subproject, you will introduce and use a helper function that will cleanly produce the indexes of the starting and ending rows of each minibatch, whenever your SGD asks for the next minibatch. You will then feed your SGD with the training minibatch that is specified by the indexes of its starting and ending rows. Your code shall never make a copy of any minibatch during SGD. 

## Sub-project 4: Train and evaluate the model. (50 points)

**(Sub-project 4 is mandatory for CSCD596, but is for extra credits for CSCD496)**

After you finished the above 3 sub-projects, play (automate your play via scripting) with your code with different choices of the hyperparameters and find which model (the $w$ vector) is the best one you will use. The model you will choose is the one that minimizes the out-of-sample error for the validation set. 

-  The set of tunable parameters for the SGD based logistic regression with L2 regularization: 

    - the degree of the $Z$ space

    - the $\lambda$ for regularization

    - the learning rate $\eta$

    - the number of iteration
    
    - the size of each minibatch


Plot and show the change of the in-sample errors and the out-of-sample errors while using a variety of different combinations of the hyperparameters listed above. You can decide your own way to plot the pictures with the principle being that the plot 1) shall show the error changes over the changes of hyperparameters, and 2) shall present why the one you chose is the best model.


Refer to the PLA discussion's slide that explains the under-fitting/overfitting concept that tells you which model to pick. Feel free to add necessary code in the `LogisticRegression` class to produce/save those in-sample/out-of-sample errors that you can plot and observe.

Report what you find in a PDF file that includes the plots and a reasonable amount of text that explains what you did and which model you chose and why. 




# Data normalization 

When you use gradient descent based methods, it is strongly suggested that your sample features be normalized. There are multiple different ways to normalize the data. The given notebook already gives the code for normalizing the data. 

**Please know that:** If you deploy your model for production, every future sample's features need to be normalized using **the same scale, range, formula, and min/max** that you used during the training phase. 

If you do not normalize the data, during the training phase, the gradient may become very large at some certain axis(es) and thus may cause over shooting. You can try it with the given data and you will see it. 


# Your submission:


- Compress `logistic_regression.py`  and the PDF report file into a zip file named as `YourLastName_Your6digitEWUID.zip`.

- Submit the zip file.

- Note: If you are a student in the CSCD496 section and you choose not to do the sub-project 4, your zip file will only include the `logistic_regression.py` file. 




# Misc.

- Also enclosed is a notebook for training/testing purpose. You do not have to use the provided notebook, if you choose to develop your own notebook for training/testing. 




