# Stock Market Prediction

There is never a right way to predict the direction of the Stock. Our goal was to use already existing **Machine Learning** algorithms.

For this project we constructed and tested 4 Stock Prediction Algorithms. 

# RNN Binary Classification
**Libraries used:**
- **pandas**
  - Used to gather financial/sentiment data using pandas datareader and read_csv.
- **numpy**
  - Used to manipulate arrays and prime the data to passed into the model, which only accepts numpy arrays as input.
- **Tensorflow.Keras**
  - Imported the tensorflow for access to Keras' layers (Input, LSTM, and Dense) and Keras' Model function.
- **Matplotlib.PyPlot**
  - Used to plot the loss and accuracy of training and testing.
- **Sklearn.preprocessing**
  - Used for the StandardSkalar() to standardize the values in our dataset (i.e., standardizing open, high, low, volume, etc.)


- ## Summary

  - First, we collected the data and formatted it into a numpy array of shape T (time/date), by D (number of features).
  - Secondly, we created the model. The model was implemented using Keras.Model to create a RNN consisting of an Input layer, a LSTM layer, and Dense output layer. 
  - Next, we tuned our hyper parameters (number of hidden nodes in the LSTM, number of epochs, and the learning rate) on both the model with sentiment and without sentiment.
  -  Finally, we fitted our models and plotted the results.


# Linear Regression Model & Support Vector Regression (SVM)

**Libraries used:**
- **quandl**
  - This was used to gather financial data using its API calls
- **numpy**
  - This was used to manipulate arrays and the dataframe containing our Financial Data
- **sklearn.linear_model**
  - Imported our linear regression model
- **sklearn.svm**
  - Imported our Support Vector Regression model
- **sklearn.model_selection**
  - Used to train our models


- ## Summary

  - Upon importing the financial data, the dataframe was rewritten to fit the parameters needed. For these models, the main coadlumn that was used was _Adj. Close_. 

  - After rewriting the dataframe, we then proceeded to create a variable that stored the number of days (_nday_). Next we adjusted a new column with more rows that would store the predictions.

  - Then we created a new _INDEPENDENT_ dataset. This was our X data.
    - This was done by converting the dataframe into a numpy array from the column _Adj. Close_.
   
  - Next, we created a new _DEPENDENT_ dataset. This was our Y data AKA our Target Data.
    - This was done by converting the dataframe into a numpy array from the _Stock Prediction_.
  - After gathering the data, we put these into our training set
    - train_test_split(x, y, test_size)


  - CREATE SUPPORT VECTOR MACHINE (REGRESSION) MODEL
    - MODEL ATTEMPTS TO PREDICT A STOCK PRICE IN THE FUTURE
    
    1. USING 3 PARAMETERS
        - KERNEL (TYPE OF MATHEMATICAL FUNCTION TO USE)
            - USING RBF FOR THIS MODEL -- RADIAL BASIS FUNCTION -- GAUSSIAN FUNCTION
            - https://scikit-learn.org/stable/modules/svm.html
            - https://www.cs.princeton.edu/sites/default/files/uploads/saahil_madge.pdf
            
        - C
            - HELPS WITH THE MISCLASSIFICATION OF TRAINING EXAMPLES
            - CHOOSING A LOW INT CREATES A SMOOTH DECISION
            - CHOOSING A HIGH INT CREATES THE CLASSIFICATION OF DATA MORE ACCURATELY
            
        - GAMMA
            - DECIDING FACTOR OF HOW MUCH INFLUENCE A SINGLE TRAINING EXAMPLE HAS
            
        - USED THE FOLLOWING TO CORRECTLY CHOOSE A PROPER C AND GAMMA VALUES
            - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        
    2. THEN REFIT THE DATA WITH THE SVM MACHINE USING .fit()
        - .fit(1, 2)
            - 1 --> TRAINING VECTOR WHERE N SAMPLES IS THE NUMBER OF SAMPLES
            - 2 --> TARGET IS RELATIVE TO X FOR REGRESSION
   - GET THE SCORE OF THE SVM MODEL -- REGRESSION
      - SCORE Returns the coefficient of determination R^2 of the prediction.
          - BEST POSSIBLE SCORE IS 1.0 -- CAN BE NEGATIVE
          - .score(1, 2) PARAMETERS
            - 1 --> TRAINING VECTOR WHERE N SAMPLES IS THE NUMBER OF SAMPLES
            - 2 --> TARGET IS RELATIVE TO X FOR REGRESSION

  -  CREATE LINEAR REGRESSION MODEL
      - GET THE SCORE OF THE LINEAR REGRESSION MODEL
        1. SCORE Returns the coefficient of determination R^2 of the prediction.
        - BEST POSSIBLE SCORE IS 1.0 -- CAN BE NEGATIVE
        - .score(1, 2) PARAMETERS
            - 1 --> TRAINING VECTOR WHERE N SAMPLES IS THE NUMBER OF SAMPLES
            - 2 --> TARGET IS RELATIVE TO X FOR REGRESSION
