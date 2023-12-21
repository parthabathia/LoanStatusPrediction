# Loan Status Prediction

This Python code is a machine learning script that aims to predict loan approval status based on various features using a Support Vector Machine (SVM) classifier. Below is a step-by-step description of the code:

    Import Libraries:
        numpy, pandas: Importing libraries for numerical operations and data manipulation.
        seaborn: Used for data visualization.
        svm from sklearn: Imports the Support Vector Machine classifier.
        StandardScaler from sklearn.preprocessing: Used for standardizing feature values.
        train_test_split from sklearn.model_selection: Splits the dataset into training and testing sets.
        accuracy_score from sklearn.metrics: Measures the accuracy of the machine learning model.

    Load and Inspect Data:
        Reads a CSV file named 'loan.csv' into a Pandas DataFrame (loan_dataset).
        Displays the first few rows of the dataset using head() and checks for missing values with isnull().sum().

    Data Cleaning and Preprocessing:
        Drops rows with missing values using dropna().
        Replaces categorical labels in the 'Loan_Status' column ('N' for No, 'Y' for Yes) with numerical values (0 and 1).
        Converts the 'Dependents' column values to numeric, replacing '3+' with 4.

    Data Visualization:
        Visualizes the distribution of loan approval status based on education and marital status using Seaborn's countplot.
        Displays a heatmap of the correlation matrix for numerical features.

    Feature Engineering:
        Converts categorical features ('Married', 'Gender', 'Self_Employed', 'Property_Area', 'Education') into numeric format.

    Data Splitting:
        Splits the dataset into features (X) and the target variable (Y).
        Further splits the data into training and testing sets using train_test_split.

    Model Training:
        Initializes an SVM classifier with a linear kernel.
        Fits the classifier on the training data.

    Model Evaluation on Training Set:
        Predicts the loan approval status on the training set.
        Calculates and prints the accuracy of the model on the training set.

    Model Evaluation on Testing Set:
        Predicts the loan approval status on the testing set.
        Calculates and prints the accuracy of the model on the testing set.

    Prediction on New Data:
        Takes a subset of the training data (rows 367 to 375) as new input.
        Predicts the loan approval status for this subset.
        Calculates and prints the accuracy of the predictions.

    Comparison with Ground Truth:
        Prints the ground truth loan approval status for the subset used in the new data prediction.

Note: The script assumes a linear SVM kernel and utilizes standardization for numeric features. The accuracy scores provide an indication of how well the SVM model performs on the training, testing, and new data subsets.
