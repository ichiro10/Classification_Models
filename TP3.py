#TP3

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


def data_preprocessing(data):
        print(data)
        print(data.info())
        print(data.nunique())
        print(data.describe(include='all'))

        #missed values 
        print(data.isnull().sum())

        #categorical data 
        categorical_features = ['Gender','Education','Marital Status','Home Ownership'] 

        for feature in categorical_features:
            # Label Encoding
            label_encoder = LabelEncoder()
            data[feature+'_encoded'] = label_encoder.fit_transform(data[feature])
            data = data.drop(columns=[feature])
        print(data)
        print(data.info())
        print(data.describe(include='all'))
        

def data_viz(data):
      
        d = data.drop(columns=['Credit Score'])

        for column in d.columns : 
            plt.figure(figsize = (14,4))
            sns.histplot(data[column])
            plt.title(column)
            plt.show()    
        
        for column in d:
                sns.boxplot(data = data, x = column)
                plt.show()    

        features_list = [feature for feature in data.columns.tolist() if feature != 'Credit Score']

        params = data[features_list]
        plt.figure(figsize=(10,10))
        sns.set_theme()
        sns.heatmap(params.corr(),annot=True, fmt="0.1g", cmap='PiYG')
        plt.show()           
      

def run(csv: str = './CS_Dataset.csv'):
        data = pd.read_csv(csv)
        print(data)
        print(data.info())

        X = data.drop('Credit Score', axis=1)
        y = data['Credit Score']
  

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Create transformers for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create a pipeline with the preprocessor and a classifier (Random Forest in this case)
        classifier = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])

        # Fit the pipeline on the training data
        classifier.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = classifier.predict(X_test)

        # Evaluate the model
        # You can use metrics like accuracy, precision, recall, etc.
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        if False:
            numerical_features = ['Age', 'Income', 'Number of Children']
            categorical_features = ['Gender', 'Education', 'Marital Status','Home Ownership']



            numerical_pipeline = make_pipeline(StandardScaler())
            categorical_pipeline = make_pipeline(LabelEncoder())
            # Updated the structure of the ColumnTransformer

            preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                    (categorical_pipeline, categorical_features))

            # List of classifiers
            clfs = [
                ('Logistic Regression', LogisticRegression()),
                ('SVM', SVC()),
                ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=3)),
                ('Random Forest', RandomForestClassifier()),
                ('DecisionTree', DecisionTreeClassifier()),
                ('GradientBoosting', GradientBoostingClassifier())
            ]

            # Iterate over classifiers
            for clf_name, clf in clfs:
                # Define the pipeline
                model = make_pipeline(
                    ('preprocessor', preprocessor),
                    (clf_name, clf)
                )
                model.fit(X_train, y_train)





            if False:
                # Cross-validate the model
                scores = cross_validate(pipeline, X_train, y_train, scoring="accuracy", cv=5)

                # Print results
                print('---------------------------------')
                print(clf_name)
                print('-----------------------------------')
                for key, values in scores.items():
                    print(key, ' mean ', values.mean())
                    print(key, ' std ', values.std())







 





if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(folder=sys.argv[1])
    else:
        run()        