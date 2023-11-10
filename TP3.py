#TP3

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def predictionVStrue(y_true, y_pred, mse, r2 ,name ,strategy):
    plt.scatter(y_true, y_pred, label=f'{name},{strategy}\nMSE: {mse:.2f}, R-squared: {r2:.2f}')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red', label='Perfect Prediction')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.legend()
    plt.title("Regression Model Predictions")
    plt.show()    


def residual_plot(y_true, y_pred, model_name):
        plt.plot(y_pred, y_true - y_pred, "*")
        plt.plot(y_pred, np.zeros_like(y_pred), "-")
        plt.legend(["Data", "Perfection"])
        plt.title("Residual Plot of " + model_name)
        plt.xlabel("Predicted Value")
        plt.ylabel("Residual")
        plt.show()   


def Preparing_models(X_train, X_test, y_train):
        # Créez une liste pour stocker les modèles
        models = []
        

        return models

def data_preprocessing(data):
        print(data.info())
        print(data.nunique())
        print(data.describe(include='all'))

        #data_viz(data)
        #missed values 
        print(data.isnull().sum())

        #categorical data 
        

       
        #outliers 
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=100)

        for col in data.columns:
            data[col] = data[col] = qt.fit_transform(pd.DataFrame(data[col]))
        

        for col in data:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                whisker_width = 1.5
                lower_whisker = q1 - (whisker_width * iqr)
                upper_whisker = q3 + whisker_width * iqr
                data[col] = np.where(data[col] > upper_whisker, upper_whisker, np.where(data[col] < lower_whisker, lower_whisker, data[col]))
       

        scalar = StandardScaler()
        scalar.fit(params)
        scaled_inputs = scalar.transform(params)

        return scaled_inputs, target

def data_viz(data):
      for column in data.columns : 
        plt.figure(figsize = (14,4))
        sns.histplot(data[column])
        plt.title(column)
        plt.show()    

      for column in data:
                sns.boxplot(data = data, x = column)
                plt.show()    

      params = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',  'TAX',
        'PTRATIO', 'B', 'LSTAT','RAD']]
      plt.figure(figsize=(10,10))
      sns.set_theme()
      sns.heatmap(params.corr(),annot=True, fmt="0.1g", cmap='PiYG')
      plt.show()            

      
data = pd.read_csv('c:/Users/ghamm/OneDrive/Bureau/UQAC/TP/AA/boston.csv')
plt.figure(figsize=(12, 6))

scaled_inputs, target = data_preprocessing(data)
#split the data
X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, target, test_size=0.2, random_state = 42)

# Loop through the baseline strategies and evaluate dummy regressors

models= Preparing_models(X_train, X_test, y_train)
for name, model in models:
    y_pred = model.predict(X_test)