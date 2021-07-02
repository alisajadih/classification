
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#
#  change this line as you wish
model_names = ['RandomForest', 'KNN', 'NaiveBayes', 'DecisionTree']


def read_model(path):
    f = open(path, 'rb')
    classifier = pickle.load(f)
    f.close
    return classifier


def run_test(x_test, y_test):
    result = []
    for model_name in model_names:
        path = './Classifications'+'/'+model_name+'.pickle'
        classifier = read_model(path)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        result.append((model_name, accuracy))
    result = sorted(result, key=lambda el: el[1], reverse=True)
    return result


data = pd.read_excel('./dataset.xls')

data.drop('Unnamed: 0', axis=1, inplace=True)


close_value_mean = data['Close_Value'].mean()
data['Close_Value'].fillna(close_value_mean, inplace=True)


index_names = data[data['Stage'] == 'In Progress'].index
data.drop(index_names , inplace=True)

data['Stage'] = data['Stage'].astype('category').cat.codes
data['Customer'] = data['Customer'].astype('category').cat.codes
data['Agent'] = data['Agent'].astype('category').cat.codes
data['SalesAgentEmailID'] = data['SalesAgentEmailID'].astype(
    'category').cat.codes
data['ContactEmailID'] = data['ContactEmailID'].astype('category').cat.codes
data['Product'] = data['Product'].astype('category').cat.codes


data['Created Date'] = data['Created Date'].astype(int)
data['Close Date'] = data['Close Date'].astype(int)

X = data.drop('Stage', axis=1)
y = data['Stage']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=1)  # 85% training and 15% test

print(run_test(X_test, y_test))
