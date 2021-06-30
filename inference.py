
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# change this line as you wish
BEST_MODEL_PATH = "./Classifications/RandomForest.pickle"


def inference(path):
    '''
    path: a DataFrame
    result is the output of function which should be 
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    '''
    f = open(BEST_MODEL_PATH, 'rb')
    classifier = pickle.load(f)
    f.close()
    print(classifier)
    result = list(classifier.predict(path))
    # your code starts here
    # model.predict()

    # your code ends here
    return result


data = pd.read_excel('./dataset.xls')
data.drop('Unnamed: 0', axis=1, inplace=True)
close_value_mean = data['Close_Value'].mean()
data['Close_Value'].fillna(close_value_mean, inplace=True)
index_names = data[data['Stage'] == 'In Progress'].index
data.drop(index_names, inplace=True)

data['Stage'] = data['Stage'].astype('category')
data['Customer'] = data['Customer'].astype('category')
data['Agent'] = data['Agent'].astype('category')
data['SalesAgentEmailID'] = data['SalesAgentEmailID'].astype('category')
data['ContactEmailID'] = data['ContactEmailID'].astype('category')
data['Product'] = data['Product'].astype('category')

data['Stage']=data['Stage'].cat.codes
data['Customer']=data['Customer'].cat.codes
data['Agent']=data['Agent'].cat.codes
data['SalesAgentEmailID']=data['SalesAgentEmailID'].cat.codes
data['ContactEmailID']=data['ContactEmailID'].cat.codes
data['Product']=data['Product'].cat.codes

data['Created Date'] = data['Created Date'].astype(int)
data['Close Date'] = data['Close Date'].astype(int)


X = data.drop('Stage', axis=1)
y = data['Stage']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=1)  # 85% training and 15% test

print(inference(X_test))
