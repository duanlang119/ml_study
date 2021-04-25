import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# data preprocessing:
def data_prep():
    # Importing the dataset
    data = pd.read_csv('../data/bank_churn_data.csv', delimiter=',')
    # Drop the columns as explained above
    # df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    X = data.iloc[:, 3:13]
    y = data.iloc[:, 13]
    
    # replace one hot coding
    # geography=pd.get_dummies(X["Geography"],drop_first=True)
    # gender=pd.get_dummies(X['Gender'],drop_first=True)
    # X = pd.concat([X, geography, gender], axis=1)
    # X = X.drop(['Geography', 'Gender'], axis=1)

    # One hot encode the categorical variables
    lst = ['Geography', 'Gender']
    remove = list()
    for i in lst:
        if (X[i].dtype == np.str or X[i].dtype == np.object):
            for j in X[i].unique():
                X[i + '_' + j] = np.where(X[i] == j, 1, -1)
            remove.append(i)
    X = X.drop(remove, axis=1)
    

    # add 3 new items
    X['BalanceSalaryRatio'] = X.Balance / X.EstimatedSalary
    # Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
    X['TenureByAge'] = X.Tenure / (X.Age)
    '''Lastly we introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
    :-)'''
    X['CreditScoreGivenAge'] = X.CreditScore / (X.Age)

    data_cols=X.columns.to_list()

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_pp = scaler.transform(X)
    X_f=pd.DataFrame(X_pp,columns=data_cols)

    X_f.to_csv('data/X_data_bank_churn.csv', index=False)
    y.to_csv('data/y_data_bank_churn.csv', index=False)


# data preprocessing:
def data_prep_without_Scaler():
    # Importing the dataset
    data = pd.read_csv('../data/bank_churn_data.csv', delimiter=',')
    # Drop the columns as explained above
    # df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    X = data.iloc[:, 3:13]
    y = data.iloc[:, 13]

    # One hot encode the categorical variables
    lst = ['Geography', 'Gender']
    remove = list()
    for i in lst:
        if (X[i].dtype == np.str or X[i].dtype == np.object):
            for j in X[i].unique():
                X[i + '_' + j] = np.where(X[i] == j, 1, -1)
            remove.append(i)
    X = X.drop(remove, axis=1)

    # add 3 new items
    X['BalanceSalaryRatio'] = X.Balance / X.EstimatedSalary
    # Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
    X['TenureByAge'] = X.Tenure / (X.Age)
    '''Lastly we introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
    :-)'''
    X['CreditScoreGivenAge'] = X.CreditScore / (X.Age)

    return X,y

def X_shape_saver():
    X,y=data_prep_without_Scaler()
    continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                       'BalanceSalaryRatio',
                       'TenureByAge', 'CreditScoreGivenAge']
    maxVec_list=[]
    minVec_list=[]
    for continuou in continuous_vars:
        # minMax scaling the continuous variables
        minVec = X[continuou].min()
        maxVec = X[continuou].max()
        minVec_list.append(minVec)
        maxVec_list.append(maxVec)

    df = pd.DataFrame({'continuous_vars': continuous_vars, 'minVec': minVec_list, 'maxVec': maxVec_list})
    df.to_csv('data/df_minVec_and_maxVec.csv', index=False)

def test_min_ori():
    df_ori=pd.read_csv('../data/bank_churn_data.csv', delimiter=',')
    df_fin=pd.read_csv('../data/X_data_bank_churn.csv', delimiter=',')
    df_max=pd.read_csv('../data/df_minVec_and_maxVec.csv', delimiter=',')
    df_new=df_ori['CustomerId']
    continuous_vars=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    # continuous_vars=['CreditScore']
    for cont in continuous_vars:
        max_info=df_max[df_max.continuous_vars==cont]
        minVec=max_info.minVec.tolist()[0]
        maxVec=max_info.maxVec.tolist()[0]
        print(cont)
        print(minVec)
        print(maxVec)

        df_ori[cont+'_ori']=  df_ori[cont]
        df_fin[cont+'_fin']=  df_fin[cont]

        df_new=pd.concat([df_new,df_fin[cont+'_fin'],df_ori[cont+'_ori']], axis=1)
        df_new[cont+'_calc'] = (df_new[cont+'_fin']) * (maxVec - minVec) + minVec

    df_new.to_csv('data/test_minVec_and_maxVec.csv', index=False)




# demise
def split_csv(x_file,y_file):
    X = pd.read_csv('../data/X_data_bank_churn.csv', delimiter=',')
    y = pd.read_csv('../data/y_data_bank_churn.csv', delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.head())
    print(y_train.head())
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)

    svm.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test

# split_csv('data/X_data_bank_churn.csv','data/y_data_bank_churn.csv')


test_min_ori()



