import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lime import lime_tabular

X = pd.read_csv('data/X_data_bank_churn.csv', delimiter=',')
y = pd.read_csv('data/y_data_bank_churn.csv', delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_pred=X.iloc[8000:10000,:]
print(X_pred.shape)

classifier1=pd.read_pickle('results/svm_rbf_mod.pkl')

interpretor = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode='classification'
)

def get_exp(raw_data):
    exp_churn = interpretor.explain_instance(
        data_row=raw_data,  ##new data
        predict_fn=classifier1.predict_proba
    )
    rs = exp_churn.as_list()
    ratio=exp_churn.predict_proba[1]
    return ratio,rs


predictions=[]
ratios=[]
details=[]
for i in range(2000):
    predictions.append(classifier1.predict(X_pred.iloc[[i],:])[0])
    ratio,detail=get_exp(X_pred.iloc[i])
    ratios.append(ratio)
    details.append(detail)

df_lime_all=pd.DataFrame({'predictions':predictions,'ratios':ratios,'details':details})

df_4_pre=pd.concat([X_pred,df_lime_all], axis=1)

df_4_pre.to_csv('results/LIME_predictions.csv', index=False)
