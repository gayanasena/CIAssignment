# %% [markdown]
# **Import libs**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:10:38.228811Z","iopub.execute_input":"2024-03-17T10:10:38.230260Z","iopub.status.idle":"2024-03-17T10:11:14.757826Z","shell.execute_reply.started":"2024-03-17T10:10:38.230200Z","shell.execute_reply":"2024-03-17T10:11:14.756365Z"}}
import numpy as np
import pandas as pd 
!pip install category_encoders

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [markdown]
# **Import Datasets**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.760827Z","iopub.execute_input":"2024-03-17T10:11:14.762517Z","iopub.status.idle":"2024-03-17T10:11:14.810514Z","shell.execute_reply.started":"2024-03-17T10:11:14.762466Z","shell.execute_reply":"2024-03-17T10:11:14.809117Z"}}
#import train data
ds_train = pd.read_csv("/kaggle/input/titanic/train.csv")
ds_train.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.812018Z","iopub.execute_input":"2024-03-17T10:11:14.812457Z","iopub.status.idle":"2024-03-17T10:11:14.824561Z","shell.execute_reply.started":"2024-03-17T10:11:14.812424Z","shell.execute_reply":"2024-03-17T10:11:14.823423Z"}}
ds_train.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.827371Z","iopub.execute_input":"2024-03-17T10:11:14.827904Z","iopub.status.idle":"2024-03-17T10:11:14.851977Z","shell.execute_reply.started":"2024-03-17T10:11:14.827870Z","shell.execute_reply":"2024-03-17T10:11:14.850762Z"}}
#import test data
ds_test = pd.read_csv("/kaggle/input/titanic/test.csv")
ds_test.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.853463Z","iopub.execute_input":"2024-03-17T10:11:14.853866Z","iopub.status.idle":"2024-03-17T10:11:14.869656Z","shell.execute_reply.started":"2024-03-17T10:11:14.853834Z","shell.execute_reply":"2024-03-17T10:11:14.868298Z"}}
# combine test and train datasets (for easy pre processing and then split before model training)
ds_train['train_data'] = 1
ds_test['train_data'] = 0
ds_test['Survived'] = 0
ds = pd.concat([ds_train,ds_test])

# %% [markdown]
# **Feature Scaling**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.871014Z","iopub.execute_input":"2024-03-17T10:11:14.871353Z","iopub.status.idle":"2024-03-17T10:11:14.891961Z","shell.execute_reply.started":"2024-03-17T10:11:14.871323Z","shell.execute_reply":"2024-03-17T10:11:14.890760Z"}}
# Get cabin type on logic - have a cabin(1) or not(0).
ds['cabin_multiple'] = ds.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

# Get cabin number pre identifing letter (if have a cabin else fill with 'n').
ds['cabin_adv'] = ds.Cabin.apply(lambda x: str(x)[0])

# Get ticket pre token letter to 'ticket_letters' column, if full numeric ticket number then add 1 in 'numeric_ticket' column).
ds['numeric_ticket'] = ds.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
ds['ticket_letters'] = ds.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)

# Get name title from names
ds.Name.head(50)
ds['name_title'] = ds.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.893455Z","iopub.execute_input":"2024-03-17T10:11:14.894002Z","iopub.status.idle":"2024-03-17T10:11:14.903406Z","shell.execute_reply.started":"2024-03-17T10:11:14.893970Z","shell.execute_reply":"2024-03-17T10:11:14.902186Z"}}
# drop column Name  and Ticket  from ds (because replaced by preprocessed columns)

ds.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# %% [markdown]
# **Handle Null Values**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.905034Z","iopub.execute_input":"2024-03-17T10:11:14.905458Z","iopub.status.idle":"2024-03-17T10:11:14.920608Z","shell.execute_reply.started":"2024-03-17T10:11:14.905428Z","shell.execute_reply":"2024-03-17T10:11:14.919599Z"}}
# Replace NaN values
ds.Age = ds.Age.fillna(ds.Age.median())
ds.Fare = ds.Fare.fillna(ds.Fare.median())

# Drop 'Embarked' column value missing 2 rows
ds.dropna(subset=['Embarked'],inplace = True)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.922119Z","iopub.execute_input":"2024-03-17T10:11:14.922608Z","iopub.status.idle":"2024-03-17T10:11:14.933028Z","shell.execute_reply.started":"2024-03-17T10:11:14.922578Z","shell.execute_reply":"2024-03-17T10:11:14.931782Z"}}
ds.isnull().sum()

# %% [markdown]
# **Encoding Non numeric fields**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:14.937066Z","iopub.execute_input":"2024-03-17T10:11:14.937398Z","iopub.status.idle":"2024-03-17T10:11:16.064538Z","shell.execute_reply.started":"2024-03-17T10:11:14.937368Z","shell.execute_reply":"2024-03-17T10:11:16.063518Z"}}
import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['Sex','cabin_adv','ticket_letters','name_title','Embarked'])
ds = encoder.fit_transform(ds)

# %% [markdown]
# **Data scaling in complex data fields**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:16.065931Z","iopub.execute_input":"2024-03-17T10:11:16.066407Z","iopub.status.idle":"2024-03-17T10:11:16.114716Z","shell.execute_reply.started":"2024-03-17T10:11:16.066376Z","shell.execute_reply":"2024-03-17T10:11:16.113779Z"}}
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

ds[['Age','Fare']] = scale.fit_transform(ds[['Age','Fare']])

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:16.116132Z","iopub.execute_input":"2024-03-17T10:11:16.116755Z","iopub.status.idle":"2024-03-17T10:11:16.135040Z","shell.execute_reply.started":"2024-03-17T10:11:16.116717Z","shell.execute_reply":"2024-03-17T10:11:16.133752Z"}}
# Preview
ds.head(5)

# %% [markdown]
# **Set Datasets**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:16.136674Z","iopub.execute_input":"2024-03-17T10:11:16.137157Z","iopub.status.idle":"2024-03-17T10:11:16.153005Z","shell.execute_reply.started":"2024-03-17T10:11:16.137116Z","shell.execute_reply":"2024-03-17T10:11:16.151653Z"}}
# re split combined dataset
ds_train_s= ds[ds.train_data == 1].drop(['train_data'], axis =1)
ds_test_s = ds[ds.train_data == 0].drop(['train_data','Survived'], axis =1)

# define y axis(training set)
y_train = ds_train_s["Survived"]

# define x axis(training set)
x_train = ds_train_s[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','ticket_letters']]

# define x axis(test set)
x_test = ds_test_s[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','ticket_letters']]

# %% [markdown]
# **Model training**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:16.174137Z","iopub.execute_input":"2024-03-17T10:11:16.175439Z","iopub.status.idle":"2024-03-17T10:11:16.863318Z","shell.execute_reply.started":"2024-03-17T10:11:16.175401Z","shell.execute_reply":"2024-03-17T10:11:16.861761Z"}}
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=2)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# %% [markdown]
# **Set Output**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:16.865070Z","iopub.execute_input":"2024-03-17T10:11:16.865856Z","iopub.status.idle":"2024-03-17T10:11:16.878200Z","shell.execute_reply.started":"2024-03-17T10:11:16.865752Z","shell.execute_reply":"2024-03-17T10:11:16.876796Z"}}
output = pd.DataFrame({'PassengerId': ds_test_s.PassengerId, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)


# %% [markdown]
# **Cross Validation**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-17T10:11:17.376555Z","iopub.execute_input":"2024-03-17T10:11:17.376903Z","iopub.status.idle":"2024-03-17T10:11:18.782284Z","shell.execute_reply.started":"2024-03-17T10:11:17.376873Z","shell.execute_reply":"2024-03-17T10:11:18.781197Z"}}
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=2)
cv = cross_val_score(rf,x_train, y_train,cv=5)
print(cv)
print(cv.mean())
