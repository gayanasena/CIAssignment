{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.13","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":3136,"databundleVersionId":26502,"sourceType":"competition"}],"dockerImageVersionId":30664,"isInternetEnabled":false,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [markdown]\n# **Import libs**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:10:38.228811Z\",\"iopub.execute_input\":\"2024-03-17T10:10:38.230260Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.757826Z\",\"shell.execute_reply.started\":\"2024-03-17T10:10:38.230200Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.756365Z\"}}\nimport numpy as np\nimport pandas as pd \n!pip install category_encoders\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# %% [markdown]\n# **Import Datasets**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.760827Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.762517Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.810514Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.762466Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.809117Z\"}}\n#import train data\nds_train = pd.read_csv(\"/kaggle/input/titanic/train.csv\")\nds_train.head()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.812018Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.812457Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.824561Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.812424Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.823423Z\"}}\nds_train.isnull().sum()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.827371Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.827904Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.851977Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.827870Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.850762Z\"}}\n#import test data\nds_test = pd.read_csv(\"/kaggle/input/titanic/test.csv\")\nds_test.head()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.853463Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.853866Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.869656Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.853834Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.868298Z\"}}\n# combine test and train datasets (for easy pre processing and then split before model training)\nds_train['train_data'] = 1\nds_test['train_data'] = 0\nds_test['Survived'] = 0\nds = pd.concat([ds_train,ds_test])\n\n# %% [markdown]\n# **Feature Scaling**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.871014Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.871353Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.891961Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.871323Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.890760Z\"}}\n# Get cabin type on logic - have a cabin(1) or not(0).\nds['cabin_multiple'] = ds.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))\n\n# Get cabin number pre identifing letter (if have a cabin else fill with 'n').\nds['cabin_adv'] = ds.Cabin.apply(lambda x: str(x)[0])\n\n# Get ticket pre token letter to 'ticket_letters' column, if full numeric ticket number then add 1 in 'numeric_ticket' column).\nds['numeric_ticket'] = ds.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)\nds['ticket_letters'] = ds.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)\n\n# Get name title from names\nds.Name.head(50)\nds['name_title'] = ds.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.893455Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.894002Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.903406Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.893970Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.902186Z\"}}\n# drop column Name  and Ticket  from ds (because replaced by preprocessed columns)\n\nds.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)\n\n# %% [markdown]\n# **Handle Null Values**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.905034Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.905458Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.920608Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.905428Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.919599Z\"}}\n# Replace NaN values\nds.Age = ds.Age.fillna(ds.Age.median())\nds.Fare = ds.Fare.fillna(ds.Fare.median())\n\n# Drop 'Embarked' column value missing 2 rows\nds.dropna(subset=['Embarked'],inplace = True)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.922119Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.922608Z\",\"iopub.status.idle\":\"2024-03-17T10:11:14.933028Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.922578Z\",\"shell.execute_reply\":\"2024-03-17T10:11:14.931782Z\"}}\nds.isnull().sum()\n\n# %% [markdown]\n# **Encoding Non numeric fields**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:14.937066Z\",\"iopub.execute_input\":\"2024-03-17T10:11:14.937398Z\",\"iopub.status.idle\":\"2024-03-17T10:11:16.064538Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:14.937368Z\",\"shell.execute_reply\":\"2024-03-17T10:11:16.063518Z\"}}\nimport category_encoders as ce\n\nencoder = ce.OrdinalEncoder(cols=['Sex','cabin_adv','ticket_letters','name_title','Embarked'])\nds = encoder.fit_transform(ds)\n\n# %% [markdown]\n# **Data scaling in complex data fields**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:16.065931Z\",\"iopub.execute_input\":\"2024-03-17T10:11:16.066407Z\",\"iopub.status.idle\":\"2024-03-17T10:11:16.114716Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:16.066376Z\",\"shell.execute_reply\":\"2024-03-17T10:11:16.113779Z\"}}\nfrom sklearn.preprocessing import StandardScaler\nscale = StandardScaler()\n\nds[['Age','Fare']] = scale.fit_transform(ds[['Age','Fare']])\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:16.116132Z\",\"iopub.execute_input\":\"2024-03-17T10:11:16.116755Z\",\"iopub.status.idle\":\"2024-03-17T10:11:16.135040Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:16.116717Z\",\"shell.execute_reply\":\"2024-03-17T10:11:16.133752Z\"}}\n# Preview\nds.head(5)\n\n# %% [markdown]\n# **Set Datasets**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:16.136674Z\",\"iopub.execute_input\":\"2024-03-17T10:11:16.137157Z\",\"iopub.status.idle\":\"2024-03-17T10:11:16.153005Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:16.137116Z\",\"shell.execute_reply\":\"2024-03-17T10:11:16.151653Z\"}}\n# re split combined dataset\nds_train_s= ds[ds.train_data == 1].drop(['train_data'], axis =1)\nds_test_s = ds[ds.train_data == 0].drop(['train_data','Survived'], axis =1)\n\n# define y axis(training set)\ny_train = ds_train_s[\"Survived\"]\n\n# define x axis(training set)\nx_train = ds_train_s[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','ticket_letters']]\n\n# define x axis(test set)\nx_test = ds_test_s[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','ticket_letters']]\n\n# %% [markdown]\n# **Model training**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:16.174137Z\",\"iopub.execute_input\":\"2024-03-17T10:11:16.175439Z\",\"iopub.status.idle\":\"2024-03-17T10:11:16.863318Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:16.175401Z\",\"shell.execute_reply\":\"2024-03-17T10:11:16.861761Z\"}}\nfrom sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=2)\nmodel.fit(x_train, y_train)\ny_pred = model.predict(x_test)\n\n# %% [markdown]\n# **Set Output**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:16.865070Z\",\"iopub.execute_input\":\"2024-03-17T10:11:16.865856Z\",\"iopub.status.idle\":\"2024-03-17T10:11:16.878200Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:16.865752Z\",\"shell.execute_reply\":\"2024-03-17T10:11:16.876796Z\"}}\noutput = pd.DataFrame({'PassengerId': ds_test_s.PassengerId, 'Survived': y_pred})\noutput.to_csv('submission.csv', index=False)\n\n\n# %% [markdown]\n# **Cross Validation**\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-17T10:11:17.376555Z\",\"iopub.execute_input\":\"2024-03-17T10:11:17.376903Z\",\"iopub.status.idle\":\"2024-03-17T10:11:18.782284Z\",\"shell.execute_reply.started\":\"2024-03-17T10:11:17.376873Z\",\"shell.execute_reply\":\"2024-03-17T10:11:18.781197Z\"}}\nfrom sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=2)\ncv = cross_val_score(rf,x_train, y_train,cv=5)\nprint(cv)\nprint(cv.mean())","metadata":{"_uuid":"35c43a73-0b1b-49e9-967f-8550018c91b5","_cell_guid":"34e8a161-e9c8-494c-8161-49624ee4019a","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}