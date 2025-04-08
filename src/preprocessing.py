import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import pickle
from sklearn.pipeline import Pipeline

df_sleep = pd.read_csv('/Users/dfarradj/Desktop/USF/601msds/regression_f24/Final Project/Health_Sleep_Statistics.csv')
df_sleep.head(3)

df_sleep.columns = df_sleep.columns.str.replace(' ', '_')
df_sleep.isnull().sum()

pd.plotting.scatter_matrix(df_sleep, diagonal='kde', figsize=(12,8))

from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings("ignore")
X = df_sleep.drop(columns = ['Sleep_Quality', 'User_ID', 'Medication_Usage'])
y = df_sleep['Sleep_Quality']
X = pd.get_dummies(X, drop_first = True)
X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25)

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include = ['int', 'float'])),
        ("cat", categorical_transformer, make_column_selector(dtype_exclude = ['int', 'float'])),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

# Create new train and test data using the pipeline
clf.fit(X_train, y_train)
train_new = clf.transform(X_train)
test_new = clf.transform(X_test)

# Transform to dataframe and save as a csv
train_new = pd.DataFrame(train_new)
test_new = pd.DataFrame(test_new)
train_new['y'] = y_train
test_new['y'] = y_test

train_new.to_csv('data/processed_train_data.csv')
test_new.to_csv('data/processed_test_data.csv')


with open('data/pipeline.pkl','wb') as f:
    pickle.dump(clf,f)