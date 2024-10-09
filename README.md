import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#for preprocessing
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#for evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

path = r"D:\Social_Network_Ads.csv"
data = pd.read_csv(path)

df = pd.DataFrame(data)
print(df)
print('_____________________________')
print(df.head(3))
print('_____________________________')
print(df.describe())
print('_____________________________')
print(df.info)
print('_____________________________')
print(df.columns)
print('_____________________________')
print(df.isna().sum())
print('_____________________________')
print('_____________________________')
print('_____________________________')
df["Gender"] = df["Gender"].replace({"Male":1,"Female":0})

print('_____________________________')
print(df)


x = df.drop(columns=["Purchased"])

y = df["Purchased"]



standard = StandardScaler()
x_norm= standard.fit_transform(x)

x_train ,x_test , y_train , y_test = train_test_split(x_norm,y,random_state=42)

model = LogisticRegression()

model.fit(x_train,y_train)
y_prec = model.predict(x_test)

acc = metrics.accuracy_score(y_test, y_prec)
print(acc)
####   0.88 percent accuracy
