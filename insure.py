#%%
#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
import scipy
import missingno as mso
# %%
#More Libraries
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# %%
#importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
# %%
#Load Data
df = pd.read_csv('insurance_set.csv')
# %%
df.head()
# %%
df.describe()
# %%
print('Rows: ', len(df))
print('Columns: ', df.shape[1])
# %%
#Count of missing values
df.isnull().sum()
# %%
#Data Types
df.dtypes
#%%
def Cat_dist(variable):
    var = df[variable]
    varValue = var.value_counts(dropna=False)
    print(varValue)
#%%
Cat_Var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']
for v in Cat_Var:
    Cat_dist(v)
# %%
#Explaratory Analysis
labels = df['Gender'].value_counts(dropna=False).index
sizes = df['Gender'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Gender: Male = M, Female = F',color = 'black',fontsize = 15)
# %%
labels = df['Married'].value_counts(dropna=False).index
sizes = df['Married'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Marital Status: Married = Yes, Single = NO',color = 'black',fontsize = 15)
# %%
sns.countplot(x="Gender", data=df, palette="hls")
plt.show()
# %%
def bar_plot(variable):
    var = df[variable]
    varValue = var.value_counts()
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable,varValue))
# %%
sns.set_style('darkgrid')
categorical_variables = ['Education', 'Property_Area', 'Loan_Status']
for v in categorical_variables:
    bar_plot(v)
# %%
labels = df['Dependents'].value_counts(dropna=False).index
sizes = df['Dependents'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Dependents:0 person, 1 person, 2 person, 3 person and above',color = 'black',fontsize = 15)
# %%
labels = df['Self_Employed'].value_counts(dropna=False).index
sizes = df['Self_Employed'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Self Employed or Employee: Self Emplyed = Yes, Employee = NO',color = 'black',fontsize = 15)
# %%
labels = df['Credit_History'].value_counts(dropna=False).index
sizes = df['Credit_History'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Credit_History: Good Credit = 1.0, Bad Credit = 0.0',color = 'black',fontsize = 15)
# %%
#Loan Duration
df.Loan_Amount_Term.value_counts(dropna=False)
# %%
sns.countplot(x="Loan_Amount_Term", data=df, palette='inferno_r')
plt.show()
# %%
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with histogram".format(variable))
    plt.show()
# %%
numerical_variables = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for m in numerical_variables:
    plot_hist(m)
# %%
#Further Analysis
#Relationship Between Variables
#Gender vs Married
pd.crosstab(df.Gender,df.Married).plot(kind="bar", stacked=True, figsize=(5,5), color=['#f67f58','#12c2e8'])
plt.title('Gender vs Married')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()
# %%
#Property Area vs Credit History
pd.crosstab(df.Property_Area,df.Credit_History).plot(kind="bar", stacked=True, figsize=(5,5), color=['#654a7d','#ffd459'])
plt.title('Property Area vs Credit History')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.legend(["Bad Credit", "Good Credit"])
plt.xticks(rotation=0)
plt.show()
# %%
#Property Area vs Loan Status
pd.crosstab(df.Property_Area,df.Loan_Status).plot(kind="bar", stacked=True, figsize=(5,5), color=['#CD0000','#308014'])
plt.title('Property Area vs Loan Status')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.legend(["Not Approved", "Approved"])
plt.xticks(rotation=0)
plt.show()
# %%
#Marital Status vs Loan Status
pd.crosstab(df.Married,df.Loan_Status).plot(kind="bar", stacked=True, figsize=(5,5), color=['#8A2BE2','#53868B'])
plt.title('Marital Status vs Loan Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency')
plt.legend(["Not Approved", "Approved"])
plt.xticks(rotation=0)
plt.show()
# %%
#Income vs Loan Approval
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df, palette='twilight_r')
# %%
sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=df, palette='gist_rainbow_r')
# %%
#Loan Status vs Loan Amount
sns.swarmplot(x=df['Loan_Status'],
              y=df['LoanAmount'])
# %%
sns.regplot(x=df['ApplicantIncome'], y=df['LoanAmount'])
# %%
#Data Processing
#Droping Loan ID
df = df.drop(['Loan_ID'], axis = 1)
# %%
#Replacing Missing Values
#Categorical Variables
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
# %%
#Replacing Missing Values
#Numerical Variables
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
# %%
df = pd.get_dummies(df)

# Drop categorical variable with pair strings columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis = 1)

# Rename column's name
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}
       
df.rename(columns=new, inplace=True)


# %%
# Removing Outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
# %%
#Adjusting Distribution of Numerical Variable
# Square Root Transformation

df.ApplicantIncome = np.sqrt(df.ApplicantIncome)
df.CoapplicantIncome = np.sqrt(df.CoapplicantIncome)
df.LoanAmount = np.sqrt(df.LoanAmount)
# %%
#Checking The New Distribution
numerical_variables = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for m in numerical_variables:
    plot_hist(m)
# %%
#Model Preparation
#independent vs Target
X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]
# %%
#Balancing Loan Status to avoid overfitting
X, y = SMOTE().fit_resample(X, y)
# %%
sns.set_theme(style="darkgrid")
sns.countplot(y=y, data=df, palette='ocean_r')
plt.ylabel('Loan Status')
plt.xlabel('Total')
plt.show()
# %%
#Data Normalization
X = MinMaxScaler().fit_transform(X)
# %%
#Splitting Data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# %%
model_lgr = 'Logistic Regression'
lr = LogisticRegression(solver='saga', max_iter=500, random_state=1)
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("confussion matrix")
print(lr_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Logistic Regression: {:.2f}%".format(lr_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,lr_predict))
# %%
model_nb = 'Naive Bayes'
nb = CategoricalNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
print("confussion matrix")
print(nb_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Naive Bayes model:{:.2f}%".format(nb_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,nbpred))
# %%
model_rfc = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=1000, random_state=12,max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Random Forest:{:.2f}%".format(rf_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,rf_predicted))
# %%
model_egb = 'Extreme Gradient Boost'
xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
xgb.fit(X_train, y_train)
xgb_predicted = xgb.predict(X_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)
print("confussion matrix")
print(xgb_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Extreme Gradient Boost:{:.2f}%".format(xgb_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,xgb_predicted))
# %%
model_knn = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("-------------------------------------------")
print("Accuracy of K-NeighborsClassifier:{:.2f}%".format(knn_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,knn_predicted))
# %%
model_dtc = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("-------------------------------------------")
print("Accuracy of DecisionTreeClassifier:{:.2f}%".format(dt_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,dt_predicted))
# %%
model_svc = 'Support Vector Machine'
svc =  SVC(kernel='rbf', max_iter=500)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Support Vector Classifier:{:.2f}%".format(svc_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,svc_predicted))
# %%
model_sgd = 'Stochastic Gradient Descent'
sgdc = SGDClassifier(max_iter=5000, random_state=0)
sgdc.fit(X_train, y_train)
sgdc_predicted = sgdc.predict(X_test)
sgdc_conf_matrix = confusion_matrix(y_test, sgdc_predicted)
sgdc_acc_score = accuracy_score(y_test, sgdc_predicted)
print("confussion matrix")
print(sgdc_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Stochastic Gradient Descent:{:.2f}%".format(sgdc_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,sgdc_predicted))
# %%
model_nn = 'Neural Nets'
mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
mlpc.fit(X_train, y_train)
mlpc_predicted = mlpc.predict(X_test)
mlpc_conf_matrix = confusion_matrix(y_test, mlpc_predicted)
mlpc_acc_score = accuracy_score(y_test, mlpc_predicted)
print("confussion matrix")
print(mlpc_conf_matrix)
print("-------------------------------------------")
print("Accuracy of : MLP Classifier{:.2f}%".format(mlpc_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,mlpc_predicted))
# %%
model_ev = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest','Extreme Gradient Boost','K-Nearest Neighbour','Decision Tree','Support Vector Machine', 'Stochastic Gradient Descent', 'Neural Nets'], 'Accuracy': [round((lr_acc_score*100), 2),
                    round((nb_acc_score*100), 2),round((rf_acc_score*100), 2),round((xgb_acc_score*100), 2),round((knn_acc_score*100), 2),round((dt_acc_score*100), 2),round((svc_acc_score*100), 2), round((sgdc_acc_score*100),2), round((mlpc_acc_score*100), 2)]})
#%%
model_ev.sort_values(by='Accuracy', ascending=False)
