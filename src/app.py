import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv',sep=";")


def replace_with_frequent(df,col):
    frequent = df[col].value_counts().idxmax()
    #print("The most frequent value is:", frequent)
    df[col].replace('unknown', frequent , inplace = True)
    #print("Replacing unknown values with the most frequent value:", frequent)

 
replace_with_frequent(data, "job")
replace_with_frequent(data, "marital")
replace_with_frequent(data, "education")
replace_with_frequent(data, "default")
replace_with_frequent(data, "housing")
replace_with_frequent(data, "loan")

#Converting Age into categorical data
age_groups = pd.cut(data['age'],bins=[10,20,30,40,50,60,70,80,90,100],
                    labels=['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-100'])
data.insert(1,'age_group',age_groups)
data.drop('age',axis=1,inplace=True)

#Grouping education categories 'basic.9y','basic.6y','basic4y' into 'middle_school'
lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    data.loc[data['education'] == i, 'education'] = "middle.school"

# Dropping pdays column
data.drop('pdays', axis=1, inplace= True)

#Removing duplicates
duplicated_data=data[data.duplicated(keep="last")]
data=data.drop_duplicates()

#Converting target variable into binary
def target_to_binary(y):
    y.replace({"yes":1,"no":0},inplace=True)

target_to_binary(data['y'])

#Encoding ordinal features
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

data['age_group'] = encoder.fit_transform(data['age_group'])
data['education'] = encoder.fit_transform(data['education'])

#Encoding categorical features that are not ordinal
data = pd.get_dummies(data, columns = ['job', 'marital', 'default','housing', 'loan', 'contact', 'poutcome'])

#Encoding month and day of the week
month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
data['month']= data['month'].map(month_dict) 

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
data['day_of_week']= data['day_of_week'].map(day_dict) 

X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print(metrics.accuracy_score(y_test, y_pred))
predictions = clf.predict(X_test)


# Confusion matrix
cm=confusion_matrix(y_test, predictions)
print(cm)

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions))

# plot confusion matrix to describe the performance of classifier.

class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
