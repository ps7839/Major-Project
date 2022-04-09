import pandas as pd 
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("blood_train.csv")


X = data.iloc[ : ,1:5]
y = data.iloc[ : ,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.3) 

#from sklearn.preprocessing import StandardScaler
#Scaler = StandardScaler()
#X_train = Scaler.fit_transform(X_train)
#X_test = Scaler.fit_transform(X_test)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train.values,y_train.values)
y_pred = rf.predict(X_test.values) 

print(y_pred)

pickle.dump(rf , open('finalmodel.pkl' , 'wb'))
model = pickle.load(open('finalmodel.pkl' , 'rb'))