import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

url = 'https://raw.githubusercontent.com/gordito100/ArchivosCSVV/main/BankNote_Authentication.csv'
df = pd.read_csv(url)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

print(classification_report(y_test, y_pred))

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)
