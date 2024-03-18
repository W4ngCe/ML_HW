import pandas as pd

# Import machine learning library
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('Machine Learning-001-dataset.csv')

# Attributes and labels
x = df[['Weather', 'Temperature', 'Wind Level', 'Number of workers', 'Day of the week']]
y = df['Approved']
# Data cleaning
x.loc[:, 'Weather'] = x['Weather'].map({'Sunny': 1.0, 'Cloudy': 0.5, 'Other': 0.0})

# pre-training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model1 = DecisionTreeClassifier()
# train the model
model1.fit(X_train, y_train)

# Evaluate model accuracy
accuracy = model1.score(X_test, y_test)
print(f'the accuracy is {accuracy}')

# Evaluate the accuracy on the model
train_predictions = model1.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
train_error_rate = 1 - train_accuracy
print(f"Training set error rate: {train_error_rate}")


def predict(weather, temperature, wind_level, num_of_wprlers, day_of_the_week):
    new_data = pd.DataFrame({'Weather': [weather],
                             'Temperature': [temperature],
                             'Wind Level': [wind_level],
                             'Number of workers': [num_of_wprlers],
                             'Day of the week': [day_of_the_week]})
    prediction = model1.predict(new_data)
    print("predicted result:", prediction)


data = pd.read_csv('data.csv')
X = data.drop('Approved', axis=1)
predictions = model1.predict(X)
print(predictions)


