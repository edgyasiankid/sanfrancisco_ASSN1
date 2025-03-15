import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#load dataset
df = pd.read_csv('./csv/crimeTime.csv')

#write the dataframe to a gzip-compressed CSV file
df.to_csv('csv/crimeTime.csv.gz', index=False, compression='gzip')

#one-hot encode 'Category' (crime type) for classification
encoder = OneHotEncoder(sparse_output=False)
category_encoded = encoder.fit_transform(df[['Category']])
category_labels = encoder.categories_[0]

#create dataframe for encoded labels
category_df = pd.DataFrame(category_encoded, columns=category_labels)

#one-hot encode 'ordinalDistrict'
district_encoder = OneHotEncoder(sparse_output=False)
district_encoded = district_encoder.fit_transform(df[['ordinalDistrict']])
district_labels = district_encoder.categories_[0]

#create dataframe for encoded districts
district_df = pd.DataFrame(district_encoded, columns=[f'District_{int(d)}' for d in district_labels])

#merge with main dataframe
df = pd.concat([df[['ordinalDOW', 'Time_Minutes']], district_df, category_df], axis=1)

#normalize 'Time_Minutes'
df['Time_Minutes'] = df['Time_Minutes'] / 1440 #1440 minutes = 1 day

#plit dataset into features (X) and target (y)
X = df.drop(category_labels, axis=1)
y = category_encoded

#train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#define MLP architecture for classification
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

#compile the model with a smaller learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, 
          epochs=100,
          batch_size=32, 
          validation_split=0.2,
          verbose=1)

#evaluate the model on the test data
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

#calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print("Model Accuracy:", accuracy)
