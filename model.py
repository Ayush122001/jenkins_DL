import pandas as pd
ds  = pd.read_csv("cardio_train.csv",delimiter=';')
y = ds['cardio']
gender = pd.get_dummies(ds['gender'],drop_first=True)
s = ds[['age', 'height', 'weight', 'ap_hi', 'ap_lo','cholesterol', 'gluc', 'smoke', 'alco', 'active']]
X = pd.concat([s,gender],axis=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
model = Sequential()
model.add(Dense(units=70,activation='relu',input_dim=11))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=20,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=5,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.00001),metrics=['accuracy'])
model.fit(X,y,epochs=10)
model.save("final.h5")