import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import matplotlib.pyplot as plt
from funkcije import graficki_prikaz_istorije, kreiraj_model, kreiraj_model1, \
    selectKBest, upsample_belo, downsample_belo

np.random.seed(3)

# broj klasa
broj_klasa = 10

# load dataset
#dataset = np.loadtxt('wine1.txt', delimiter=",")
# split dataset into sets for testing and training
#X = dataset[:,0:11]
#Y = dataset[:,11]

#učitavanje skupa podatka
data = pd.read_csv('winequality-white1.csv')
#veličitina prozora za prikaz grafikona
plt.rcParams["figure.figsize"] = (8,8)

#korelaciona matrica
corrMatrix = data.corr()
sn.heatmap(corrMatrix, annot=True)
#prikaz heatmap-e
plt.show()

#Statistički podatak
plt.hist(data.alcohol, 10, facecolor='red', alpha=0.7)
#konfiguracija grafika na kome ce biti prikazani podaci
plt.title("Distribution of Alcohol in % Vol")
plt.xlabel("Alcohol in % Vol")
plt.ylabel("Frequency")
plt.show()

#Provera da li postoje null vrednosti
a=pd.isnull(data)
a=a.to_numpy()
if True in a:
    print("Postoje null vrednosti u okviru skupa podataka")
else:
    print("Ne postoje null vrednosti u okviru skupa podataka")

#odabir labela
X = data.iloc[:,0:11]
Y=data['quality']

#provera koliko stavki u okviru skupa podataka pripada nekoj klasi
a = data['quality'].value_counts()
print(a)
a1 = a.values
a2=data['quality'].unique()

#graficki prikaz klasne raspodele
plt.bar(a2, a1, color=(0, 0, 1, 0.7))
plt.show()

#podela u test i train
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.66, random_state=5)

# potrebno je izvršiti konverziju u kategoričke
y_train = keras.utils.to_categorical(y_train-1, broj_klasa)
y_test = keras.utils.to_categorical(y_test-1, broj_klasa)

# kreiranje prvog modela
model = kreiraj_model(11,broj_klasa)
# kompajliranje i fitovanje
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)
#statistike vezane za model
print(model.summary())

#graficki prikaz istorije
graficki_prikaz_istorije(history)
# kreiranje drugog modela
model1 = kreiraj_model1(11,broj_klasa)

# kompajliranje i fitovanje
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model1.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)
#statistike vezane za model
print(model1.summary())
#graficki prikaz istorije
graficki_prikaz_istorije(history)

#upsample balansiranje
df_upsampled = upsample_belo(data)
print(df_upsampled['quality'].value_counts())

#odabir labela
X = df_upsampled.iloc[:,0:11]
Y=df_upsampled['quality']

#podela podataka na trening i test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.66, random_state=5)

# potrebno je izvršiti konverziju u kategoričke
y_train = keras.utils.to_categorical(y_train-1, broj_klasa)
y_test = keras.utils.to_categorical(y_test-1, broj_klasa)

# kompajliranje i fitovanje
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)


#graficki prikaz istorije
graficki_prikaz_istorije(history)

# kompajliranje i fitovanje
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model1.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)

#graficki prikaz istorije
graficki_prikaz_istorije(history)

#downsample balansiranje, kako bi postoja podjednak broj stavki za sve klase
df_downsampled = downsample_belo(data)
print(df_downsampled['quality'].value_counts())

#odabir labela
X = df_downsampled.iloc[:,0:11]
Y=df_downsampled['quality']

#podela podataka na trening i test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.66, random_state=5)

# potrebno je izvršiti konverziju u kategoričke
y_train = keras.utils.to_categorical(y_train-1, broj_klasa)
y_test = keras.utils.to_categorical(y_test-1, broj_klasa)

# kompajliranje i fitovanje
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)

#graficki prikaz istorije
graficki_prikaz_istorije(history)

# kompajliranje i fitovanje
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model1.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)

#graficki prikaz istorije
graficki_prikaz_istorije(history)

#selectKBest
data1=data
data_new = selectKBest(data1)

X1 = data_new.iloc[:,0:3]
Y=data_new['quality']
x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.66, random_state=5)

# potrebno je izvršiti konverziju u kategoričke
y_train = keras.utils.to_categorical(y_train-1, broj_klasa)
y_test = keras.utils.to_categorical(y_test-1, broj_klasa)

# kreiranje  modela
model = kreiraj_model(3,broj_klasa)
model1 = kreiraj_model(3,broj_klasa)


# kompajliranje i fitovanje
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)
#statistike vezane za model
print(model.summary())
#graficki prikaz istorije
graficki_prikaz_istorije(history)

# kompajliranje i fitovanje
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history=model1.fit(x_train, y_train, batch_size=15, epochs=50, validation_data=(x_test, y_test),verbose = 1)

#statistike vezane za model
print(model1.summary())
#graficki prikaz istorije
graficki_prikaz_istorije(history)



