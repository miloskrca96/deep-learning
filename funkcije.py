import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
import pandas as pd

#funkcija koja služi da na grafiku prikaže istoriju kretanja vrednosti accuracy i loss
def graficki_prikaz_istorije(h):
    # istorija za loss
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epohe')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # istorija za loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#funkcija koja izdvaja k najboljih atributa iz skupa atributa samog dataset-a
#koristi se sa ciljem da se rezultat poboljša
def selectKBest(data):
    data1 = data.drop('quality', axis=1)
    #izdvajanje kolona
    cols = data1.columns
    Y = data['quality']
    #poziv funkcije pri čemu je parametar broj atributa koji se izdvaja iz početnog skupa
    # i score funkcija
    test = SelectKBest(score_func=f_classif, k=3)
    fit = test.fit(data1, Y)
    broj = fit.get_support(indices=True)

    #kreiranje novog skupa podataka sa onim kolonama čiji su atributi izdvojeni
    col = []
    for i in range(len(cols)):
        if i not in broj:
            col.append(cols[i])
    #kreira se tako što se ostale kolone dropuju
    data2 = data1.drop(columns=col, axis=1)
    #pridruživanje klasnog atributa
    data2 = data2.join(Y)
    return data2

#funkcija za upsample balansiranje skupa podataka, pri čemu se prave posebne funkcije za
#skup koji se odnosi na crveno i posebne za skup koji se odnosi na belo
def upsample_crveno(data):
    #najviše je stavki sa klasnom pripadnošću 5
    df_majority = data[data['quality'] == 5]
    #sve ostale klasne pripadnosti su manje
    df_minority = data[data['quality'] == 3]
    df_minority1 = data[data['quality'] == 4]
    df_minority2 = data[data['quality'] == 8]
    df_minority3 = data[data['quality'] == 6]
    df_minority4 = data[data['quality'] == 7]

    #resample u grupama po dva, u ovom slučaju 5 i 3
    df_minority_upsampled1 = resample(df_minority,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 5 i 4
    df_minority_upsampled2 = resample(df_minority1,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 5 i 8
    df_minority_upsampled3 = resample(df_minority2,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 5 i 6
    df_minority_upsampled4 = resample(df_minority3,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 5 i 7
    df_minority_upsampled5 = resample(df_minority4,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)

    # Kombinovanje u novi skup podataka
    df_upsampled = pd.concat([df_majority, df_minority_upsampled1, df_minority_upsampled2, df_minority_upsampled3, df_minority_upsampled4, df_minority_upsampled5])
    return  df_upsampled

#funkcija za downsample balansiranje skupa podataka, pri čemu se prave posebne funkcije za
#skup koji se odnosi na crveno i posebne za skup koji se odnosi na belo
def downsample_crveno(data):
    #slična ideja kao kod prethodne samo se sad gleda klasa sa najmanje
    df_minority = data[data['quality'] == 3]
    # sve ostale klasne pripadnosti su veće
    df_majority = data[data['quality'] == 5]
    df_majority1 = data[data['quality'] == 4]
    df_majority2 = data[data['quality'] == 8]
    df_majority3 = data[data['quality'] == 6]
    df_majority4 = data[data['quality'] == 7]

    # resample u grupama po dva, u ovom slučaju 3 i 5
    df_majority_downsampled1 = resample(df_majority,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 3 i 4
    df_majority_downsampled2 = resample(df_majority1,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 3 i 8
    df_majority_downsampled3 = resample(df_majority2,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 3 i 6
    df_majority_downsampled4 = resample(df_majority3,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 3 i 7
    df_majority_downsampled5 = resample(df_majority4,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # Kombinovanje u novi skup podataka
    df_downsampled = pd.concat(
        [df_minority, df_majority_downsampled1, df_majority_downsampled2, df_majority_downsampled3, df_majority_downsampled4,
         df_majority_downsampled5])
    return df_downsampled

#za sledeće dve funkcije je priča ista, sa razlikom da u okviru ovog skupa podataka postoji drugačije klasna pripadnost
#te se piše posebna funkcija
def upsample_belo(data):
    # najviše je stavki sa klasnom pripadnošću 6
    df_majority = data[data['quality'] == 6]
    # sve ostale klasne pripadnosti su manje
    df_minority = data[data['quality'] == 5]
    df_minority1 = data[data['quality'] == 7]
    df_minority2 = data[data['quality'] == 8]
    df_minority3 = data[data['quality'] == 4]
    df_minority4 = data[data['quality'] == 3]
    df_minority5 = data[data['quality'] == 9]

    # resample u grupama po dva, u ovom slučaju 6 i 5
    df_minority_upsampled1 = resample(df_minority,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 6 i 7
    df_minority_upsampled2 = resample(df_minority1,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 6 i 8
    df_minority_upsampled3 = resample(df_minority2,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 6 i 4
    df_minority_upsampled4 = resample(df_minority3,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 6 i 3
    df_minority_upsampled5 = resample(df_minority4,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 6 i 9
    df_minority_upsampled6 = resample(df_minority5,
                                      replace=True,
                                      n_samples=df_majority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najviše zastupljena
                                      random_state=0)

    # Kombinovanje u novi skup podataka
    df_upsampled = pd.concat([df_majority, df_minority_upsampled1, df_minority_upsampled2, df_minority_upsampled3, df_minority_upsampled4, df_minority_upsampled5,df_minority_upsampled6])
    return  df_upsampled

def downsample_belo(data):
    # slična ideja kao kod prethodne samo se sad gleda klasa sa najmanje
    df_minority = data[data['quality'] == 9]
    # sve ostale klasne pripadnosti su veće
    df_majority = data[data['quality'] == 5]
    df_majority1 = data[data['quality'] == 7]
    df_majority2 = data[data['quality'] == 8]
    df_majority3 = data[data['quality'] == 4]
    df_majority4 = data[data['quality'] == 3]
    df_majority5 = data[data['quality'] == 6]

    # resample u grupama po dva, u ovom slučaju 9 i 5
    df_majority_downsampled1 = resample(df_majority,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 9 i 7
    df_majority_downsampled2 = resample(df_majority1,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 9 i 8
    df_majority_downsampled3 = resample(df_majority2,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 9 i 4
    df_majority_downsampled4 = resample(df_majority3,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 9 i 3
    df_majority_downsampled5 = resample(df_majority4,
                                      replace=True,
                                      n_samples=df_minority['quality'].count(),  # kako bi broj bio isti kao kod klase koja je najmanje zastupljena
                                      random_state=0)
    # resample u grupama po dva, u ovom slučaju 9 i 6
    df_majority_downsampled6 = resample(df_majority5,
                                      replace=True,  # sample with replacement
                                      n_samples=df_minority['quality'].count(),  # to match majority class
                                      random_state=0)  # reproducible results
    # Kombinovanje u novi skup podataka
    df_upsampled = pd.concat(
        [df_minority, df_majority_downsampled1, df_majority_downsampled2, df_majority_downsampled3, df_majority_downsampled4,
         df_majority_downsampled5,df_majority_downsampled6])
    return df_upsampled

#funkcija za kreiranje jednog od tipova mreže koji je korišćen u testiranju
def kreiraj_model(br,br1):
    # kreiranje
    model = Sequential()
    # dodavanje nivoa u okviru modela
    model.add(Dense(10, input_dim=br, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(br1, activation='softmax'))
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

#funkcija za kreiranje jednog od tipova mreže koji je korišćen u testiranju
def kreiraj_model1(br,br1):
    #kreiranje
    model = Sequential()
    # dodavanje nivoa u okviru modela
    model.add(Dense(12, activation='relu', input_shape=(br,)))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(br1, activation='softmax'))
    plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
    return model