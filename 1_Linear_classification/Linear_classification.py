# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from timeit import timeit
from ucimlrepo import fetch_ucirepo 
# %% [markdown]
#Stworzenie zestawu danych


# %%
r_state = 102452
#Steps to check avg time
Time_steps = 100 

n_samples = int(str(r_state)[0:2] + str(r_state)[-2:])

x_cluster, y_cluster = skl.datasets.make_classification(
                                        n_samples=n_samples,
                                        n_features=2,
                                        n_informative=2,
                                        n_redundant=0,
                                        n_clusters_per_class=1,
                                        random_state=r_state
                                    )

x_moon, y_moon = skl.datasets.make_moons(n_samples=n_samples,
                                         random_state=r_state)

# %% [markdown]
#Funkcje
# %%
#Wizualizacja
def Visualize_data(x, y, title):
    colors = ['green','red']
    labels = ['Class 0','Class 1']
    for class_value in [0,1]:
        idx = (y == class_value)
        plt.scatter(
          x[idx,0], x[idx,1],
          c=colors[class_value],
          label=labels[class_value],
          edgecolor='k',
          s=60
        )
    plt.legend()
    plt.title(title)
    
    
#regresja ->klasa
def Sigmoid_class(y):

    
    return y


#Skl klasyfikator
def SKL_Classifier(x_train, y_train, x_test):
    #Utworz klasyfikator SKL
    Skl_ridge = skl.linear_model.Ridge(alpha = 2)
    Skl_ridge.fit(x_train, y_train)
    #Regresja
    y_prob = Skl_ridge.predict(x_test)
    
    #Dodanie funkcji sigmoid aby okreslic klase
    #Mnozenie przez sigmoid aby otrzymac prawdopodobienstwo klasy
    #sigmoid = 1 / (1 + np.exp(-y))
    #y = y*sigmoid
    #Przydzielenie do klasy
    y_pred = (y_prob > 0.5).astype(int)
    
    return y_pred, y_prob

#Skl klasyfikator

def OWN_Classifier(x_train, y_train, x_test, L2 = 2):
    #Dodaj bias (jest domyslnie wlaczony w skl)
    poly = PolynomialFeatures(degree=1, include_bias=True)
    x_train = poly.fit_transform(x_train)
    x_test = poly.fit_transform(x_test)
    
    
    #Macierz jednostkowa tworzenie
    _,matrix_shape = x_train.shape
    I_matrix = np.eye(matrix_shape)

    weights = np.linalg.inv(x_train.T @ x_train + L2 * I_matrix) @ x_train.T @ y_train
    
    y_prob = x_test*weights
    y_prob = np.sum(y_prob ,axis = 1)
    
    #Mnozenie przez sigmoid aby otrzymac prawdopodobienstwo klasy
    #sigmoid = 1 / (1 + np.exp(-y))
    #y = y*sigmoid
    #Przydzielenie do klasy
    y_pred = (y_prob > 0.5).astype(int)
    
    return y_pred, y_prob

def Plot_meshgrid(x_train, y_train, classifier):
    #Krok
    h = 0.02  
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    #Predykcja klasy dla punktów siatki
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    c,_ = classifier(x_train, y_train, grid_points)
    c = c.reshape(xx.shape)
    
    plt.contourf(xx, yy, c, alpha=0.3, cmap="coolwarm")
    
def Calculate_metrics(y_true,y_pred, y_prob, show_plots = True, title = "Metrics"):
    
    tn, fp, fn, tp = skl.metrics.confusion_matrix(y_true, y_pred).ravel()
    
    acc = skl.metrics.accuracy_score(y_true, y_pred)
    recall = skl.metrics.recall_score(y_true, y_pred)
    specificity = float(tn / (tn + fp))
    auc = float(skl.metrics.roc_auc_score(y_true, y_prob))
    
    metrics = {}
    metrics["Conf_matrix"] = {"True_positive":tp,"False_positive":fp,"False_negative":fn,"True_negative":tn}
    metrics["Accuracy"] = acc
    metrics["Recall"] = recall
    metrics["Specificity"] = specificity
    metrics["AUC_ROC"] = auc
    
    if show_plots:
        #Conf matrix
        cm = skl.metrics.confusion_matrix(y_true, y_pred)
    
        plt.figure()
        plt.suptitle(title)
        plt.subplot(1,2,1)
        plt.imshow(cm, cmap='Greens')
        
        #Tytul
        plt.title("Confusion matrix")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        #nazwy klas
        plt.xticks([0, 1], ['Class 0', 'Class 1'])
        plt.yticks([0, 1], ['Class 0', 'Class 1'])
        
        #Tekst
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        #########################################################3
        #ROC curve
        plt.subplot(1,2,2)
        plt.title("ROC Curve")
        fpr, tpr, _ = skl.metrics.roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='green', label=f'ROC Curve (AUC = {auc:.2f})')
        
        #Labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        #Visuals
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return metrics
    
# %% [markdown]
#Wlasciwy program (Ocena 3)

# %%
#Przygotowanie funkcji do wywolania na cluster i moon tak samo
def Give_me_thr33_pls(x,y,r_state):
    #Podzial danych na treningowy i testowy ze stratyfikacja
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #Skalowanie danych
    scaler = skl.preprocessing.StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    

    #Wywolanie funkcji SKL (trening modelu + predykcja)
    y_pred_SKL,y_prob_SKL = SKL_Classifier(x_train_scaled, y_train, x_test_scaled)
    y_pred_OWN,y_prob_OWN = OWN_Classifier(x_train_scaled, y_train, x_test_scaled)
    
    #Wykresy
    ################################
    plt.figure()
    #Train data
    plt.suptitle("SKL vs OWN model [Both with BIAS]")
    plt.subplot(2,2,1)
    Visualize_data(x_train_scaled, y_train, title = "Train Data ; SKL")
    Plot_meshgrid(x_train_scaled, y_train, SKL_Classifier)
    
    #Test data
    plt.subplot(2,2,2)
    Visualize_data(x_test_scaled, y_test, title = "Test Data ; SKL")
    Plot_meshgrid(x_train_scaled, y_train, SKL_Classifier)
    
    ###############################
    plt.subplot(2,2,3)
    Visualize_data(x_train_scaled, y_train, title = "Train Data ; OWN")
    Plot_meshgrid(x_train_scaled, y_train, OWN_Classifier)
    
    plt.subplot(2,2,4)
    Visualize_data(x_test_scaled, y_test, title = "Test Data ; OWN")
    Plot_meshgrid(x_train_scaled, y_train, OWN_Classifier)
    

    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_SKL = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_SKL,
                                         y_prob = y_prob_SKL,
                                         title = "Metrics SKL"
                                         )
    
    Metrics_dict_OWN = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_OWN,
                                         y_prob = y_prob_OWN,
                                         title = "Metrics OWN"
                                         )
    for dictionary, header in zip([Metrics_dict_SKL, Metrics_dict_OWN] , ["SKL Metrics", "OWN Metrics"]):
        print("\n")
        print(header)
        print("Accuracy: ",dictionary.get("Accuracy"))
        print("Recall: ",dictionary.get("Recall"))
        print("Specificity: ",dictionary.get("Specificity"))
        print("AUC_ROC: ",dictionary.get("AUC_ROC"))
    

    #Test czasu wykonania
    
    print("\n\n")
    #Wywolanie funkcji SKL (trening modelu + predykcja)
    y_pred_SKL,y_prob_SKL = SKL_Classifier(x_train_scaled, y_train, x_test_scaled)
    y_pred_OWN,y_prob_OWN = OWN_Classifier(x_train_scaled, y_train, x_test_scaled)
    
    Total_time_SKL = timeit(lambda: SKL_Classifier(x_train_scaled, y_train, x_test_scaled), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in SKL method: {Total_time_SKL:.4f} s")
    
    Total_time_OWN= timeit(lambda: OWN_Classifier(x_train_scaled, y_train, x_test_scaled), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")



# %% [markdown]
#Ostateczne wywolanie na danych
# %% 
print("CLUSTER Data:")
Give_me_thr33_pls(x = x_cluster,
                  y = y_cluster,
                  r_state = r_state
                  )
print("-----------------------------------------------------------------------")
print("MOON Data:")
Give_me_thr33_pls(x = x_moon,
                  y = y_moon,
                  r_state = r_state
                  )



# %% [markdown]
#Na 4

# %% [markdown]
#Zaladowanie zestawu danych
# %%
#############################
heart_disease = fetch_ucirepo(id=45) 
  
#Ladowanie danych (pandas df)
x = heart_disease.data.features 
y = heart_disease.data.targets 
###############################

# %% [markdown]
#1
#Policz ilość brakujących wartości a następnie usuń wiersze je zawierajace
# %%
#Wyrzuc wiersz jesli zawiera NAn lub None
x_cleaned = x.dropna()
#Uzywamy jeszcze nie wyczyszczonych indexow do wziecia odpowiadajacych Y
y_cleaned = y.loc[x_cleaned.index]

print("Liczba brakujacych wartosci: ", len(y) - len(y_cleaned))
# %% [markdown]      
#2
# Zastąp zmienną przewidywaną wartością binarną opisującą występowanie choroby serca

# %%
y_cleaned = [1 if y>0 else 0 for y in y_cleaned["num"]]

# %% [markdown]
#3
#Dla wszystkich cech policz podstawowe miary statystyczne (średnia/dominanta, odchylenie
#standardowe, minimum, maksimum) uwzględniając typ cechy (kategoryczna/dyskretna/ciągła).
# %%
features = x_cleaned.columns.tolist()
print(features)
categorical = [False,True,True,False,False,True,True,False,True,False,True,True,True]

Feature_dict = {}

for f,categorical in zip(features, categorical):
    feature = x_cleaned[f].tolist()
    
    if categorical:
        #Dominanta
        vals, counts = np.unique(feature, return_counts=True)
        dominant = float(vals[np.argmax(counts)])
        
        minimum = float(np.min(feature))
        maximum = float(np.max(feature))
        
        #Create stats
        stats = {"Dominant": dominant, "Min": minimum, "Max": maximum}
  
    else:
        mean = float(np.mean(feature))
        std = float(np.std(feature))
        minimum = float(np.min(feature))
        maximum = float(np.max(feature))
        
        #Create stats
        stats = {"Mean": mean,"Std":std, "Min": minimum, "Max": maximum}
        
    #Add stats to dict    
    Feature_dict[f] = stats

print("Features stats are succesfully stored in the dictionary: 'Feature_dict' ")

# %% [markdown]
#4
#Przedstaw wartości cech w formie histogramów pokolorowanych zależnie od wartości przewidywanej
# %%
plt.figure(figsize = (16,6))
plt.suptitle("Features comparision")
for i,col in enumerate(features):
    
    healthy_vector = [True if y==0 else False for y in y_cleaned]
    sick_vector = [True if y==1 else False for y in y_cleaned]
    plt.subplot(2,7,i+1)
    plt.hist(x_cleaned[col][healthy_vector], bins=20, alpha=0.5, label='healthy', color='green')
    plt.hist(x_cleaned[col][sick_vector], bins=20, alpha=0.5, label='disease', color='red')
    
    plt.legend()
    plt.title(col)

# %% [markdown]
#5
#Narysuj macierz korelacji pomiędzy wartościami.

# %%
#Macierz korelacji z pandas
corr = x_cleaned.corr()
#Wykres z 
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr)
fig.colorbar(cax)

#Zrobienie zeby labele pojawialy sie nie co 2, co 3 tylko kazdy po kolei
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
#Labele
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
plt.title("Correlation Matrix")


# %% [markdown]
#Odpowiedz:
#Kierujac sie wyborem cech powinnismy wziac pod uwagę:
#Czy roznia sie one swoim rozkladem w zaleznosci od etykiety, labela.
#Czy nie sa ze sobą mocno skorelowane - mocno skorelowane cechy czesto przekazuja ta sama informacje
#Analizujac macierz korelacji oraz rozklady histogramow, wybrane przeze mnie cechy to:
#1 - Age
#2 - Thalach
#3 - cp
#4 - thal


# %% [markdown]
#Na 5

# %%
#Przygotowanie funkcji do wywolania na cluster i moon tak samo
def f1fe_is_good(x,y,r_state):
    #Podzial danych na treningowy i testowy ze stratyfikacja
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #Skalowanie danych
    scaler = skl.preprocessing.StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    #Wywolanie funkcji SKL (trening modelu + predykcja)
    y_pred_SKL,y_prob_SKL = SKL_Classifier(x_train_scaled, y_train, x_test_scaled)
    y_pred_OWN,y_prob_OWN = OWN_Classifier(x_train_scaled, y_train, x_test_scaled)
    
    Metrics_SKL = Calculate_metrics(y_test, y_pred_SKL, y_prob_SKL, show_plots = True)
    Metrics_OWN = Calculate_metrics(y_test, y_pred_OWN, y_prob_OWN, show_plots = True)

    #Comparision and time
    for dictionary, header in zip([Metrics_SKL, Metrics_OWN] , ["SKL Metrics", "OWN Metrics"]):
        print("\n")
        print(header)
        print("Accuracy: ",dictionary.get("Accuracy"))
        print("Recall: ",dictionary.get("Recall"))
        print("Specificity: ",dictionary.get("Specificity"))
        print("AUC_ROC: ",dictionary.get("AUC_ROC"))
    

    #Test czasu wykonania
    print("\n\n")
    #Wywolanie funkcji SKL (trening modelu + predykcja)
    y_pred_SKL,y_prob_SKL = SKL_Classifier(x_train_scaled, y_train, x_test_scaled)
    y_pred_OWN,y_prob_OWN = OWN_Classifier(x_train_scaled, y_train, x_test_scaled)
    
    Total_time_SKL = timeit(lambda: SKL_Classifier(x_train_scaled, y_train, x_test_scaled), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in SKL method for 13 features: {Total_time_SKL:.4f} s")
    
    Total_time_OWN= timeit(lambda: OWN_Classifier(x_train_scaled, y_train, x_test_scaled), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in OWN method for 13 features: {Total_time_OWN:.4f} s")

    
    return Metrics_SKL, Metrics_OWN


# %% [markdown]
#1
#Porównaj skuteczność własnej implementacji i RidgeClassifier na zbiorze danych o chorobach
#serca wykorzystując 13 głównych cech

# %%
print("\n\n13 features Data:")
#Transform into the array
x_13 = np.array(x_cleaned)
Metrics_SKL, Metrics_OWN = f1fe_is_good(x = x_13,
                                        y = y_cleaned,
                                        r_state = r_state
                                        )
# %% [markdown]
#2
#Następnie w oparciu o RidgeClassifier z sklearn i sklearn.feature_selection.SequentialFeatureSelector wybierz 4 najlepsze cechy.
# %% 

#Wybranie najlepszych cech w oparciu o knn
knn = skl.neighbors.KNeighborsClassifier(n_neighbors=5)
sfs = skl.feature_selection.SequentialFeatureSelector(knn, n_features_to_select=4)
#Fitting model
sfs.fit(x_13, y_cleaned)
#Sprawdzenie i wybranie najlepszych 4 cech
best_features = sfs.get_support()
columns = x_cleaned.columns
best_features = columns[best_features].tolist()
print("Najlepsze 4 wedlug algorytmu: ",best_features)
print("Najlepsze 4 cechy wybrane przeze mnie: Age, Thalach, cp, thal ")
#Ekstrakcja danych 4 najlepszych cech

# %% [markdown]
#3 
#Porównaj skuteczność własnej implementacji i RidgeClassifier na obu wybranych zestawach cech.
# %%

own_features = ['age', 'thalach', 'cp','thal']
x_4_own = np.array(x_cleaned[own_features])
print("\n\n4 Own features Data:")
_,_ = f1fe_is_good(x = x_4_own,
                   y = y_cleaned,
                   r_state = r_state
                   )

x_4_alg = sfs.transform(x_13)
print("\n\n4 Algorithm features Data:")
_,_ = f1fe_is_good(x = x_4_alg,
                   y = y_cleaned,
                   r_state = r_state
                   )
