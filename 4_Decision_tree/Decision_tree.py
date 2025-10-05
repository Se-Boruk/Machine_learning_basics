# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from timeit import timeit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
# %%
r_state = 102452
#Steps to check avg time
Time_steps = 100

n_samples = int(str(r_state)[0:2] + str(r_state)[-2:])
#Singlemdode data
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
#Multimode data
x_multimode, y_multimode = skl.datasets.make_classification(n_samples=n_samples,
                                                            n_features=5,
                                                            n_informative=3,
                                                            n_redundant=2,
                                                            random_state = r_state
                                                            )


# %% [markdown]
#Functions
# %%
def Plot_meshgrid(x_train, y_train, classifier):
    #Krok
    h = 0.02  
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    #Predykcja klasy dla punktów siatki
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    c = classifier.predict(grid_points)
    c = c.reshape(xx.shape)
    
    plt.contourf(xx, yy, c, alpha=0.3, cmap="coolwarm")

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

def Calculate_metrics(y_true,y_pred, y_prob, show_plots = True, title = "Metrics"):
    
    tn, fp, fn, tp = skl.metrics.confusion_matrix(y_true, y_pred).ravel()
    #If lenght of probs is 2 (so it has probabilities for 2 classes, take only for class 1)
    try:
        y_prob_1 = y_prob[:,1]
    except:
        y_prob_1 = y_prob
    
    acc = skl.metrics.accuracy_score(y_true, y_pred)
    recall = skl.metrics.recall_score(y_true, y_pred)
    specificity = float(tn / (tn + fp))
    auc = float(skl.metrics.roc_auc_score(y_true, y_prob_1))
    
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
        fpr, tpr, _ = skl.metrics.roc_curve(y_true, y_prob_1)
        plt.plot(fpr, tpr, color='green', label=f'ROC Curve (AUC = {auc:.2f})')
        
        #Labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        #Visuals
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return metrics



#Skl klasyfikator
def SKL_Classifier(x_train, y_train, x_test, depth):
    
    model = DecisionTreeClassifier(max_depth = depth,
                                   criterion='entropy',
                                   random_state = r_state
                                   )
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    return model, y_pred, y_prob

#Skl klasyfikator
def OWN_Classifier(x_train, y_train, x_test, depth):
    
    model = DecisionTree(max_depth = depth,
                         min_samples_split = 2
                         )
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    return model, y_pred, y_prob


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        if max_depth is None:
            self.max_depth = 999
        else:
            self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        #Poki co brak korzenia
        self.root = None 
        #To bedzie zdefiniowane przez unique y potem
        self.n_classes_ = None

    #Fit, zbudowanie drzewa, trening
    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y, dtype=int)
        #Ilosc klas
        self.n_classes_ = len(np.unique(y))
        #Ustawiamy korzeń (od jakiego 'drzewa' - bo jest to zbior drzew polaczanych tak naprawdę, zacząć)
        self.root = self.build_tree(X, y, depth=0)
        
    #Entropia zbioru
    def entropy(self, y):
        #Policz liczbe wystapien klasy
        counts = np.bincount(y, minlength=self.n_classes_)
        #'Prawdopodobienstwo' globalne wystapienia count
        probs = counts / counts.sum()
        #Tylko prob dla prob >0 aby nie bylo log(0)
        mask = probs > 0
        #Wzor
        return -np.sum(probs[mask] * np.log2(probs[mask]))

    def best_split(self, X, y):
        #Sprawdzamy dla kazdej unikalnej wartosci cechy w probkach treningowych wartosc entropii po podziale
        #A następnie powtarzamy dla kazdej cechy i sprawdzamy najlepsza wartosc

        best_gain = 0
        best_feat, best_thresh = None, None
        parent_entropy = self.entropy(y)
        #Ile probek i ile cech - tyle bedzie kombinacji
        n_samples, n_feats = X.shape
    
        #1) Dla każdej cechy „feat” 
        for feat in range(n_feats):
    
            #2) Weź wszystkie unikalne wartości w tej kolumnie jako kandydatów na próg
            for thresh in np.unique(X[:, feat]):
    
                #3) Utwórz maskę, która dzieli zbiór na lewo (<= thresh) i prawo (> thresh)
                left_mask = X[:, feat] <= thresh
    
                #4) Pomijamy podziały, które dają pustą gałąź
                # (wszystkie dane poszły tylko w lewo lub tylko w prawo)
                if left_mask.sum() < 1 or left_mask.sum() == n_samples:
                    continue
    
                #5) Podziel etykiety y na y_left i y_right
                y_left  = y[left_mask]      # te probki ktore spelnily warunek
                y_right = y[~left_mask]     # te probki ktore nie spelnily warunku
    
                #6) Oblicz wagi dla obu podzbiorów (proporcje do całego zbioru)
                w_left  = len(y_left)  / n_samples
                w_right = 1 - w_left   # czyli len(y_right) / n_samples
    
                #7) Oblicz information gain:
                gain = parent_entropy \
                       - ( w_left  * self.entropy(y_left) 
                           + w_right * self.entropy(y_right) )
    
                #8) Jeśli gain jest lepszy niż cokolwiek do tej pory,
                #    zapamiętaj ten feat i thresh jako najlepszy
                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat
                    best_thresh = thresh
    
    
        #9) Po sprawdzeniu wszystkich cech i progów zwróć najlepszy podział
        return best_feat, best_thresh
    
    #Tylko rzecz do polaczen drzew, zawiera informacje o podziale
    class Node:
        def __init__(self, feat=None, thresh=None, left=None, right=None, counts=None):
            # jeśli counts != None, to jest liść; counts to tablica liczebności klas
            self.feat = feat
            self.thresh = thresh
            self.left = left
            self.right = right
            self.counts = counts

        def is_leaf(self):
            return self.counts is not None

    #Rekurencyjnie zbuduj drzewo, 
    #jesli probki klasy sa takie same lub tylko jedna klasa, koniec galezi
    # Jesli nie to znajdz best split, podziel i wroc do "poczatku" - powtorz
    
    #Tak naprawdę budujemy N drzew decyzyjnych i za kazdym razem nastepne rozgalezienie
    #jest kolejnym drzewem, ktore na koncu ma albo lisc (koniec), albo kolejne drzewo
    def build_tree(self, X, y, depth):
        
        #Stop jesli len(probka) == 1, przekroczono glebokosc, 
        #lub osiagnieto koniec - za malo probek do podzialu domyslnie 1
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            counts = np.bincount(y, minlength=self.n_classes_)
            return DecisionTree.Node(counts=counts)

        #Najlepszy podzial
        feat, thresh = self.best_split(X, y)
        if feat is None:
            counts = np.bincount(y, minlength=self.n_classes_)
            return DecisionTree.Node(counts=counts)

        #Podzial danychi jesli brak konca do tej pory --> czyli mozna jesszcze tworzyc warunki/drzewa
        #Stworz nowe drzewo
        left_mask = X[:, feat] <= thresh
        left_node = self.build_tree(X[left_mask],  y[left_mask],  depth+1)
        right_node = self.build_tree(X[~left_mask], y[~left_mask], depth+1)
        return DecisionTree.Node(feat=feat, thresh=thresh, left=left_node, right=right_node)

    def predict_one(self, x, node):

        while not node.is_leaf():
            if x[node.feat] <= node.thresh:
                node = node.left
            else:
                node = node.right
        return node.counts.argmax()

    #Zwroc klasy, uzywa pojedynczej predict_one w petli
    def predict(self, X):
        X = np.array(X)
        return np.array([self.predict_one(x, self.root) for x in X])


    #Prawdopodobienstwo (przez liczebnosc klas w zbiorze treningowym w tym 'lisciu')
    def predict_proba(self, X):

        X = np.array(X)
        probs = []
        for x in X:
            node = self.root
            while not node.is_leaf():
                node = node.left if x[node.feat] <= node.thresh else node.right
            total = node.counts.sum()
            if total > 0:
                probs.append(node.counts / total)
            else:
                #gdyby counts były wszystkie zero
                probs.append(np.zeros(self.n_classes_))
        return np.vstack(probs)
    
def SKL_vs_OWN(x,y, max_depth):
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #predicting by SKL
    skl_model, y_pred_SKL, y_prob_SKL = SKL_Classifier(x_train, y_train, x_test, depth = max_depth)
    own_model, y_pred_OWN, y_prob_OWN = OWN_Classifier(x_train, y_train, x_test, depth = max_depth)

    #Plot only 2 dim data, more or less throw away
    if x_train.shape[1] == 2:
        #Wykresy
        plt.figure()
        #Subplot with decision boundary
        plt.suptitle("SKL vs OWN model")
        
        #SKL section
        plt.subplot(2,2,1)
        Visualize_data(x_train, y_train, "Train Data ; SKL")
        Plot_meshgrid(x_train, y_train, classifier = skl_model)
        
        plt.subplot(2,2,2)
        Visualize_data(x_test, y_test, "Test Data ; SKL")
        Plot_meshgrid(x_train, y_train, classifier = skl_model)
        
        #OWN section
        plt.subplot(2,2,3)
        Visualize_data(x_train, y_train, "Train Data ; OWN")
        Plot_meshgrid(x_train, y_train, classifier = own_model)
        
        plt.subplot(2,2,4)
        Visualize_data(x_test, y_test, "Test Data ; OWN")
        Plot_meshgrid(x_train, y_train, classifier = own_model)
    

    ####################################################
    print("SKL metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_SKL = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_SKL,
                                         y_prob = y_prob_SKL,
                                         title = "Metrics SKL"
                                         )
    print("OWN metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_OWN = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_OWN,
                                         y_prob = y_prob_OWN,
                                         title = "Metrics OWN"
                                         )
    
    #Printing metrics
    for dictionary, header in zip([Metrics_dict_SKL, Metrics_dict_OWN] , ["SKL Metrics", "OWN Metrics"]):
        print("\n")
        print(header)
        print("Accuracy: ",round(dictionary.get("Accuracy"),3))
        print("Recall: ",round(dictionary.get("Recall"),3))
        print("Specificity: ",round(dictionary.get("Specificity"),3))
        print("AUC_ROC: ",round(dictionary.get("AUC_ROC"),3))
        
    #Time of each method
    Total_time_SKL = timeit(lambda: SKL_Classifier(x_train, y_train, x_test, max_depth), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in SKL method: {Total_time_SKL:.4f} s")
    if max_depth is None:
        Total_time_OWN= timeit(lambda: OWN_Classifier(x_train, y_train, x_test, max_depth), number = 10)
        print(f"Total time for {10} steps and in OWN method: {Total_time_OWN:.4f} s")    
    else :
        Total_time_OWN= timeit(lambda: OWN_Classifier(x_train, y_train, x_test, max_depth), number = Time_steps)
        print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")  




# %% [markdown]
#Test na 3 SKL vs OWN
# %% 
################################
#Nieograniczona glebokosc
print("\n###################################################")
print("Max Tree depth: None")
print("###################################################")

print("CLUSTER Data:")
SKL_vs_OWN(x = x_cluster, y = y_cluster, max_depth = None)

print("-----------------------------------------------------------------------")
print("MOON Data:")
SKL_vs_OWN(x = x_moon, y = y_moon, max_depth = None)

print("-----------------------------------------------------------------------")
print("MultiMode Data:")
SKL_vs_OWN(x = x_multimode, y = y_multimode, max_depth = None)

##############################
#Nieograniczona glebokosc
print("\n###################################################")
print("Max Tree depth: n-features")
print("###################################################")

print("CLUSTER Data:")
n_features = x_cluster.shape[1]
SKL_vs_OWN(x = x_cluster, y = y_cluster, max_depth = n_features)

print("-----------------------------------------------------------------------")
print("MOON Data:")
n_features = x_moon.shape[1]
SKL_vs_OWN(x = x_moon, y = y_moon, max_depth = n_features)

print("-----------------------------------------------------------------------")
print("MultiMode Data:")
n_features = x_multimode.shape[1]
SKL_vs_OWN(x = x_multimode, y = y_multimode, max_depth = n_features)


########################
#Glebokosc z iloscia klastrow
print("\n###################################################")
print("Max Tree depth: Cluster number")
print("###################################################")

print("CLUSTER Data:")
n_classes = int(len(np.unique(y_cluster)))
SKL_vs_OWN(x = x_cluster, y = y_cluster, max_depth = n_classes)

print("-----------------------------------------------------------------------")
print("MOON Data:")
n_classes = int(len(np.unique(y_moon)))
SKL_vs_OWN(x = x_moon, y = y_moon, max_depth = n_classes)

print("-----------------------------------------------------------------------")
print("MultiMode Data:")
n_classes = int(len(np.unique(y_multimode)))
SKL_vs_OWN(x = x_multimode, y = y_multimode, max_depth = n_classes)

# %% [markdown]
#Na 4
# %% 
#Ladowanie danych
htru2 = fetch_ucirepo(id=372) 
  
# data (as pandas dataframes) 
x_pulsar = htru2.data.features 
y_pulsar = htru2.data.targets 

# %% [markdown]
#Preprocessing danych
# %%
print(htru2.variables)

#Podzial na dane treningowe testowe
x_train, x_test, y_train, y_test = train_test_split(x_pulsar, y_pulsar, test_size=0.2, random_state=r_state, stratify = y_pulsar)

#1 missing values
print("No missing values --> No need for removal / replacement")

print("Also all values continuous --> No need for distinguishing categorical and numerical")


#####

print("Winsoryzacja + normalizacja")
columns = x_train.columns.tolist()
for column in columns:
    #Train part
    #########################################
    #Winsoryzacja
    c_data_train = x_train[column]
    
    Q1 = c_data_train.quantile(0.25)
    Q3 = c_data_train.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    c_data_train = c_data_train.clip(lower=lower_limit, upper=upper_limit)
    ###################
    #Reshape
    #Normalizacja
    arr_train = c_data_train.values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaler.fit(arr_train)
    c_data_train = scaler.transform(arr_train)
    
    #Wstawienie danych
    x_train[column] = c_data_train.flatten()
    
    #Test part
    #########################################
    #Winsoryzacja
    c_data_test = x_test[column]
    
    c_data_test= c_data_test.clip(lower=lower_limit, upper=upper_limit)
    ###################
    #Normalizacja
    arr_test = c_data_test.values.reshape(-1, 1)
    c_data_test = scaler.transform(arr_test)
    
    #Wstawienie danych
    x_test[column] = c_data_test.flatten()


# %% [markdown]
# Przygotowanie grida
# %%

param_grid = {
    'criterion':           ['gini', 'entropy'],
    'max_depth':           [None,2, 5, 10, 15, 20, 30, 40, 50, 75],
    #minimalna liczba próbek, by rozdzielić węzeł
    'min_samples_split':   [2, 4, 5, 8, 10, 20, 50],
    #minimalna liczba próbek w liściu
    'min_samples_leaf':    [1, 2, 4, 5, 8, 10, 20, 50]
}

############
#Drzewo decyzyjne
model = DecisionTreeClassifier(random_state = r_state)

#Grid search
grid = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs=-1,  #Wszystkie rdzenie (-1)          
    verbose = 1
)

##Tet najlepszych hiperparametrow
grid.fit(x_train, y_train)

# %% [markdown]
# Najlepszy model i otrzymane parametry
# %%

#Parametry i wynik
print("Najlepsze parametry:", grid.best_params_)
print("Train accuracy:", grid.best_score_)

#Zbior testowy i wynik
best_clf = grid.best_estimator_
y_pred = best_clf.predict(x_test)
print("Test accuracy:", skl.metrics.accuracy_score(y_test, y_pred))











