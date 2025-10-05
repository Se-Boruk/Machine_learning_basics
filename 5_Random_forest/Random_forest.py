# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from timeit import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import warnings
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
                                                            n_features=3,
                                                            n_informative=3,
                                                            n_redundant=0,
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
def SKL_Classifier(x_train, y_train, x_test, depth, n_trees = None):
    
    if n_trees is None:
        #Wybor optymalnego rozmiaru lasu
        best_error = 1
        for number in tqdm(range(1, 50, 3), desc = "Checking OOB best tree number (range 1-50) for SKL classifier..."):
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Some inputs do not have OOB scores.*")
                
                model = RandomForestClassifier(
                    n_estimators=number,
                    max_depth=depth,
                    max_features='sqrt',
                    oob_score=True,
                    bootstrap=True,
                    criterion='gini',
                    random_state=r_state,
                    n_jobs=-1,
                    min_samples_split=2,
                    min_samples_leaf=1
                )
                
                model.fit(x_train, y_train)
                oob_error = 1 - model.oob_score_
                
                if best_error > oob_error:
                    best_error = oob_error
                    n_trees = number
        print(f"Best n_trees for SKL method testet using OOB is:   {n_trees}")

    #########################  
    #Wytrenowanie modelu po wybraniu optymalnego rozmiaru lasu za pomoca OOB

    model = RandomForestClassifier( n_estimators=n_trees,
                                    max_depth = depth,
                                    max_features='sqrt',
                                    oob_score=True,
                                    bootstrap=True, 
                                    criterion='gini',
                                    random_state = r_state,
                                    n_jobs = -1,
                                    min_samples_split=2,
                                    min_samples_leaf = 1
                                )
        
    model.fit(x_train, y_train)    
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    return n_trees, model, y_pred, y_prob

#Skl klasyfikator
def OWN_Classifier(x_train, y_train, x_test, depth, n_trees = None):
    
    if n_trees is None:
        #Wybor optymalnego rozmiaru lasu
        best_error = 1
        for number in tqdm(range(1, 50, 3), desc = "Checking OOB best tree number (range 1-50) for OWN classifier..."):
            #Inicjalizacja modelu
            model = RandomForest(n_trees = number,
                                 max_depth = depth,
                                 min_samples_split = 2
                                 )
            model.fit(x_train, y_train)

            oob_error = 1 - model.oob_score(x_train, y_train)
            
            if best_error > oob_error:
                best_error = oob_error
                n_trees = number
                
        print(f"Best n_trees for SKL method testet using OOB is:   {n_trees}")
        
        
    model = RandomForest(n_trees = n_trees,
                         max_depth = depth,
                         min_samples_split = 2
                         )
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    return n_trees, model, y_pred, y_prob


class RandomForest:
    def __init__(self, n_trees = 10, max_depth=5, min_samples_split=2):
        if max_depth is None:
            self.max_depth = 999
        else:
            self.max_depth = max_depth
        
        self.n_trees = n_trees
        
        self.min_samples_split = min_samples_split
        #Poki co brak korzenia
        self.root = None 
        #To bedzie zdefiniowane przez unique y potem
        self.n_classes_ = None
        #Features
        self.features = None
        #Bagging_features
        self.bag_features = None
        

    #Fit, zbudowanie drzewa, trening
    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y, dtype=int)
        #Ilosc klas
        self.n_classes_ = len(np.unique(y))
        #Ilosc cech
        self.lenght, self.features = X.shape
        self.bag_features = int(self.features**0.5)
        
        #Procedura baggingu (losujemy procent probek dla danego drzewa
        #(las ma inne losowania wiec wykorzystamy wszyskie probki)) i uzyskamy wieksza generalizacje
        #Poza tym, probki poza drzewem będą uzyte to jego ewaluacji i oceny 

        #Stworz zbiory danych dla n drzew w lesie
        self.train_idx_list = []
        self.val_idx_list = []
        for i in range(self.n_trees):
            train_idx, val_idx = self.bootstrap_train_val_indices(n_samples = self.lenght,
                                                                  bootstrap_frac = 0.65,
                                                                  random_state = r_state
                                                                  )
            self.train_idx_list.append(train_idx)
            self.val_idx_list.append(val_idx)
        
        #Stworz n drzew w lesie
        self.tree_list = []
        for i in range(self.n_trees):
            #Ustawiamy korzeń (od jakiego 'drzewa' - bo jest to zbior drzew polaczanych tak naprawdę, zacząć)
            root = self.build_tree(X[self.train_idx_list[i]], y[self.train_idx_list[i]], depth=0)

            self.tree_list.append(root)
            
            
    def oob_score(self, X, y):
        tree_list = self.tree_list
        val_idx_list = self.val_idx_list
        # Number of training samples
        n_samples = X.shape[0]
    
        # Initialize vote storage: one list per sample
        votes = [[] for _ in range(n_samples)]
    
        # For each tree and its OOB indices
        for tree, oob_idx in zip(tree_list, val_idx_list):
            for sample_idx in oob_idx:
                x_sample = X[sample_idx]
                node = tree
    
                # Traverse down to a leaf
                while not node.is_leaf():
                    if x_sample[node.feat] <= node.thresh:
                        node = node.left
                    else:
                        node = node.right
    
                # Record the hard vote (class with max count in the leaf)
                predicted_class = node.counts.argmax()
                votes[sample_idx].append(predicted_class)
    
        # Prepare arrays to hold true labels (y_val) and predictions (y_pred)
        y_val = []
        y_pred = []
    
        # Gather OOB predictions per sample
        for sample_idx, vote_list in enumerate(votes):
            if len(vote_list) > 0:
                # Only consider samples with at least one vote
                y_val.append(y[sample_idx])
                # Majority vote via bincount + argmax
                majority_class = np.bincount(vote_list).argmax()
                y_pred.append(majority_class)
    
        # Convert to NumPy arrays for metric computation
        y_val = np.array(y_val)
        y_pred = np.array(y_pred)
    
        # Compute accuracy on OOB samples
        oob_acc = skl.metrics.accuracy_score(y_val, y_pred)
    
        return oob_acc


# Example usage (assuming you have tree_list and val_idx_list):
# oob_accuracy = oob_score(tree_list, val_idx_list, X_train, y_train)
# print("OOB Accuracy:", oob_accuracy)

    
    
    def bootstrap_train_val_indices(self, n_samples, bootstrap_frac=0.65, random_state=None):

        rng = np.random.default_rng(random_state)
        n_boot = int(np.floor(bootstrap_frac * n_samples))
        
        #Sample with repetitions
        train_idx = rng.integers(0, n_samples, size=n_boot)
        
        #OOB (validation) idx
        all_idx = np.arange(n_samples)
        #Unique indices in the bootstrap sample
        seen = np.unique(train_idx)
        #negation - OOB data for validation
        val_idx = np.setdiff1d(all_idx, seen, assume_unique=True)
    
        return train_idx, val_idx


    #Gini score
    def gini(self, y):

        #wystąpienia każdej klasy
        counts = np.bincount(y, minlength=self.n_classes_)
        
        #Prawdopodobienstwa
        probs = counts / counts.sum()
        
        #Gini = 1 - sum p^2
        return 1.0 - np.sum(probs**2)

    def best_split(self, X, y):
        #Sprawdzamy dla kazdej unikalnej wartosci cechy w probkach treningowych wartosc entropii po podziale
        #A następnie powtarzamy dla kazdej cechy i sprawdzamy najlepsza wartosc

        best_gain = 0
        best_feat, best_thresh = None, None
        parent_entropy = self.gini(y)
        #Ile probek i ile cech - tyle bedzie kombinacji
        n_samples, n_feats = X.shape
        
        bag_feats = np.random.choice(n_feats, size = self.bag_features, replace=False).tolist()
        
    
        #1) Dla każdej cechy ze zbioru bagging
        for feat in bag_feats:
    
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
                       - ( w_left  * self.gini(y_left) 
                           + w_right * self.gini(y_right) )
    
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
            return RandomForest.Node(counts=counts)

        #Najlepszy podzial
        feat, thresh = self.best_split(X, y)
        if feat is None:
            counts = np.bincount(y, minlength=self.n_classes_)
            return RandomForest.Node(counts=counts)

        #Podzial danychi jesli brak konca do tej pory --> czyli mozna jesszcze tworzyc warunki/drzewa
        #Stworz nowe drzewo
        left_mask = X[:, feat] <= thresh
        left_node = self.build_tree(X[left_mask],  y[left_mask],  depth+1)
        right_node = self.build_tree(X[~left_mask], y[~left_mask], depth+1)
        return RandomForest.Node(feat=feat, thresh=thresh, left=left_node, right=right_node)


    #Prawdopodobienstwo (przez liczebnosc klas w zbiorze treningowym w tym 'lisciu')
    def predict_proba(self, X):
        forest = self.tree_list
        
        n_samples = X.shape[0]
        n_classes = self.n_classes_
        all_probs = np.zeros((n_samples, n_classes))
    
        # Zbieramy i sumujemy prawdopodobieństwa z każdego drzewa
        for tree in forest:
            for i, x in enumerate(X):
                node = tree
                while not node.is_leaf():
                    node = node.left if x[node.feat] <= node.thresh else node.right
                counts = node.counts
                total = counts.sum()
                if total > 0:
                    all_probs[i] += counts / total
        # Uśredniamy
        all_probs /= len(forest)
        return all_probs
    
    def predict(self, X):
        
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)



    
def SKL_vs_OWN(x,y, max_depth):
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #predicting by SKL
    skl_trees, skl_model, y_pred_SKL, y_prob_SKL = SKL_Classifier(x_train, y_train, x_test, depth = max_depth)
    own_tree, own_model, y_pred_OWN, y_prob_OWN = OWN_Classifier(x_train, y_train, x_test, depth = max_depth)

    #Plot only 2 dim data, more or less throw away
    if x_train.shape[1] == 2:
        #Wykresy
        plt.figure()
        #Subplot with decision boundary
        plt.suptitle("SKL vs OWN model")
        
        #SKL section
        plt.subplot(2,2,1)
        Visualize_data(x_train, y_train, f"Train Data ; SKL (trees: {skl_trees})")
        Plot_meshgrid(x_train, y_train, classifier = skl_model)
        
        plt.subplot(2,2,2)
        Visualize_data(x_test, y_test, f"Test Data ; SKL (trees: {skl_trees})")
        Plot_meshgrid(x_train, y_train, classifier = skl_model)
        
        #OWN section
        plt.subplot(2,2,3)
        Visualize_data(x_train, y_train, f"Train Data ; OWN (trees: {skl_trees})")
        Plot_meshgrid(x_train, y_train, classifier = own_model)
        
        plt.subplot(2,2,4)
        Visualize_data(x_test, y_test, f"Test Data ; OWN (trees: {skl_trees})")
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
    Total_time_SKL = timeit(lambda: SKL_Classifier(x_train, y_train, x_test, max_depth, n_trees = 30), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in SKL method: {Total_time_SKL:.4f} s")

    Total_time_OWN= timeit(lambda: OWN_Classifier(x_train, y_train, x_test, max_depth,n_trees = 30), number = 10)
    print(f"Total time for {10} steps and in OWN method: {Total_time_OWN:.4f} s")    





# %% [markdown]
#Test na 3 SKL vs OWN
# %% 
################################
#Ograniczona glebokosc
Max_tree_depth = 10

print("\n###################################################")
print(f"Max Tree depth: {Max_tree_depth}")
print("###################################################")

print("CLUSTER Data:")
SKL_vs_OWN(x = x_cluster, y = y_cluster, max_depth = Max_tree_depth)

print("-----------------------------------------------------------------------")
print("MOON Data:")
SKL_vs_OWN(x = x_moon, y = y_moon, max_depth = Max_tree_depth)

print("-----------------------------------------------------------------------")
print("MultiMode Data:")
SKL_vs_OWN(x = x_multimode, y = y_multimode, max_depth = Max_tree_depth)


# %% [markdown]
#Na 4

# %%

#Load asteroids
df = pd.read_csv("dataset.csv")

# %% [markdown]
# Data preprocessing #1

#Pipeline
#1 Drop if y has nan
#2 Drop if column is in 30% nan
#4 2 way filter - remove from column if some numerical is in the string, and if some string in numerical
#5 Remove unnecesary columns (id, name etc.), Split for x and y
#6 train test split
#7.1 if numerical:
    #Fill nans with mean
    #Winsorize
    #min max scaler
#7.2 if categorical:
    #Fil nans with dominant
    #Onehot categorical values


#%%
def prepare_data(df, y_label_column_name):
    #Remove nans from y label
    df = df.dropna(subset=[y_label_column_name])
    
    #Drop columns where more than 30% values are NaN
    thresh = len(df) * 0.3
    df = df.dropna(axis=1, thresh=thresh)

    # Data split and preprocessing #2
    
    #2 way filter - throw away numerical from mostly categorical column and vice versa
    for col in df.columns:
        c = df[col]
        #detect the majority datatype in column
        major_type = c.apply(type).value_counts().idxmax()
        
        if major_type == str:
            #mostly categorical, keep only strings
            is_valid = c.apply(lambda val: isinstance(val, str))
            df[col] = c[is_valid]
        else:
            #mostly numerical, keep only numeric values, leave bool in peace alone
            is_valid = c.apply(lambda val: isinstance(val, (int, float)) and not isinstance(val, bool))
            df[col] = c[is_valid]
    
    
    
    
    #Wyrzucenie kolumn nieistotnych i podzial na x i y
    x = df.drop(columns=['full_name' , "orbit_id", "id", y_label_column_name])
    y = df[y_label_column_name]
    
    
    #How many nulls before
    print("Before preprocessing:")
    print("NaNs in X:", x.isnull().sum().sum())
    print("NaNs in y:", y.isnull().sum())
    
    
    #Train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify = y, random_state = r_state)
    
    
    #Preprocessing
    columns = x_train.columns
    for column in columns:
        
        c_data = x_train[column]
        #Take dtypes of elements in the series column of dataframe
        dtype = c_data.apply(type).value_counts().idxmax()
        #Check if string ---> Categorical
        categorical = (dtype == str)
    
        if not categorical:
            #Fill nans
            mean = x_train[column].mean()
            x_train[column] = x_train[column].fillna(mean)
            x_test[column] = x_test[column].fillna(mean)
            #########################3#####
            #Winsoryzacja
            c_data_train = x_train[column]
            
            Q1 = c_data_train.quantile(0.25)
            Q3 = c_data_train.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            #Train winsorize
            x_train[column] = c_data_train.clip(lower=lower_limit, upper=upper_limit)
            #Test winsorize
            c_data_test = x_test[column]
            x_test[column]  = c_data_test.clip(lower=lower_limit, upper=upper_limit)
            
            #############################
            #Normalize data if numerical
            scaler = MinMaxScaler()
            scaler.fit(np.array(x_train[column]).reshape(-1,1))
            x_train_scaled = scaler.transform(np.array(x_train[column]).reshape(-1,1))
            x_test_scaled = scaler.transform(np.array(x_test[column]).reshape(-1,1))
    
            #Input the scaled column into the region
            x_train[column] = x_train_scaled
            x_test[column] = x_test_scaled
            
        else:
            #Fill nans with dominant
            dominant = x_train[column].mode()[0]
            x_train[column] = x_train[column].fillna(dominant)
            x_test[column] = x_test[column].fillna(dominant)
            
            #GPT PART Memory lekeage Ktorego nie potrafilem zalatac 
            ###############################
            #Compute the union of categories once
            cats = sorted(
                set(x_train[column].dropna().unique()) |
                set(x_test[column].dropna().unique())
            )
            
            #Cast both to Categorical with the same categories
            x_train[column] = pd.Categorical(x_train[column], categories=cats)
            x_test[column] = pd.Categorical(x_test[column], categories=cats)
            
            #One-hot them separately but with identical columns
            d_train = pd.get_dummies(x_train[column], prefix=column, dtype=int)
            d_test  = pd.get_dummies(x_test[column], prefix=column, dtype=int)
            
            #Drop original and join new
            x_train = x_train.drop(columns=[column]).join(d_train)
            x_test  = x_test.drop(columns=[column]).join(d_test)
            ##################################################
                    
    #######################
    #Onehot the data
    y_train = y_train.map({"N": 0, "Y": 1})  
    y_test = y_test.map({"N": 0, "Y": 1})    
    
    print("After preprocessing:")
    print("NaNs in X:", x_train.isnull().sum().sum())
    print("NaNs in y:", y_train.isnull().sum())
    
    print("NaNs in X:", x_test.isnull().sum().sum())
    print("NaNs in y:", y_test.isnull().sum())
    
    return x_train, x_test, y_train, y_test



# %% [markdown]
# Na 4 Label NEO
#%%
print("\n Y label: neo")
df_neo = df.copy()
x_train, x_test, y_train, y_test = prepare_data(df_neo, y_label_column_name = "neo")
# Przygotowanie grida
# %%

param_grid = {
    'n_estimators':        [5, 20, 50, 75],
    'criterion':           ['gini', 'entropy'],
    'max_depth':           [2, 5, 10, 15],
    'min_samples_split':   [2, 5, 20],
    'min_samples_leaf':    [2, 5, 20]
}

############
#Drzewo decyzyjne
model = RandomForestClassifier(random_state = r_state)

#Grid search
grid = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs=-1,  #Wszystkie rdzenie (-1)          
    verbose = 2
)

##Tet najlepszych hiperparametrow
grid.fit(x_train, y_train)

# %%
# Najlepszy model ( i otrzymane parametry


#Parametry i wynik
print("Najlepsze parametry:", grid.best_params_)
print("Train accuracy:", grid.best_score_)

#Zbior testowy i wynik
best_clf = grid.best_estimator_
y_pred = best_clf.predict(x_test)
print("Test accuracy:", skl.metrics.accuracy_score(y_test, y_pred))








# %% [markdown]
# Na 4 Label PHA
#%%
print("\n Y label: pha")
df_pha = df.copy()
x_train, x_test, y_train, y_test = prepare_data(df_pha, y_label_column_name = "pha")
# Przygotowanie grida
# %%

param_grid = {
    'n_estimators':        [5, 20, 50, 75],
    'criterion':           ['gini', 'entropy'],
    'max_depth':           [5, 10, 15],
    'min_samples_split':   [5, 20],
    'min_samples_leaf':    [5, 20]
}

############
#Drzewo decyzyjne
model = RandomForestClassifier(random_state = r_state)

#Grid search
grid = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs=-1,  #Wszystkie rdzenie (-1)          
    verbose = 2
)

##Tet najlepszych hiperparametrow
grid.fit(x_train, y_train)

# %%
# Najlepszy model ( i otrzymane parametry


#Parametry i wynik
print("Najlepsze parametry:", grid.best_params_)
print("Train accuracy:", grid.best_score_)

#Zbior testowy i wynik
best_clf = grid.best_estimator_
y_pred = best_clf.predict(x_test)
print("Test accuracy:", skl.metrics.accuracy_score(y_test, y_pred))

