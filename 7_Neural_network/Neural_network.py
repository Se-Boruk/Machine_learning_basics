# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from timeit import timeit
from sklearn.datasets import make_blobs
# %%
r_state = 102452
#Steps to check avg time
Time_steps = 1

n_samples = int(str(r_state)[0:2] + str(r_state)[-2:])

#Multimode data
# Create 6 blobs (3 for each class)
x_multimode, y_multimode = make_blobs(n_samples=n_samples,
                                      centers=6,
                                      n_features=2,
                                      cluster_std=1.0,
                                      random_state = r_state)
#klastry do jednej klasy
y_multimode = (y_multimode >= 3).astype(int)


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

def Calculate_metrics(y_true, y_pred, y_prob, show_plots=True, title="Metrics"):
    n_classes = len(np.unique(y_true))
    metrics = {}

    if n_classes == 2:
        # Binary classification
        tn, fp, fn, tp = skl.metrics.confusion_matrix(y_true, y_pred).ravel()
        try:
            y_prob_1 = y_prob[:, 1]
        except:
            y_prob_1 = y_prob
        
        acc = skl.metrics.accuracy_score(y_true, y_pred)
        recall = skl.metrics.recall_score(y_true, y_pred)
        specificity = float(tn / (tn + fp))
        auc = skl.metrics.roc_auc_score(y_true, y_prob_1)
        
        metrics["Conf_matrix"] = {"True_positive": tp, "False_positive": fp, "False_negative": fn, "True_negative": tn}
        metrics["Accuracy"] = acc
        metrics["Recall"] = recall
        metrics["Specificity"] = specificity
        metrics["AUC_ROC"] = auc

        if show_plots:
            cm = skl.metrics.confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(12,5))
            plt.suptitle(title)

            plt.subplot(1, 2, 1)
            plt.imshow(cm, cmap='Greens')
            plt.title("Confusion matrix")
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks([0, 1], ['Class 0', 'Class 1'])
            plt.yticks([0, 1], ['Class 0', 'Class 1'])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

            plt.subplot(1, 2, 2)
            plt.title("ROC Curve")
            fpr, tpr, _ = skl.metrics.roc_curve(y_true, y_prob_1)
            plt.plot(fpr, tpr, color='green', label=f'ROC Curve (AUC = {auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    else:
        #Multiclass classification
        cm = skl.metrics.confusion_matrix(y_true, y_pred)
        acc = skl.metrics.accuracy_score(y_true, y_pred)
        recall_per_class = skl.metrics.recall_score(y_true, y_pred, average=None)
        
        #Calculate specificity per class:
        specificity_per_class = []
        for i in range(n_classes):
            #True negatives:
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        specificity_per_class = np.array(specificity_per_class)
        
        try:
            auc = skl.metrics.roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception as e:
            print(f"ROC AUC calculation error: {e}")
            auc = None
        
        metrics["Conf_matrix"] = cm
        metrics["Accuracy"] = acc
        metrics["Recall"] = recall_per_class
        metrics["Specificity"] = specificity_per_class
        

        if show_plots:
            plt.figure(figsize=(14, 6))
            plt.suptitle(title)

            plt.subplot(1, 2, 1)
            plt.imshow(cm, cmap='Greens')
            plt.title("Confusion matrix")
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(np.arange(n_classes), [f'Class {i}' for i in range(n_classes)])
            plt.yticks(np.arange(n_classes), [f'Class {i}' for i in range(n_classes)])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

            plt.subplot(1, 2, 2)
            plt.title("Avg ROC Curve")
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            auc_scores = []
            
            for i in range(n_classes):
                fpr, tpr, _ = skl.metrics.roc_curve((y_true == i).astype(int), y_prob[:, i])
                auc_score = skl.metrics.auc(fpr, tpr)
                auc_scores.append(auc_score)
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
            
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(auc_scores)
            
            plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f})')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            metrics["AUC_ROC"] = np.array(auc_scores).mean()

    return metrics

class MLP:
    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        #Wielkosc sieci
        self.LENGTH = len(layer_sizes) - 1 
        #Wagi
        self.W = []
        #Bias
        self.B = []
        # Budowa sieci - odpowiednie wielkosci macierzy wag i bias
        for i in range(self.LENGTH):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            
            self.W.append(np.random.randn(n_in, n_out))
            self.B.append(np.zeros(n_out))
        
        #Lista arrayow, kazdy array to warstwa
        #Kazda kolumna to 'wirtualny' neuron, mnozniki poszczegolnych wejsc na ten neuron z wierszy - cech
        
        #Bias jest skonstruowany podobnie, ale tam nie ma cech, 
        #jest po prostu jedna wartosc dla neuronu - czyli BIAS
            
        
    def Test(self):
        return self.LENGTH, self.W, self.B  
      
    #Activation functions
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        #Clip aby zapobiec overflow - wczesniej wystepowal dosc czesto
        z = np.clip(z, -500, 500)  
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y, epochs=100, lr=0.001, batch_size=None, verbose = True):
        
        #Rozmiar zbioru treningowego, wejsciowego
        n = len(X)
        
        #Jesli batch size nie jest zdefiniowany - brak czyli trening na pojedynczym batchu
        if batch_size is None:
            batch_size = n
            
        for epoch in range(1, epochs+1):
            #Losowa kolejnosc próbek od 0 do n = len(x)
            perm = np.random.permutation(n)

            #Pomieszane probki, taka by byla losowosc przy treningu batchowym
            X_perm, Y_perm = X[perm], Y[perm]
            
            #Trening po batchach
            for start in range(0, n, batch_size):
                x_batch = X_perm[start:start+batch_size]
                y_batch = Y_perm[start:start+batch_size]
                
                #Obliczenia sieci
                #---------------------------------------------------------------
                #Sygnal przekazywany do nastepnej warstwy, 
                #w pierwszej warstwie jest to po prostu sygnal wejsciowy, 
                #potem bedzie to wyjscie neuronow 1 warstwy
                A = x_batch
                
                #Trzymamy wyjscia warstwy przed funkcja aktywacji, oraz po
                Zs, As = [], [A]
                #Przejscie po kazdej warstwie
                for i in range(self.LENGTH):
                    #obliczamy wejscie * wagi + bias
                    Z = A @ self.W[i] + self.B[i]
                    Zs.append(Z)
                    
                    #Jesli warstwa wewnetrzna, relu
                    #Jesli warstwa koncowa, sigmoid
                    if i < self.LENGTH - 1:
                        A = self.relu(Z)
                    else:
                        A = self.sigmoid(Z)

                    #Dodaj wyjscie po aktywacji do listy wyjsc z warstwy
                    As.append(A)
                #---------------------------------------------------------------
                #Blad sieci, korekcja wag
                
                #Oblizcenie roznicy pomiedzy batchem a predykcja
                delta = As[-1] - y_batch
                
                #Petla od tylu, od koncowej warstwy do pierwszej
                for i in reversed(range(self.LENGTH)):
                    A_prev = As[i]
                    
                    #Gradient wag i biasu
                    grad_w = (A_prev.T @ delta) / batch_size
                    grad_b = delta.mean(axis=0)
                    
                    #Korekta wag o gradient *learning rate
                    self.W[i] -= lr * grad_w
                    self.B[i] -= lr * grad_b
                    
                    #Wsteczna propagacja bledu - przekazujemy jak 
                    #bardzo output poprzedniej warstwy wplywa na output w kolejnej
                    if i > 0:
                        delta = delta @ self.W[i].T
                        delta *= self.relu_derivative(Zs[i-1])
                        
                #---------------------------------------------------------------
            if verbose:
                if epoch % 10 == 0:
                    preds = self.predict(X)
                    
                    acc = (preds == np.argmax(Y, axis=1))
                    acc = acc.mean()
                    
                    print(f"Epoch {epoch}/{epochs} — Acc: {acc:.4f}")

    def predict(self, X):
        A = X

        for i in range(self.LENGTH):
            Z = A @ self.W[i] + self.B[i]
            A = self.relu(Z) if i < self.LENGTH-1 else self.sigmoid(Z)
            
        return A.argmax(axis=1)
    
    def predict_proba(self, X):
        A = X

        for i in range(self.LENGTH):
            Z = A @ self.W[i] + self.B[i]
            if i < self.LENGTH - 1:
                A = self.relu(Z)
            else:
                A = self.sigmoid(Z)
        return A
        



 
def OWN(x, y, epochs, lr, batch_size = None, layers = [2,12,24,12,6,2], time_test = False):
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #Onehot data
    n_classes = len(np.unique(y))
    
    y_onehot_train = np.zeros((len(y_train), n_classes))
    y_onehot_train[np.arange(len(y_train)), y_train] = 1
    
    
    y_onehot_test = np.zeros((len(y_test), n_classes))
    y_onehot_test[np.arange(len(y_test)), y_test] = 1
    

    ###################################################
    model = MLP(layers)
    
    model.fit(x_train, y_onehot_train, epochs = epochs, lr = lr, batch_size = batch_size)
    
    
    
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    #Plot only 2 dim data, more or less throw away
    if x_train.shape[1] == 2:
        #Wykresy
        plt.figure()
        #Subplot with decision boundary
        plt.suptitle("Neural Network")
        
        
        #OWN section
        plt.subplot(2,1,1)
        Visualize_data(x_train, y_train, "Train Data")
        Plot_meshgrid(x_train, y_train, classifier = model)
        
        plt.subplot(2,1,2)
        Visualize_data(x_test, y_test, "Test Data")
        Plot_meshgrid(x_train, y_train, classifier = model)
    

    ####################################################
    print("OWN metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_OWN = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred,
                                         y_prob = y_proba,
                                         title = "Metrics"
                                         )
    
    #Printing metrics
    for dictionary, header in zip([Metrics_dict_OWN] , ["Metrics"]):
        print("\n")
        print(header)
        print("Accuracy: ",dictionary.get("Accuracy"))
        print("Recall: ",dictionary.get("Recall"))
        print("Specificity: ",dictionary.get("Specificity"))
        print("AUC_ROC: ",dictionary.get("AUC_ROC"))
        
    #Time of each method

    Total_time_OWN= timeit(lambda:     model.fit(x_train,
                                                 y_onehot_train,
                                                 epochs = epochs,
                                                 lr = lr,
                                                 batch_size = batch_size,
                                                 verbose = False) , number = Time_steps)
    print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")    



# %% [markdown]
#Na 3 i 4 (relu i logistyczna funkcja aktywacji + modularna struktura)
#%%
print("\n########################################################################")
print("\nMultimode data")
OWN(x_multimode,
    y_multimode,
    epochs = 150,
    lr = 0.0001,
    batch_size = 32,
    layers = [2,6,12,24,12,6,2],
    time_test = True
    )

# %% [markdown]
#Na 3 i 4 (relu i logistyczna funkcja aktywacji + modularna struktura)
#%%
print("\n########################################################################")
print("\nMnist dataset")
from sklearn.datasets import fetch_openml

#Ladownie mnist z skl
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x_mnist = mnist['data'].astype(np.float32) / 255.0
y_mnist = mnist['target'].astype(int)

# Normalizacja

x_mnist = x_mnist / 255 

OWN(x_mnist,
    y_mnist,
    epochs = 20,
    lr = 0.001,
    batch_size = 32,
    layers = [x_mnist.shape[1],128, 64, 32, 10]
    )












