# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from timeit import timeit
# %%
r_state = 102452
#Steps to check avg time
Time_steps = 50

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



# %% [markdown]
#Functions
#%%
class Neuron:
    def __init__(self):
        
        self.activation_functions = {
            'sigmoid':       (self.sigmoid, self.sigmoid_derivative),
            'heaviside':     (self.heaviside, self.heaviside_derivative),
            'sin':           (self.sin, self.sin_derivative),
            'tanh':          (self.tanh, self.tanh_derivative),
            'sign':          (self.sign, self.sign_derivative),
            'relu':          (self.relu, self.relu_derivative),
            'leaky_relu':    (self.leaky_relu, self.leaky_relu_derivative),
        }
    
    #Activation functions:
    ####################################################3
    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self,y_hat):
        return y_hat * (1 - y_hat)
    
    #########################################################
    #Heaviside zwraca 1 dla z >= 0 i 0 dla z < 0
    def heaviside(self, z):
        return (z >= 0).astype(float)
    
    #Pochodna geaviside zwraca zero wszędzie, bo jest niecciagla w z = 0
    def heaviside_derivative(self, z):
        return np.zeros_like(z)
    
    #########################################################
    def sin(self, z):
        return np.sin(z)

    def sin_derivative(self, z):
        return np.cos(z)
    
    #########################################################    
    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, y_hat):
        return 1.0 - np.square(y_hat)

    #########################################################
    #Signum zwraca -1 dla z < 0, 0 dla z = 0, 1 dla z > 0
    def sign(self, z):
        return np.sign(z)
    
    #Pochodna signum to zero wszedzie, bojest nieciągła w z = 0
    def sign_derivative(self, z):
        return np.zeros_like(z)
    
    #########################################################
    #ReLU zwraca max(0, z)
    def relu(self, z):
        return np.maximum(0, z)
    
    #Pochodna ReLU to 1 dla z > 0 i 0 dla z <= 0
    def relu_derivative(self, z):
        grad = np.zeros_like(z)
        grad[z > 0] = 1.0
        return grad
    
    #########################################################
    #Leaky ReLU zwraca z dla z >= 0 i alpha*z dla z < 0
    def leaky_relu(self, z, alpha=0.1):
        return np.where(z >= 0, z, alpha * z)

    #Pochodna Leaky ReLU to 1 dla z > 0 i alpha dla z <= 0
    def leaky_relu_derivative(self, z, alpha=0.1):
        grad = np.ones_like(z)
        grad[z < 0] = alpha
        return grad    
    
    #Fit
    #####################################  
    def fit(self, x, y, epochs, lr, batch_size = None, activation_f = "sigmoid", cosine_lr = False, lr_min = None, lr_max = None, verbose = False):   
        #Basic parameters
        self.x = x
        self.y = y
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
        
        #Weights and bias
        self.bias = 0
        shape = self.x.shape[1]
        self.w = np.random.randn(shape) * 0.01
        
        
        #Activation function:
        if activation_f not in self.activation_functions:
            available = ", ".join(self.activation_functions.keys())
            raise ValueError(f"Błędna funkcja aktywacji: '{activation_f}'. Dostępne opcje to: {available}")
        else:
            self.activation_func = self.activation_functions.get(activation_f)[0]   
            self.activation_dev = self.activation_functions.get(activation_f)[1]   
        

        #Zmienne do batch i cosine lr
        if self.batch_size is None: 
            self.batch_size = 1
            
        n_batches = len(self.y) // self.batch_size
        pi = 3.14159265359
        
        #For epoch in epochs
        for epoch in range(self.epochs):
            #Batch handling
            perm = np.random.permutation(len(self.y))
            x_shuffled = self.x[perm]
            y_shuffled = self.y[perm]
            
            
            
            if cosine_lr:
                if lr_min is None or lr_max is None:
                    raise ValueError("Przy cosine_lr musisz zdefiniowac lr_min i lr_max")
            #Cosine lr 
                current_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(pi * epoch / self.epochs))
            
            else:
                current_lr = self.lr
                
            #For batch

            for i in tqdm(range(n_batches),desc = f"Epoch: {epoch}", total = n_batches, disable=not verbose):
                #############################
                
                #Batch handling
                #Wyliczamy indeksy początku i końca aktualnego batcha
                start = i * self.batch_size
                end   = start + self.batch_size
                
                #Wycinamy batch ze spłaszczonych, przetasowanych danyc
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                #############################################################
                n = len(y_batch)
                ###########################################3
                #z = w1*x1 + w2*x2... + bias
                z = x_batch @ self.w + self.bias
                y_hat = self.activation_func(z)
                
                #Blad i delta (za pomoca pochodnej funkcji aktywacji)
                error = y_batch - y_hat
                
                #Delta w zależności od wybranej funkcji aktywacji:
                if activation_f in ['relu', 'leaky_relu']:
                    delta = error * self.activation_dev(z)
                else:
                    delta = error * self.activation_dev(y_hat)
                
                #Zredukowanie wartosci batcha do rzedu 1 probki
                grad_w = x_batch.T @ delta / n
                grad_b = np.mean(delta)
                
                #Aktualizacja wag - delta
                self.w += current_lr * grad_w
                self.bias += current_lr * grad_b
                ####################################################
                
            #Acc
            z_all = np.dot(self.x, self.w) + self.bias
            y_hat_all = self.activation_func(z_all)
            y_pred_all = (y_hat_all > 0.5).astype(int)
            
            acc = skl.metrics.accuracy_score(self.y, y_pred_all)
            
            if verbose:
                print("Used lr:", current_lr)
                print("Train_acc:",round(acc,4))
                print("===========================================================")
            
            
    def predict(self,x):
        
        z_all = np.dot(x, self.w) + self.bias
        y_hat_all = self.activation_func(z_all)
        y_pred_all = (y_hat_all > 0.5).astype(int)
        
        return y_pred_all
    
    def predict_proba(self, x):
        z = np.dot(x, self.w) + self.bias
        y_prob = self.activation_func(z)
        return y_prob
        



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



 
def SKL_vs_OWN(x,y, epochs, lr, batch_size = None, activation_f = "sigmoid", cosine_lr = False, lr_min = 0.00001, lr_max = 0.01):
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    

    model = Neuron()
    model.fit(x = x_train, 
              y = y_train,
              epochs = epochs,
              lr = lr,
              batch_size = batch_size,
              activation_f = activation_f,
              cosine_lr = cosine_lr,
              lr_min = lr_min,
              lr_max = lr_max
              ) 
    
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    #Plot only 2 dim data, more or less throw away
    if x_train.shape[1] == 2:
        #Wykresy
        plt.figure()
        #Subplot with decision boundary
        plt.suptitle(f"OWN model, activation: {activation_f}")
        
        
        #OWN section
        plt.subplot(2,1,1)
        Visualize_data(x_train, y_train, "Train Data ; OWN")
        Plot_meshgrid(x_train, y_train, classifier = model)
        
        plt.subplot(2,1,2)
        Visualize_data(x_test, y_test, "Test Data ; OWN")
        Plot_meshgrid(x_train, y_train, classifier = model)
    

    ####################################################
    print("OWN metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_OWN = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred,
                                         y_prob = y_proba,
                                         title = f"Metrics OWN, activation {activation_f}"
                                         )
    
    #Printing metrics
    for dictionary, header in zip([Metrics_dict_OWN] , ["OWN Metrics"]):
        print("\n")
        print(header)
        print("Accuracy: ",round(dictionary.get("Accuracy"),3))
        print("Recall: ",round(dictionary.get("Recall"),3))
        print("Specificity: ",round(dictionary.get("Specificity"),3))
        print("AUC_ROC: ",round(dictionary.get("AUC_ROC"),3))
        
    #Time of each method

    Total_time_OWN= timeit(lambda:     model.fit(x = x_train, 
                                                  y = y_train,
                                                  epochs = epochs,
                                                  lr = lr,
                                                  batch_size = batch_size,
                                                  activation_f = activation_f,
                                                  cosine_lr = cosine_lr,
                                                  lr_min = lr_min,
                                                  lr_max = lr_max
                                                  ) , number = Time_steps)
    print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")    



# %% [markdown]
#Na 3
#%%
print("\n########################################################################")
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "sigmoid"
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "sigmoid"
           )

# %% [markdown]
#Na 4
#%%
print("\n########################################################################")

#Heaviside
#########################################
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "heaviside"
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "heaviside"
           )
#########################################

#Sin
#########################################
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "sin"
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "sin"
           )
#########################################

#Tanh
#########################################
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "tanh"
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "tanh"
           )
#########################################    
    
#sign
#########################################
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "sign"
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "sign"
           )
#########################################    

#Relu
#########################################
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "relu"
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "relu"
           )
#########################################    

#leaky_relu
#########################################
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "leaky_relu"
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = None,
           activation_f = "leaky_relu"
           )
#########################################    

# %% [markdown]
#Na 5
#%%
print("\n########################################################################")
print("\nCluster data")
SKL_vs_OWN(x_cluster,
           y_cluster,
           epochs = 300,
           lr = 0.01,
           batch_size = 32,
           activation_f = "leaky_relu",
           cosine_lr = True
           )

print("\nMoon data")
SKL_vs_OWN(x_moon,
           y_moon,
           epochs = 300,
           lr = 0.01,
           batch_size = 32,
           activation_f = "leaky_relu",
           cosine_lr = True
           )


# %% [markdown]
# Logistyczna funkcja aktywacji sprawdzila sie dosc dobrze (ta na 3)




    
    
