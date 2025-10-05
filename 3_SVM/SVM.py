# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, LinearConstraint
# %% [markdown]
#Stworzenie zestawu danych
# %%
r_state = 102452
#Steps to check avg time
Time_steps = 10

n_samples = int(str(r_state)[0:2] + str(r_state)[-2:])
n_samples = 444
x_cluster, y_cluster = skl.datasets.make_classification(
                                        n_samples=n_samples,
                                        n_features=2,
                                        n_informative=2,
                                        n_redundant=0,
                                        n_clusters_per_class=1,
                                        random_state=r_state
                                    )
y_cluster = np.where(y_cluster == 0, -1, 1)

x_moon, y_moon = skl.datasets.make_moons(n_samples=n_samples,
                                         random_state=r_state)
y_moon = np.where(y_moon == 0, -1, 1)

# %% [markdown]
#Definicja potrzebnych funkcji
# %%

class Kernel:
    def __call__(self, A, B):
        raise NotImplementedError

class LinearKernel(Kernel):
    def __call__(self, A, B):
        return A @ B.T

class PolyKernel(Kernel):
    def __init__(self, degree=3, gamma=None, coef0=1.0):
        self.degree = degree
        self.gamma  = gamma
        self.coef0  = coef0

    def __call__(self, A, B):

        #Gamma, jesli brak to 1/n cech
        g = self.gamma if self.gamma is not None else 1.0 / A.shape[1]

        #skalowanie przez gamma + coef, potem podniesienie do degree
        #Mamy 'symulacje' wielu wymiarow ale liczymy tylko jedną wagę. Mniej obliczen
        return (g * (A @ B.T) + self.coef0) ** self.degree

class RBFKernel(Kernel):
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, A, B):

        g = self.gamma if self.gamma is not None else 1.0 / A.shape[1]

        #Suma kwadratow kolumn
        sqA = np.sum(A**2, axis=1).reshape(-1, 1)
        #SUma kwadratow wierszy
        sqB = np.sum(B**2, axis=1).reshape(1, -1)

        #Odleglosci
        dists = sqA + sqB - 2 * (A @ B.T)

        #?Gaussian bump? Punkty bliskie ~1, punkty odlegle ~0
        return np.exp(-g * dists)

class OWN_SVM:
    def __init__(self, C=1.0, kernel=None):
        self.C = C
        self._kernel = kernel if kernel is not None else LinearKernel()
        self._lambdas = None
        self._suppX = None
        self._suppy = None


    def fit(self, X, y):
        n = X.shape[0]
        y = y.astype(float)

        def svm_loss(lambdas, X, y, kernel):
            K = kernel(X, X)
            sum_l = np.sum(lambdas)
            sum_quad = 0.5 * lambdas @ (np.outer(y, y) * K) @ lambdas
            return -(sum_l - sum_quad)

        x0 = np.random.rand(n) * 1e-3
        lin_constr = LinearConstraint(y, 0.0, 0.0)
        bounds = [(0, self.C)] * n

        res = minimize(
            fun=svm_loss,
            x0=x0,
            args=(X, y, self._kernel),
            method='SLSQP',
            bounds=bounds,
            constraints=[lin_constr],
            options={'maxiter': 2000, 'disp': False}
        )

        lambdas = res.x
        sv_mask = lambdas > 1e-5
        self._lambdas = lambdas[sv_mask]
        self._suppX    = X[sv_mask]
        self._suppy    = y[sv_mask]
        
        


    def transform(self, X):
        K = self._kernel(X, self._suppX)
        return (K * (self._lambdas * self._suppy)).sum(axis=1)

    def predict(self, X):
        y_prob = self.transform(X)
        y_pred = np.sign(y_prob)
        y_prob =  1.0 / (1.0 + np.exp(-y_prob))
        return y_pred, y_prob



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
def SKL_Classifier(x_train, y_train, x_test, kernel = 'linear'):
    # Create and train model
    model = skl.svm.SVC(kernel = kernel, probability = True)
    model.fit(x_train, y_train)
    
    # Predict
    y_pred = model.predict(x_test)
    
    #Take probs also for metric calculation
    y_prob = model.predict_proba(x_test)
    y_prob = y_prob[:, 1]

    return y_pred, y_prob

def OWN_Classifier(x_train, y_train, x_test, kernel = 'linear'):
    if kernel == 'linear':
        kernel = None
    elif kernel == 'rbf':
        kernel = RBFKernel(gamma=0.5)
    elif kernel == 'poly':
        kernel = PolyKernel(degree=4, gamma=None, coef0=1.0)
        
    # Create and train model
    model = OWN_SVM(kernel = kernel)
    model.fit(x_train, y_train)
    # Predict
    y_pred, y_prob = model.predict(x_test)
    
    return y_pred, y_prob

#Wizualizacja
def Visualize_data(x, y, title):
    colors = ['green','red']
    labels = ['Class -1','Class +1']
    classes = np.unique(y)

    for idx_cls, cls in enumerate(classes):
        mask = (y == cls)
        plt.scatter(
          x[mask,0], x[mask,1],
          c=colors[idx_cls],
          label=labels[idx_cls],
          edgecolor='k',
          s=60
        )
    plt.legend()
    plt.title(title)

def Plot_meshgrid(x_train, y_train, classifier, kernel = 'linear'):
    #Krok
    h = 0.02  
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    #Predykcja klasy dla punktów siatki
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    c,_ = classifier(x_train, y_train, grid_points, kernel = kernel)
    c = c.reshape(xx.shape)
    
    plt.contourf(xx, yy, c, alpha=0.3, cmap="coolwarm")
    

def SKL_vs_OWN(x,y):
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #predicting by SKL
    print("Predicting SKL...")
    y_pred_SKL, y_prob_SKL = SKL_Classifier(x_train, y_train, x_test)
    print("Predicting OWN...")
    y_pred_OWN, y_prob_OWN = OWN_Classifier(x_train, y_train, x_test)

    
    #Wykresy
    plt.figure()
    #Subplot with decision boundary
    plt.suptitle("SKL vs OWN model")
    
    print("Plotting SKL...")
    #SKL section
    plt.subplot(2,2,1)
    Visualize_data(x_train, y_train, "Train Data ; SKL")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier)
    
    plt.subplot(2,2,2)
    Visualize_data(x_test, y_test, "Test Data ; SKL")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier)
    
    print("Plotting OWN...")
    #OWN section
    plt.subplot(2,2,3)
    Visualize_data(x_train, y_train, "Train Data ; OWN")
    Plot_meshgrid(x_train, y_train, classifier = OWN_Classifier)
    
    plt.subplot(2,2,4)
    Visualize_data(x_test, y_test, "Test Data ; OWN")
    Plot_meshgrid(x_train, y_train, classifier = OWN_Classifier)
    

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
    #Total_time_SKL = timeit(lambda: SKL_Classifier(x_train, y_train, x_test), number = Time_steps)
    #print(f"Total time for {Time_steps} steps and in SKL method: {Total_time_SKL:.4f} s")
    
    #Total_time_OWN= timeit(lambda: OWN_Classifier(x_train, y_train, x_test), number = Time_steps)
    #print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")



def SKL_vs_OWN_4(x, y):
    ######################################################
    #Rbf kernel
    print("Comparision for rbf kernel:")
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #predicting by SKL
    print("Predicting SKL...")
    y_pred_SKL, y_prob_SKL = SKL_Classifier(x_train, y_train, x_test, kernel = 'rbf')
    print("Predicting OWN...")
    y_pred_OWN, y_prob_OWN = OWN_Classifier(x_train, y_train, x_test, kernel = 'rbf')

    
    #Wykresy
    plt.figure()
    #Subplot with decision boundary
    plt.suptitle("SKL vs OWN model")
    
    print("Plotting SKL...")
    #SKL section
    plt.subplot(2,2,1)
    Visualize_data(x_train, y_train, "Train Data ; SKL rbf kernel")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier, kernel = 'rbf')
    
    plt.subplot(2,2,2)
    Visualize_data(x_test, y_test, "Test Data ; SKL rbf kernel")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier, kernel = 'rbf')
    
    print("Plotting OWN...")
    #OWN section
    plt.subplot(2,2,3)
    Visualize_data(x_train, y_train, "Train Data ; OWN rbf kernel")
    Plot_meshgrid(x_train, y_train, classifier = OWN_Classifier, kernel = 'rbf')
    
    plt.subplot(2,2,4)
    Visualize_data(x_test, y_test, "Test Data ; OWN rbf kernel")
    Plot_meshgrid(x_train, y_train, classifier = OWN_Classifier, kernel = 'rbf')
    

    ####################################################
    print("SKL metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_SKL = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_SKL,
                                         y_prob = y_prob_SKL,
                                         title = "Metrics SKL rbf kernel"
                                         )
    print("OWN metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_OWN = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_OWN,
                                         y_prob = y_prob_OWN,
                                         title = "Metrics OWN rbf kernel"
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
    #Total_time_SKL = timeit(lambda: SKL_Classifier(x_train, y_train, x_test), number = Time_steps)
    #print(f"Total time for {Time_steps} steps and in SKL method: {Total_time_SKL:.4f} s")
    
    #Total_time_OWN= timeit(lambda: OWN_Classifier(x_train, y_train, x_test), number = Time_steps)
    #print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")
    
    
    ######################################################
    #Poly kernel
    print("Comparision for poly kernel:")
    
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #predicting by SKL
    print("Predicting SKL...")
    y_pred_SKL, y_prob_SKL = SKL_Classifier(x_train, y_train, x_test, kernel = 'poly')
    print("Predicting OWN...")
    y_pred_OWN, y_prob_OWN = OWN_Classifier(x_train, y_train, x_test, kernel = 'poly')

    
    #Wykresy
    plt.figure()
    #Subplot with decision boundary
    plt.suptitle("SKL vs OWN model")
    
    print("Plotting SKL...")
    #SKL section
    plt.subplot(2,2,1)
    Visualize_data(x_train, y_train, "Train Data ; SKL poly kernel")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier, kernel = 'poly')
    
    plt.subplot(2,2,2)
    Visualize_data(x_test, y_test, "Test Data ; SKL poly kernel")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier, kernel = 'poly')
    
    print("Plotting OWN...")
    #OWN section
    plt.subplot(2,2,3)
    Visualize_data(x_train, y_train, "Train Data ; OWN poly kernel")
    Plot_meshgrid(x_train, y_train, classifier = OWN_Classifier, kernel = 'poly')
    
    plt.subplot(2,2,4)
    Visualize_data(x_test, y_test, "Test Data ; OWN poly kernel")
    Plot_meshgrid(x_train, y_train, classifier = OWN_Classifier, kernel = 'poly')
    

    ####################################################
    print("SKL metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_SKL = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_SKL,
                                         y_prob = y_prob_SKL,
                                         title = "Metrics SKL poly kernel"
                                         )
    print("OWN metrics plot:")
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_OWN = Calculate_metrics(y_true = y_test,
                                         y_pred = y_pred_OWN,
                                         y_prob = y_prob_OWN,
                                         title = "Metrics OWN poly kernel"
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
    #Total_time_SKL = timeit(lambda: SKL_Classifier(x_train, y_train, x_test), number = Time_steps)
    #print(f"Total time for {Time_steps} steps and in SKL method: {Total_time_SKL:.4f} s")
    
    #Total_time_OWN= timeit(lambda: OWN_Classifier(x_train, y_train, x_test), number = Time_steps)
    #print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")




# %% [markdown]
#Test na 3 SKL vs OWN
# %% 
print("CLUSTER Data:")
SKL_vs_OWN(x = x_cluster, y = y_cluster)

print("-----------------------------------------------------------------------")
print("MOON Data:")
SKL_vs_OWN(x = x_moon, y = y_moon)

# %% [markdown]
#Test na 4 SKL vs OWN
# %% 
print("Ocena 4: rbf i poly kernel")

print("CLUSTER Data:")
SKL_vs_OWN_4(x = x_cluster, y = y_cluster)

print("-----------------------------------------------------------------------")
print("MOON Data:")
SKL_vs_OWN_4(x = x_moon, y = y_moon)








