# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from timeit import timeit
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
#Definicja potrzebnych funkcji
# %%
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
def SKL_Classifier(x_train, y_train, x_test):
    
    model = skl.linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    return y_pred, y_prob


def OWN_Classifier(x_train,y_train,x_test):
    #Define sigmoid
    def sigmoid(s, b=1):
        return 1 / (1 + np.exp(-b * s))
    
    #Define shortened own regressor
    def Own_regressor(x , weights):
        y_prob = x*weights
        y_prob = np.sum(y_prob,axis = 1)
        y_prob = sigmoid(y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        
        return y_pred, y_prob
    
    #Define own weight step update
    def weight_step(x,y,w,lr = 0.01):
        _, y_prob = Own_regressor(x, w)
        delta_w = -lr*(y-y_prob)*y_prob*(1-y_prob)
        delta_w = delta_w[:, np.newaxis]
        delta_w = delta_w*x

        delta_w = np.sum(delta_w,axis = 0)
        return delta_w
    
    #Find optimal logistic weights
    def Find_logistic_weights(x,y,tolerance = 1e-4, epochs = 100):
        w = np.random.rand(*(1,len(x[0])))

        for e in range(epochs):
            #Calculate delta_w
            delta_w = weight_step(x, y, w)
            #Update w
            w = w - delta_w
            
            y_pred, _ = Own_regressor(x,w)
            mse = np.mean((y - y_pred) ** 2)
            if mse <= tolerance:
                break
            
        return w   
    
    #Run target function:
    w = Find_logistic_weights(x_train, y_train)   
    y_pred, y_prob = Own_regressor(x_test, w)
    
    return y_pred, y_prob

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
    

def SKL_vs_OWN(x,y):
    #Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_state, stratify = y)
    
    #predicting by SKL
    y_pred_SKL, y_prob_SKL = SKL_Classifier(x_train, y_train, x_test)
    y_pred_OWN, y_prob_OWN = OWN_Classifier(x_train, y_train, x_test)

    
    #Wykresy
    plt.figure()
    #Subplot with decision boundary
    plt.suptitle("SKL vs OWN model [Both with BIAS]")
    
    #SKL section
    plt.subplot(2,2,1)
    Visualize_data(x_train, y_train, "Train Data ; SKL")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier)
    
    plt.subplot(2,2,2)
    Visualize_data(x_test, y_test, "Test Data ; SKL")
    Plot_meshgrid(x_train, y_train, classifier = SKL_Classifier)
    
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
    Total_time_SKL = timeit(lambda: SKL_Classifier(x_train, y_train, x_test), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in SKL method: {Total_time_SKL:.4f} s")
    
    print("Max epoch number is set to 100:")
    Total_time_OWN= timeit(lambda: OWN_Classifier(x_train, y_train, x_test), number = Time_steps)
    print(f"Total time for {Time_steps} steps and in OWN method: {Total_time_OWN:.4f} s")

# %% [markdown]
#Test na 3 SKL vs OWN
# %% 
print("CLUSTER Data:")
SKL_vs_OWN(x = x_cluster, y = y_cluster)

print("-----------------------------------------------------------------------")
print("MOON Data:")
SKL_vs_OWN(x = x_moon, y = y_moon)

# %% [markdown]
#Na ocene 4
# %% [markdown]
#Loading data, dropping columns
# %% 

weather_data = pd.read_csv("weatherAUS.csv")
columns = weather_data.columns.tolist()

if 'Risk-MM' in columns:
    print("Risk-MM in columns")
else:
    print("Risk-MM not in columns")

columns_to_drop = []
for column in columns:
    c_data = weather_data[column]
    
    missing_part = float(c_data.isna().mean())
    if missing_part > (1/3):
        columns_to_drop.append(column)
        
print("Columns to drop (>33% is nan): ",columns_to_drop)

#Wyrzucenie jakichkolwiek danych jesli ich label jest nan
drop = weather_data["RainTomorrow"].isna()
drop = ~drop
weather_data = weather_data[drop]


#Separacja kolumny rain tomorrow
label = weather_data["RainTomorrow"]
columns_to_drop.append("RainTomorrow")

weather_data = weather_data.drop(columns=columns_to_drop)
# %% 
#Winsorizacja (funkcja)
def Winsorize_column(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    return column.clip(lower=lower_limit, upper=upper_limit)



#Wtloczenie brakujacych danych oraz ewentualna winsoryzacja danych (dla numerycznych)
columns = weather_data.columns.tolist()
for column in columns:
    
    c_data = weather_data[column]
    #Take dtypes of elements in the series column of dataframe
    dtype = c_data.apply(type).value_counts().idxmax()
    #Check if string ---> Categorical
    categorical = (dtype == str)
    
    if categorical:
        dominant_element = c_data.value_counts().idxmax()
        c_data = c_data.fillna(dominant_element)
        
    else:
        #Median function with ignoring nans
        median_element = float(np.nanmedian(c_data))
        c_data = c_data.fillna(median_element)
        
        #Winsorization
        c_data = Winsorize_column(c_data)
        
    weather_data[column] = c_data
# %% [markdown]
#Trening wraz z ostatecznym dokonczeniem preprocesingu danych

# %%
from sklearn.preprocessing import MinMaxScaler

Regions_score_SKL = {}
Regions_score_OWN = {}
print("Training local models for each region...")
regions = np.unique(weather_data["Location"]).tolist()
for region in regions:

    #Taking given region
    binary_region = weather_data['Location'] == region
    region_weather = weather_data[binary_region].copy()
    region_label = label[binary_region].copy()
    
    #Dropping region (as it is not informative feature now)
    region_weather.drop(columns = ['Location'], inplace = True)
    
    #Normalize data to days
    days = pd.to_datetime(region_weather['Date'])
    days = days.dt.dayofyear
    
    #Add syntetic cos day parameter to mimic seasons
    region_weather['Date'] = np.cos(2*3.1415 * days / 365)
    
    x_region_train, x_region_test, y_region_train, y_region_test = train_test_split(
    region_weather, region_label, 
    test_size=0.2, 
    stratify=region_label, 
    random_state = r_state
    )

    
    
    #Normalize numerical data
    columns = region_weather.columns.tolist()
    for column in columns:
        #Do not normalize date as it is already prepared
        if column == "Date":
            continue
        
        c_data = region_weather[column]
        #Take dtypes of elements in the series column of dataframe
        dtype = c_data.apply(type).value_counts().idxmax()
        #Check if string ---> Categorical
        categorical = (dtype == str)
        
        if not categorical:
            #Normalize data if numerical
            scaler = MinMaxScaler()
            scaler.fit(np.array(x_region_train[column]).reshape(-1,1))
            x_region_train_scaled = scaler.transform(np.array(x_region_train[column]).reshape(-1,1))
            x_region_test_scaled = scaler.transform(np.array(x_region_test[column]).reshape(-1,1))

            #Input the scaled column into the region
            x_region_train[column] = x_region_train_scaled
            x_region_test[column] = x_region_test_scaled
            
        else:
            # One‑hot: połącz train+test tylko tę kolumnę, żeby złapać wszystkie kategorie
            combined = pd.concat(
                [x_region_train[[column]], x_region_test[[column]]],
                axis=0
            )
            # Robimy dummy dla jednej kolumny, wymuszając dtype=int
            dummies = pd.get_dummies(
                combined[column], 
                prefix=column, 
                dtype=int          # <— ensure 0/1 ints, not bools
            )
            # Rozdzielamy z powrotem
            dummies_train = dummies.loc[x_region_train.index]
            dummies_test  = dummies.loc[x_region_test.index]
            # Wstawiamy do DataFrame, usuwając oryginał
            x_region_train = (
                x_region_train.drop(columns=[column])
                              .join(dummies_train)
            )
            x_region_test  = (
                x_region_test .drop(columns=[column])
                              .join(dummies_test)
            )
                    

    #######################
    #Onehot the data
    y_region_train = y_region_train.map({"No": 0, "Yes": 1})   
    y_region_test = y_region_test.map({"No": 0, "Yes": 1})   
    
    #Train the models finally
    y_pred_SKL, y_prob_SKL = SKL_Classifier(x_region_train, y_region_train, x_region_test)
    y_pred_OWN, y_prob_OWN = OWN_Classifier(np.array(x_region_train), np.array(y_region_train), np.array(x_region_test))
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_SKL = Calculate_metrics(y_true = y_region_test,
                                    y_pred = y_pred_SKL,
                                    y_prob = y_prob_SKL,
                                    show_plots = False
                                    )
    
    Metrics_dict_OWN = Calculate_metrics(y_true = y_region_test,
                                    y_pred = y_pred_OWN,
                                    y_prob = y_prob_OWN,
                                    show_plots = False
                                    )
    
    Regions_score_SKL[region] = Metrics_dict_SKL
    Regions_score_OWN[region] = Metrics_dict_OWN
#######################
# %% [markdown]
#Wyswietlenie wynikow

# %%
#Taking best region
best_acc_region   = max(Regions_score_SKL, key=lambda r: Regions_score_SKL[r]['Accuracy'])
best_auc_region   = max(Regions_score_OWN, key=lambda r: Regions_score_OWN[r]['AUC_ROC'])
   
print("---------Scores for SKL regressor classifier-------")
print("Best region by Accuracy is: ",best_acc_region, "with score: accuracy = ",round(Regions_score_SKL[best_acc_region]["Accuracy"],3))
print("Best region by ROC_AUC is: ",best_auc_region, "with score: auc = ",round(Regions_score_SKL[best_auc_region]["Accuracy"],3))
##################
#Porownanie skutecznosci na swoim modelu:
print("---------Scores for OWN regressor classifier-------")
print("Best region by Accuracy is: ",best_acc_region, "with score: accuracy = ",round(Regions_score_OWN[best_acc_region]["Accuracy"],3))
print("Best region by ROC_AUC is: ",best_auc_region, "with score: auc = ",round(Regions_score_OWN[best_auc_region]["Accuracy"],3))

# %% [markdown]
#Na 5

# %%
#Porownanie modeli regionalnych z calym zestawem danych testowych (zlozonych z testowych regionalnych)
print("\n\n")

regions = np.unique(weather_data["Location"]).tolist()
full_x_test = []
full_y_test = []

x_train_dict = {}
y_train_dict = {}
print("Preparing test and train data...")
for region in regions:
    #Taking given region
    binary_region = weather_data['Location'] == region
    region_weather = weather_data[binary_region].copy()
    region_label = label[binary_region].copy()
    
    #Dropping region (as it is not informative feature now)
    region_weather.drop(columns = ['Location'], inplace = True)
    
    #Normalize data to days
    days = pd.to_datetime(region_weather['Date'])
    days = days.dt.dayofyear
    
    #Add syntetic cos day parameter to mimic seasons
    region_weather['Date'] = np.cos(2*3.1415 * days / 365)
    
    x_region_train, x_region_test, y_region_train, y_region_test = train_test_split(
    region_weather, region_label, 
    test_size=0.2, 
    stratify=region_label, 
    random_state = r_state
    )

    
    
    #Append to the train sets
    x_train_dict[region] = x_region_train
    y_train_dict[region] = y_region_train
    
    #Append to full test set
    x_region_test = x_region_test
    y_region_test = np.array(y_region_test)
    full_x_test.append(x_region_test)
    full_y_test.append(y_region_test)
    
#Stack all test data alltogether and adjust it to df
full_x_test = np.vstack(full_x_test)
full_y_test = np.hstack(full_y_test)

#X
first_key = next(iter(x_train_dict))
columns = x_train_dict[first_key].columns.tolist()
full_x_test_original = pd.DataFrame(full_x_test, columns = columns )

#Y
full_y_test = pd.Series(full_y_test).map({"No": 0, "Yes": 1})   




# %%
#Performing all normalization which has been done in point for 4
Regions_score_SKL = {}
print("Training models on local train data and evaluating on global data...")
for key in x_train_dict.keys():
    x_region_train = x_train_dict[key]
    y_region_train = y_train_dict[key]
    full_x_test = full_x_test_original.copy(deep = True)
    #Normalize numerical data
    columns = x_region_train.columns.tolist()
    for column in columns:
        #Do not normalize date as it is already prepared
        if column == "Date":
            continue
        
        c_data = x_region_train[column]
        #Take dtypes of elements in the series column of dataframe
        dtype = c_data.apply(type).value_counts().idxmax()
        #Check if string ---> Categorical
        categorical = (dtype == str)

        if not categorical:
            #Normalize data if numerical
            scaler = MinMaxScaler()
            scaler.fit(np.array(x_region_train[column]).reshape(-1,1))
            x_region_train_scaled = scaler.transform(np.array(x_region_train[column]).reshape(-1,1))
            x_region_test_scaled = scaler.transform(np.array(full_x_test[column]).reshape(-1,1))

            #Input the scaled column into the region
            x_region_train[column] = x_region_train_scaled
            full_x_test[column] = x_region_test_scaled
            
        else:
            #GPT PART Memory lekeage Ktorego nie potrafilem zalatac 
            ###############################
            # 1) Compute the union of categories once
            cats = sorted(
                set(x_region_train[column].dropna().unique()) |
                set(full_x_test   [column].dropna().unique())
            )
            
            # 2) Cast both to Categorical with the same categories
            x_region_train[column] = pd.Categorical(x_region_train[column], categories=cats)
            full_x_test   [column] = pd.Categorical(full_x_test   [column], categories=cats)
            
            # 3) One-hot them separately but with identical columns
            d_train = pd.get_dummies(x_region_train[column], prefix=column, dtype=int)
            d_test  = pd.get_dummies(full_x_test   [column], prefix=column, dtype=int)
            
            # 4) Drop original and join new
            x_region_train = x_region_train.drop(columns=[column]).join(d_train)
            full_x_test    = full_x_test   .drop(columns=[column]).join(d_test)
            ##################################################
                    

    #######################
    #Onehot the data
    y_region_train = y_region_train.map({"No": 0, "Yes": 1})   
    
    #Train the models finally
    y_pred_SKL, y_prob_SKL = SKL_Classifier(x_region_train, y_region_train, full_x_test)
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_SKL = Calculate_metrics(y_true = full_y_test,
                                    y_pred = y_pred_SKL,
                                    y_prob = y_prob_SKL,
                                    show_plots = False
                                    )
    
    Regions_score_SKL[key] = Metrics_dict_SKL


    


    

# %%
#Taking best region
best_acc_region   = max(Regions_score_SKL, key=lambda r: Regions_score_SKL[r]['Accuracy'])
best_auc_region   = max(Regions_score_OWN, key=lambda r: Regions_score_OWN[r]['AUC_ROC'])
   
print("---------Scores for SKL regressor classifier (GLOBAL TEST)-------")
print("Best regional classifier by Accuracy is: ",best_acc_region, "with score: accuracy = ",round(Regions_score_SKL[best_acc_region]["Accuracy"],3))
print("Best regional classifier by ROC_AUC is: ",best_auc_region, "with score: auc = ",round(Regions_score_SKL[best_auc_region]["Accuracy"],3))
##################
# %% [markdown]
#Nejlepsze modele lokalne mialy o wiele wieksza dokladnosc dla regionow dla ktorych zostaly wytrenowane
#W porownaniu do wszystkich skumulowanych regionow

#Najlepszy model lokalny sprawdzony na globalnych danych "Wollongong" 
#roznil sie od tego majacego najlepszy wynik lokalnie "Woomera"
#Choc modele majace najlepszy ROC_AUC reprezentowaly ten sam region: "Perth"
# %%
#Porownanie z dummy
print("\n\n -----------------------Dummy - SKL classifier comparision--------------------")
for key in ["Perth"]:
    x_region_train = x_train_dict[key]
    y_region_train = y_train_dict[key]
    full_x_test = full_x_test_original.copy(deep = True)
    #Normalize numerical data
    columns = x_region_train.columns.tolist()
    for column in columns:
        #Do not normalize date as it is already prepared
        if column == "Date":
            continue
        
        c_data = x_region_train[column]
        #Take dtypes of elements in the series column of dataframe
        dtype = c_data.apply(type).value_counts().idxmax()
        #Check if string ---> Categorical
        categorical = (dtype == str)

        if not categorical:
            #Normalize data if numerical
            scaler = MinMaxScaler()
            scaler.fit(np.array(x_region_train[column]).reshape(-1,1))
            x_region_train_scaled = scaler.transform(np.array(x_region_train[column]).reshape(-1,1))
            x_region_test_scaled = scaler.transform(np.array(full_x_test[column]).reshape(-1,1))

            #Input the scaled column into the region
            x_region_train[column] = x_region_train_scaled
            full_x_test[column] = x_region_test_scaled
            
        else:
            #GPT PART Memory lekeage Ktorego nie potrafilem zalatac 
            ###############################
            # 1) Compute the union of categories once
            cats = sorted(
                set(x_region_train[column].dropna().unique()) |
                set(full_x_test   [column].dropna().unique())
            )
            
            # 2) Cast both to Categorical with the same categories
            x_region_train[column] = pd.Categorical(x_region_train[column], categories=cats)
            full_x_test   [column] = pd.Categorical(full_x_test   [column], categories=cats)
            
            # 3) One-hot them separately but with identical columns
            d_train = pd.get_dummies(x_region_train[column], prefix=column, dtype=int)
            d_test  = pd.get_dummies(full_x_test   [column], prefix=column, dtype=int)
            
            # 4) Drop original and join new
            x_region_train = x_region_train.drop(columns=[column]).join(d_train)
            full_x_test    = full_x_test   .drop(columns=[column]).join(d_test)
            ##################################################
                    
    #######################
    #Onehot the data
    y_region_train = y_region_train.map({"No": 0, "Yes": 1})   
    #Train the models finally
    y_pred_SKL, y_prob_SKL = SKL_Classifier(x_region_train, y_region_train, full_x_test)
    #Zwrocenie metryk oraz macierzy pomylek z wykresami krzywej operacyjnej
    Metrics_dict_SKL = Calculate_metrics(y_true = full_y_test,
                                    y_pred = y_pred_SKL,
                                    y_prob = y_prob_SKL,
                                    show_plots = False
                                    )
    #Dla dummy
    dummy_clf = skl.dummy.DummyClassifier(strategy="most_frequent")

    dummy_clf.fit(x_region_train, y_region_train)
    dummy_clf.predict(full_x_test)
    
    print("SKL accuracy: ", Metrics_dict_SKL["Accuracy"])
    print("SKL auc_ROC: ", Metrics_dict_SKL["Accuracy"])
    
    print("DUMMY Classifier acc: ",dummy_clf.score(full_x_test, full_y_test))

# %% [markdown]
#Nie jest to jak widać zawsze najlepsza metoda. Może dawać dobre z pozoru wyniki jednak
#W przypadku danych z australii gdzie wiekszosc dni nie jest deszczowa (zbior niezbalansowany)
#klasyfikator wiekszosciowy moze dawac podobne wyniki
#Recall w tym przypadku byl na dosc niskim poziomie co oznacza ze model czesciej przewidywal
# nawet na "slepo", mial bias przesuniety w strone dni niedeszczowych, przewidywania 0



























