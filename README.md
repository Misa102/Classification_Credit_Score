# Classification_Credit_Score

## Introduction
Dans ce projet, nous explorons un ensemble de données sur les scores de crédit dans le but de comprendre les relations entre différentes variables et de préparer les données pour la modélisation des scores de crédit.

### Importation de bibliothèques
    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
    import warnings, re, joblib
    warnings.filterwarnings("ignore")
    from scipy.stats import probplot

## Partie 1: Analyse exploratoire des données

### Aperçu statistique de l'ensemble de données

#### Lecture des données
    df = pd.read_csv("train.csv")
    df.head()
#### Dimensions de l'ensemble de données
    df.shape
#### Affichage des détails statistiques de base
    df.describe()
#### Compter les valeurs manquantes
    df.isna().sum()
#### Vérification des doublons
    df.duplicate().sum()
#### Affichage alternatif des informations

     def columns_info (df):
        columns=[]
        dtypes=[]
        unique=[]
        nunique=[]
        nulls=[]
        
        for cols in df.columns:
            columns.append(cols)
            dtypes.append(df[cols].dtypes)
            unique.append(df[cols].unique())
            nunique.append(df[cols].nunique())
            nulls.append(df[cols].isna().sum())
        
        return pd.DataFrame({'Columns': columns,
                             'Data Types': dtypes,
                             'Unique Values': unique,
                             'Number of unique': nunique,
                             'Missing Values': nulls
                            })
    columns_info(df)


## Partie 2 : Ingénierie des fonctionnalités
### Gestion des valeurs aberrantes et nettoyage des données
#### Fonction de valeurs aberrantes
    def check_outliers(col, df):
        col_data= df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr= q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3+1.5*iqr
        outliers = []
    
        #outliers = col_data[(col_data<lower_bound)|(col_data>upper_bound)]
        
        for i in range(len(df)):
            value = df.loc[i, col]
            if value > upper_bound or value < lower_bound:
                outliers.append(value)
        
        return outliers
#### Fonction de gestion des valeurs aberrantes
      def handle_outliers(col, df):
        col_data= df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr= q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3+1.5*iqr
        outliers = []
    
        # Remplacer les valeurs aberrantes par les bornes
        #df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
        for i in range(len(df)):
            if df.loc[i, col] > upper_bound:
                df.loc[i, col] = upper_bound
            elif df.loc[i, col] < lower_bound:
                df.loc[i, col] = lower_bound
            
### Nettoyage des données et traitement des valeurs aberrantes

#### Nettoyage 'Month'
    df['Month'] = df['Month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8})
    df['Month'].unique()

#### Cleaning 'Age'
    df['Age'] = df['Age'].str.replace('-', '')
    df['Age'] = df['Age'].str.replace('_', '')
    df['Age'] = df['Age'].astype(int)
    #### Nettoyage 'Occupation'
    df['Occupation'] = df['Occupation'].replace('_______', 'FreeLancer')
    df['Occupation'].unique()
... (Continuer le nettoyage pour les autres colonnes)

## Partie 3 : Apprentissage automatique

### Classificateur de forêt aléatoire

#### Modélisation et évaluation
    from sklearn.ensemble import RandomForestClassifier
    
    X = df.drop('Credit_Score', axis=1).values
    y = df['Credit_Score'].values
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    pd.DataFrame({'Random Forest Classifier (Test)': evaluate_model(X_test, y_test, model),
                  'Random Forest Classifier (Train)': evaluate_model(X_train, y_train, model)
                 })
             
### K-Voisins les plus proches (KNN)

#### Modélisation et évaluation
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    
    X = df.drop('Credit_Score', axis=1).values
    y = df['Credit_Score'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model2 = KNeighborsClassifier(n_neighbors=5)
    model2.fit(X_train, y_train)
    
    pd.DataFrame({'KNN (Test)': evaluate_model(X_test, y_test, model2),
                  'KNN (Train)': evaluate_model(X_train, y_train, model2)
                 })
             
## Conclusion

### Classificateur de forêt aléatoire

Le Random Forest Classifier a atteint une précision d’environ 77,8 % sur l’ensemble de tests. Les performances du modèle sur l'ensemble de test sont légèrement inférieures à celles sur l'ensemble d'entraînement, ce qui suggère un potentiel surajustement des données d'entraînement. Le rappel est d'environ 75,6 %, la précision de 76,4 % et le score F1 d'environ 76 %, ce qui indique une mesure équilibrée entre précision et rappel.

### K-Nearest Neighbors (KNN)
Le modèle K-Nearest Neighbours (KNN) a atteint une précision d’environ 73,2 % sur l’ensemble de test. Semblable au Random Forest Classifier, le modèle fonctionne légèrement moins bien sur l'ensemble de test que sur l'ensemble d'entraînement, ce qui indique un certain surajustement. Le rappel est d'environ 71,5 %, la précision de 72,2 % et le score F1 d'environ 71,8 %.

En résumé, les deux modèles affichent des performances décentes en matière de classification des cotes de crédit. Un réglage supplémentaire des hyperparamètres et une ingénierie des fonctionnalités pourraient potentiellement améliorer les performances du modèle.
