import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib
import seaborn as sns
import webcolors
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from collections import Counter



""" 
------------------------------------------------------------------------------------------------------------------------------------        
     Function to use KnnImputer on a dataset
------------------------------------------------------------------------------------------------------------------------------------        
"""

def encode_data_keep_nan(train: pd.DataFrame, test: pd.DataFrame):
    """
    Label-encodes categorical columns in the DataFrame while preserving NaN values.
    
    Args:
        train (pd.DataFrame): 
            DataFrame with columns to encode.
        
    Returns:
        pd.DataFrame: 
            A DataFrame with label-encoded categorical columns.
    """
    print("[INFO] : labelize data...")
    train_encoded = train.copy()  # Make a copy to avoid modifying the original
    test_encoded = test.copy()  # Make a copy to avoid modifying the original
    
    # Loop through each column in the DataFrame
    for col in train_encoded.select_dtypes(include=['object', 'category']).columns:
        
        print("labelize " + col)
        le = LabelEncoder()
        
        # Fit LabelEncoder only on non-null values to preserve NaN
        non_null_train = train_encoded[col].dropna()
        le.fit(non_null_train)
        
        # Transform with LabelEncoder, and leave NaN as NaN
        train_encoded[col] = train_encoded[col].apply(lambda x: le.transform([x])[0] if pd.notna(x) else x)
        test_encoded[col] = test_encoded[col].apply(lambda x: le.transform([x])[0] if pd.notna(x) else x)
        
    print("[INFO] : end labelize ...")
    print()
    
    print("[INFO] : KNNImputer process...")
    imputer = KNNImputer()
    train_encoded = pd.DataFrame(imputer.fit_transform(train_encoded), columns = [i for i in train_encoded])
    test_encoded = pd.DataFrame(imputer.transform(test_encoded), columns = [i for i in test_encoded])
    print("[INFO] : end KNNImputer process...")
    print()
    
    return train_encoded, test_encoded


# ------------------------------------------------------------------------------------------------------------------------------------        
# Fonction pour remplacer les valeurs manquantes du dataset de manière automatique (moyenne pour float et 
# valeur majoritaire pour str)
# ------------------------------------------------------------------------------------------------------------------------------------        

    
import pandas as pd
import logging

# Configure logging for optional detailed output
logging.basicConfig(level=logging.INFO)

def fillna_process(data: pd.DataFrame, details: bool = False) -> pd.DataFrame:
    """
    Automatically fills missing values in a DataFrame based on data type.
    
    For columns with missing values:
        - Fills numerical (float) columns with the mean.
        - Fills categorical (string/object) columns with the most common value (mode).
    
    Args:
        data (pd.DataFrame): The DataFrame with missing values to process.
        details (bool): If True, logs detailed information about the filling process.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled.
    """
    # Set logging level based on the `details` parameter
    if details:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Iterate over columns with missing values only
    for column in data.columns[data.isna().any()]:
        if data[column].dtype == 'object':
            # Fill categorical columns with the most common value (mode)
            most_common = data[column].mode()[0]
            data[column].fillna(most_common, inplace=True)
            logging.info(f"Column '{column}' (categorical): filled with most common value '{most_common}'.")
            
        else:
            # Fill numerical columns with the mean
            mean_value = data[column].mean()
            data[column].fillna(mean_value, inplace=True)
            logging.info(f"Column '{column}' (numerical): filled with mean value '{mean_value:.2f}'.")

    return data

    
    
"""
Functions I used for the Flask project, but you can re use for KNN search
"""


### pour faire la recherche des voisins, on réduit le dataset labelisé (lb) aux labels sélectionnés (X)
### puis on cherche les voisins des valeurs need qui correspondent aux labels des choix de l'utilisateurs de la page web
def get_knn_similarities(df: pd.DataFrame,
                            nb_similarities: int,
                            reference_case_encoded: list,
                            List_index: list,
                            algorithm: str = "brute",
                            metric: str = "euclidean"):
    """
        This fonction takes the critere of search on the dataset, and use KNN
        to return the line of the dataset which corresponde to the request

        Args:
            df_labelize (pd.DataFrame):
                the labelized dataframe where we look for similar case
                
            nb_similarities (int):
                the number of similar cases we want
                
            reference_case_encoded (Dict):
                dict of columns we search similarities with it encoded value corresponding
                
            List_index (list):
                the list of index were we could find the similar case
            
        Returns:
            sol(list):
                list of index corresponding to similar case
    """
    # we initiate the KNN with the labelize dataset with only the columns corresponding to the choices of the user
    knn = NearestNeighbors(algorithm = "brute", metric = "euclidean")
    
    # we get only the column sected for the similarities
    df_target = df[reference_case_encoded.keys()]
        
    # we labelize both dataset
    encode = LabelEncoder()
    for col in df_target:
        df_target[col] = encode.fit_transform(df_target[col])
        reference_case_encoded[col] = encode.transform(reference_case_encoded[col])
            
    # find similarities
    knn = knn.fit(df_target)
    
    
    # the most similar line in the dataset
    distance, indices = list(knn.kneighbors(np.array(reference_case_encoded.values), nb_similarities, return_distance = True))
    sol = [List_index[indice] for indice in indices[0]]
        
    return sol

def get_search_by_knn(reference_case, nb_similarities, df, List_index):
    """
        This algo takes the input selected by the user on the web site, and return the id of the images that correspond to the input (request)
        
        Args:
            reference_case (Dict):
                the case we want to find similarities, with columns and value corresponding
                
            nb_similarities (int):
                the number of similar case the function will returns
                
            df (pd.DataFrame):
                the non labelize DataFrame without id column
                
            List_index (list):
                list of the index of the dataframe df
            
        returns:
            sol:
                list of index corresponding to similar case
    
    """

    # check the non empty values selected by the user
    reference_case_encoded = {}
    for key in reference_case.keys():
        # we check that there is a value selected
        if reference_case[key] != '':
            reference_case_encoded[key] = reference_case[key]
            
    # we convert into a dataframe
    reference_case_encoded = pd.DataFrame([reference_case_encoded])
    
    print("search of similare case to:", reference_case_encoded)
    # encode user choice
        
    sol = get_knn_similarities(df, int(nb_similarities), reference_case_encoded, List_index)

    return sol

""" ---------------------------------------------------------------------------------------------------------------------------------------
Functions I used for generating dataset of colthes images description
-------------------------------------------------------------------------------------------------------------------------------------------
"""

# extract dominant color

def get_dominant_colors(image, k=1):
    """This function take an image and return the rgb of the dominant color

    Args:
        image (np.arry): the image we want the dominant color
        k (int): the number of similar color we want

    Returns:
        dominant_color: rgb list (R, G, B)
    """
    # preprocess image
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # get the similar color
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    # Compter la fréquence des étiquettes de cluster
    label_counts = Counter(labels)

    # Trier les clusters par fréquence décroissante
    sorted_clusters = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)

    # Obtenir les couleurs dominantes à partir des clusters les plus fréquents
    dominant_colors = kmeans.cluster_centers_[sorted_clusters[:k]].astype(int)

    # return dataframe
    
    return pd.DataFrame(dominant_colors, columns=["R", "G", "B"])

def get_color(rgb, colors_label, knn):
    """
    This fonction return the most similar color giving a list of RGB values and the list 
    of colors names

    Args:
        rgb (list): list of RGB values for the image in process
        colors_label (list): list of colors names in the dataset

    Returns:
        str: name of the similar color
    """
    
    indices = list(knn.kneighbors(np.array(rgb.values), 1, return_distance = False))
    return colors_label[indices[0][0]]

def main_get_color(image):
    """
    This function take an image and a dataset of color (cols= 'color', 'R', 'G', 'B'), 
    and return the name of the dominant color on the picture.

    Args:
        image (np.array): the image we want the dominant color
        df_color (pd.DataFrame): the datafram which contains the color with RGB values associated

    Returns:
        str: name of the dominant color
    """
    
    # import df of colors 
    df_color= pd.read_csv("color_dataframe.csv")
    colors_label = df_color['color'].tolist()
    df_color = df_color.drop(['color'], axis=1)

    # train KNN for the dominant color
    knn = NearestNeighbors(algorithm = "brute", metric = "euclidean")
    knn = knn.fit(df_color)
    
    # we search the dominant color
    dominant_colors = get_dominant_colors(image, k=1)
    # return the name of the most similar color contains in the color dataset
    return get_color(dominant_colors, colors_label, knn)

def see_color(rgb):
    """
    Display the color giving a list (R, G, B)

    Args:
        rgb (list): RGB list
        
    Return:
        None
    """
    
    color = rgb
    image = [[color for _ in range(100)] for _ in range(100)]

    # Afficher l'image
    plt.imshow(image)
    plt.axis('off')  # Masquer les axes
    plt.show()
    
    
# utilities for spotify

