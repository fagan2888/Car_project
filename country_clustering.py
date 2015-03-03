import pandas as pd
import numpy as np
from itertools import product, chain
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import pickle as pkl
from scipy import sparse

file = open('/Users/LaughingMan/Desktop/Data/Cars/cleaned_car_data.pkl', 'rb')
cars = pkl.load(file)
file.close()

origin = pd.get_dummies(cars['customer_country'])
origin.columns = ['o'+ x for x in origin.columns]
dest = pd.get_dummies(cars['destination'])
dest.columns = ['d'+ x for x in dest.columns]

pairwise_names = []
for o in origin.columns:
    for d in dest.columns:
        pairwise_names.append(o+d)

pairwise_matrix = sparse.csc_matrix((96341,1))
for o in origin.columns:
    for d in dest.columns:
        pairwise_matrix = sparse.hstack([pairwise_matrix, sparse.csr_matrix(origin[o]*dest[d]).T])
pairwise_matrix = pairwise_matrix[:,1:]

colnames = origin.columns + dest.columns + pairwise_names

origin_sparse = sparse.csc_matrix(origin)
dest_sparse = sparse.csc_matrix(dest)

all_dummies = sparse.hstack([origin_sparse,dest_sparse,pairwise_matrix])

features = pd.DataFrame()
features = pd.concat([features,pd.get_dummies(cars['customer_country']), axis=1)

pd.get_dummies(book['customer_country'])
