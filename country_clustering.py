import pandas as pd
import numpy as np
from itertools import product, chain
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import pickle as pkl
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import NMF
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

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

X_train, X_test, y_train, y_test = train_test_split(all_dummies, cars['cluster'], test_size=0.2, random_state=42)
nb.fit(X_train,y_train)
pred = nb.predict(X_test)
cm = confusion_matrix(y_test,pred)

nb.fit(all_dummies,cars['cluster'])
car_preds = nb.predict_proba(all_dummies)

pca = PCA(n_components=20)
reduced_pclass = pca.fit_transform(car_preds.T)

nmf = NMF(n_components=20)
reduced_pclass = nmf.fit_transform(car_preds.T)

U, s, V = np.linalg.svd(car_preds.T, full_matrices=False)


# dmat = squareform(pdist(car_preds,'euclidean'))
# dmat = np.nan_to_num(dmat)
# links = linkage(dmat, method = 'complete')
# plt.figure(num=None, figsize=(20, 9), dpi=80, facecolor='w', edgecolor='k')
# dend = dendrogram(links, color_threshold = 2.8)

# normalized_pclass = normalize(car_preds.T)
# tsvd = TruncatedSVD(n_components = 10, n_iter =10)
# reduced_pclass = tsvd.fit_transform(normalized_pclass)



# kf = KFold(n=96341, n_folds=5, shuffle=False, random_state=None)
# nb = BernoulliNB()
# for train_index, test_index in kf:
#     X_train, X_test = all_dummies[train_index,:], all_dummies[test_index,:]
#     y_train, y_test = cars['cluster'][train_index], cars['cluster'][test_index]
#     nb.fit(X_train,y_train)
#     pred = nb.predict(X_test)
#     print confusion_matrix(y_test,ypred)

ohc = OneHotEncoder()
y = ohc.fit_transform(cars['cluster'].apply(lambda x : str(x)))
