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

class CustomerCluster(object):

    def __init__(self,car_data,sipp_decomp):
        self.cars = None
        self.binarized_sipp = None
        self.country_dummies = None
        self.country_dummies_colnames = None
        self.nb_confusion_mat = None
        self.km_confusion_mat = None
        self.car_preds = None

    def load(self,
             car_data = '/Users/LaughingMan/Desktop/Data/Cars/cleaned_car_data.pkl',
             sipp_decomp = '/Users/LaughingMan/Desktop/Data/Cars/cleaned_car_data.pkl'):
        file = open(car_data, 'rb')
        self.cars = pkl.load(file)
        file.close()

        file = open(sipp_decomp, 'rb')
        self.binarized_sipp = pkl.load(file)
        file.close()

    def binary_pca_clustering(self,pca_components=15,kmean_clusters=15):
        pca = PCA(n_components = pca_components) #this captures about 90% of the variance.
        target_pca = pca.fit_transform(self.binarized_sipp)
        kmean = KMeans(n_clusters = kmean_clusters)
        kmean.fit(target_pca)
        clusters = kmean.predict(target_pca)
        self.cars['cluster'] = clusters

    def dummy_matrix(self):
        origin = pd.get_dummies(self.cars['customer_country'])
        origin.columns = ['o'+ x for x in origin.columns]
        dest = pd.get_dummies(self.cars['destination'])
        dest.columns = ['d'+ x for x in dest.columns]

        self.cars['origin_dest'] = self.cars['customer_country'].apply(lambda x: 'o' + str(x)) + \
                                   self.cars['destination'].apply(lambda x: 'd' + str(x))

        pairwise_names = []
        for o in origin.columns:
            for d in dest.columns:
                pairwise_names.append(o+d)

        pairwise_matrix = sparse.csc_matrix((96341,1))
        for o in origin.columns:
            for d in dest.columns:
                pairwise_matrix = sparse.hstack([pairwise_matrix, sparse.csr_matrix(origin[o]*dest[d]).T])
        pairwise_matrix = pairwise_matrix[:,1:]

        self.country_dummies_colnames = list(origin.columns) + list(dest.columns) + pairwise_names

        origin_sparse = sparse.csc_matrix(origin)
        dest_sparse = sparse.csc_matrix(dest)

        self.country_dummies = sparse.hstack([origin_sparse,dest_sparse,pairwise_matrix])

    def fit_nb_cartypes(self):
        self.nb = BernoulliNB()
        X_train, X_test, y_train, y_test = train_test_split(self.country_dummies,
                                                            self.cars['cluster'],
                                                            test_size=0.2,
                                                            random_state=42)
        self.nb.fit(X_train,y_train)
        pred = self.nb.predict(X_test)
        self.nb_confusion_mat = confusion_matrix(y_test, pred)

        self.nb.fit(all_dummies, self.cars['cluster'])
        self.car_preds = self.nb.predict_proba(all_dummies)

    def fit_kmeans_customer_types(self, clusters = 20):
        X_train, X_test, y_train, y_test = train_test_split(self.car_preds,
                                                            self.cars['cluster'],
                                                            test_size=0.2,
                                                            random_state=42)
        self.nb.fit(X_train,y_train)
        pred = self.nb.predict(X_test)
        self.km_confusion_mat = confusion_matrix(y_test,pred)

        km = KMeans(n_clusters = clusters)
        km.fit(self.car_preds)
        self.cars['car_pref_cluster'] = km.predict(self.car_preds)


# file = open('/Users/LaughingMan/Desktop/Data/Cars/customer_p_data.pkl', 'wb')
# cars = pkl.dump(file)
# file.close()

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

# ohc = OneHotEncoder()
# y = ohc.fit_transform(cars['cluster'].apply(lambda x : str(x)))
