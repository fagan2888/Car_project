import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle as pkl
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit
from scipy.stats import gamma
import json
import pdb

'''
README:
The CustomerCluster object takes in a sipp code decomposition and a dataset of
origin/destination countries. Note that the rows of the sipp code decomposition
need to correspond with the origin/destination rows. The CustomerCluster
operates in the following order:
1) load() : load the datasets
2) binary_pca_clustering() : does a PCA on the sipp code decomposition,
                             extracts similar cars clusters via KMeans
3) dummy_matrix() : binarizes the origin/destination dataset and creates all
                    possible interaction terms.
4) create_nb_priors() : Optional. Creates a prior distribution for the Naieve
                        Bayes that spreads the likelyhood of each car class.
5) fit_nb_cartypes() : creates a probability ranking of the kinds of cars a
                       customer will buy depending on the origin_destion dummy
                       matrix.
6) fit_kmeans_customer_types() : creates clusters of different customer types
                                 depending on their probability ranking from
                                 step 4.
'''

class CustomerCluster(object):
    def __init__(self):
        self.cars = None
        self.binarized_sipp = None
        self.country_dummies = None
        self.country_dummies_colnames = None
        self.nb_confusion_mat = None
        self.km_confusion_mat = None
        self.car_preds = None
        self.new_priors = None
        self.pca = None
        self.orig_dest = None

    def load(self,
             car_data = '/Users/LaughingMan/Desktop/Data/Cars/cleaned_car_data.pkl',
             sipp_decomp = '/Users/LaughingMan/Desktop/Data/Cars/sipp_decomp.pkl'):
        '''
        INPUT: file paths for the origin/destination data (i.e. car_data) and
               sipp code decomposition (i.e. sipp_decomp)
        OUTPUT: None.

        This just reads the datasets into the object.
        '''
        file = open(car_data, 'rb')
        self.cars = pkl.load(file)
        file.close()

        file = open(sipp_decomp, 'rb')
        self.binarized_sipp = pkl.load(file)
        file.close()

    def binary_pca_clustering(self,pca_components=15,kmean_clusters=15):
        '''
        INPUT: number of components for pca, number of clusters for kmeans
        OUTPUT: None.

        This runs a PCA on the decomposed sipp codes, and then does a KMeans
        clustering to find cars with similar properties. It then assigns those
        car clusters to customers in the origin_destion dataframe.
        '''
        self.pca = PCA(n_components = pca_components)
        target_pca = self.pca.fit_transform(self.binarized_sipp)
        kmean = KMeans(n_clusters = kmean_clusters)
        kmean.fit(target_pca)
        clusters = kmean.predict(target_pca)
        self.cars['cluster'] = clusters

    def dummy_matrix(self):
        '''
        INPUT: None
        OUTPUT: None

        This takes the origin_destion data and turns it into an enormous sparse
        matrix that has dummy variables for each country indicating where the
        person is coming from and where they are going. It also includes all
        pairwise mulitplicaations of the origin/distination dummy variables.
        '''
        origin = pd.get_dummies(self.cars['customer_country'])
        origin.columns = ['o'+ x for x in origin.columns]
        dest = pd.get_dummies(self.cars['destination'])
        dest.columns = ['d'+ x for x in dest.columns]

        self.cars['origin_dest'] = \
            self.cars['customer_country'].apply(lambda x: 'o' + str(x)) + \
            self.cars['destination'].apply(lambda x: 'd' + str(x))

        pairwise_names = []
        for o in origin.columns:
            for d in dest.columns:
                pairwise_names.append(o+d)

        pairwise_matrix = sparse.csc_matrix((96341,1))
        for o in origin.columns:
            for d in dest.columns:
                pairwise_matrix = sparse.hstack([pairwise_matrix,
                    sparse.csr_matrix(origin[o]*dest[d]).T])
        pairwise_matrix = pairwise_matrix[:,1:]

        self.country_dummies_colnames = list(origin.columns) + \
                                        list(dest.columns) + \
                                        pairwise_names

        origin_sparse = sparse.csc_matrix(origin)
        dest_sparse = sparse.csc_matrix(dest)

        self.country_dummies = sparse.hstack([origin_sparse,
                                              dest_sparse,
                                              pairwise_matrix])

    def create_nb_priors(self,flatten = .15):
        '''
        INPUT: flatten parameter.
        OUTPUT: None.

        This fits the prior distribution of car selections to a gaussian
        distribtuion it then creates a new series of priors by flattening the
        distribution. These priors are then used in the Naieve Bayes predict
        proba.
        '''
        prefs = self.cars['cluster'].value_counts()/ \
                sum(self.cars['cluster'].value_counts())
        vals = np.array([x+1 for x in range(len(prefs))])
        params, pcov = curve_fit(tg, vals, prefs)
        new_priors = tg(vals, params[0]+flatten, params[1], params[2])
        self.new_priors = new_priors/sum(new_priors)

    def fit_nb_cartypes(self, use_weights=False):
        '''
        INPUT: None
        OUTPUT: None

        This runs a naieve bayes algorithim on the country_dummies matrix to
        predict the kind of car cluster that that individual preffers. It then
        creates a probability ranking over clusters for each individual in the
        dataset.
        '''
        if use_weights:
            self.nb = BernoulliNB(class_prior = self.new_priors)
        else:
            self.nb = BernoulliNB()

        X_train, X_test, y_train, y_test = \
            train_test_split(self.country_dummies, self.cars['cluster'],
                             test_size=0.2, random_state=42)
        self.nb.fit(X_train, y_train)
        pred = self.nb.predict(X_test)
        self.nb_confusion_mat = confusion_matrix(y_test, pred)

        self.nb.fit(self.country_dummies, self.cars['cluster'])
        self.car_preds = self.nb.predict_proba(self.country_dummies)

    def fit_kmeans_customer_types(self, clusters = 20):
        '''
        INPUT: integer, representing the number of customer types
        OUTPUT: None

        This uses KMeans on the prefference rankings in order to identify
        customer clusters with similar prefferences.
        '''
        X_train, X_test, y_train, y_test = \
            train_test_split(self.car_preds, self.cars['cluster'],
                             test_size=0.2, random_state=42)
        self.nb.fit(X_train, y_train)
        pred = self.nb.predict(X_test)
        self.km_confusion_mat = confusion_matrix(y_test, pred)

        km = KMeans(n_clusters = clusters)
        km.fit(self.car_preds)
        self.cars['car_pref_cluster'] = km.predict(self.car_preds)

    def _present_car_cluster(self,i):
        '''
        INPUT: integer, representing a car cluster
        OUTPUT: None.

        Print top 10 most common cars in the specified cluster.
        '''
        print list(self.cars[self.cars['cluster'] == i]\
                ['car_name'].value_counts()[0:9].index)
        print list(self.cars[self.cars['cluster'] == i]
                ['sipp_code'].value_counts()[0:9].index)

    def _present_customer_cluster(self, i):
        '''
        INPUT: integer, representing a customer cluster
        OUTPUT: None

        Print top 6 most common cars for each car cluster in the order
        that the car clusters are preffered as well as the top 5 associated
        origin/destination pairs for that customer cluster.
        '''
        indices = self.cars[self.cars['car_pref_cluster'] == i]\
            ['cluster'].value_counts()[0:6].index
        top_cars = []
        for k in indices:
            top_cars.append((list(self.cars[self.cars['cluster'] == k]\
                ['car_name'].value_counts()[0:5].index),k))

        print 'origin/destination pairs: ' + \
              str(list(self.cars[self.cars['car_pref_cluster'] == i]\
                  ['origin_dest'].value_counts()[0:5].index))
        for j in top_cars:
            print 'car cluster: ' + str(j[1])
            print 'top cars in cluster: ' + str(j[0])

    def _create_orig_dest_dict(self):
        '''
        INPUT: None
        OUTPUT: writes out a javascript hashable array for use in d3

        This will create an origin/destination dict, i.e. for each country
        there is an entry, and each entry is itself a dict that associates
        all incoming traffic to the original country with sub countries and
        their clusters. E.g.
        {US: {SE:1, GB:2}}
        Means that people coming to the US from SE are in cluster 1 and people
        coming to US from GB are in cluster 2.
        '''
        d = {}
        origin = np.unique(self.cars['customer_country'])
        origin = [x for x in origin if str(x) != 'nan']
        destination = np.unique(self.cars['destination'])
        destination = [x for x in destination if str(x) != 'nan']

        for o in origin:
            d[o] = {}
            cond_o = self.cars['customer_country'] == o
            for d in destination:
                cond_d = self.cars['destination'] == d
                if self.cars[cond_o & cond_d].shape[0] > 0:
                    print o, d
                    d[o][d] = self.cars[cond_o & cond_d]['car_pref_cluster'].value_counts()
        self.orig_dest = d


#This is just a function for returning the pdf of a gamma distriubtuion.
def tg(x, a, b, c ):
    return gamma(a, loc = b, scale = c).pdf(x)

if __name__ == '__main__':
    cc = CustomerCluster()
    cc.load()
    cc.binary_pca_clustering(pca_components=10,kmean_clusters=10)
    cc.dummy_matrix()
    cc.create_nb_priors(flatten=1)
    cc.fit_nb_cartypes(use_weights=True)
    cc.fit_kmeans_customer_types(clusters=10)
    cc._create_orig_dest_dict()
