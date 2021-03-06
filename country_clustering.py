import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle as pkl
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from scipy.optimize import curve_fit
from scipy.stats import gamma
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json

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

Other methods also exist, mostly to describe the outputs or evaluate the
results of this process. See each method description for further details.
'''


class CustomerCluster(object):
    def __init__(self):
        self.cars = None
        self.binarized_sipp = None
        self.country_dummies = None
        self.country_dummies_colnames = None
        self.nb_confusion = None
        self.nb_accuracy = None
        self.nb_reports = None
        self.car_preds = None
        self.new_priors = None
        self.pca = None
        self.path = '/Users/LaughingMan/Desktop/zipfian/zipfian_project/'
        self.baseline = None
        self.nb_top2_accuracy = None

    def load(self, car_data=None, sipp_decomp=None):
        '''
        INPUT: file paths for the origin/destination data (i.e. car_data) and
               sipp code decomposition (i.e. sipp_decomp)
        OUTPUT: None.

        This just reads the datasets into the object.
        '''
        if car_data:
            pass
        else:
            car_data = '/Users/LaughingMan/Desktop/Data/Cars' \
                       '/cleaned_car_data.pkl'
        if sipp_decomp:
            pass
        else:
            sipp_decomp = '/Users/LaughingMan/Desktop/Data/Cars' \
                          '/sipp_decomp.pkl'

        file = open(car_data, 'rb')
        self.cars = pkl.load(file)
        file.close()

        file = open(sipp_decomp, 'rb')
        self.binarized_sipp = pkl.load(file)
        file.close()

    def binary_pca_clustering(self, pca_components=15, kmean_clusters=15):
        '''
        INPUT: number of components for pca, number of clusters for kmeans
        OUTPUT: None.

        This runs a PCA on the decomposed sipp codes, and then does a KMeans
        clustering to find cars with similar properties. It then assigns those
        car clusters to customers in the origin_destion dataframe.
        '''
        self.pca = PCA(n_components=pca_components)
        target_pca = self.pca.fit_transform(self.binarized_sipp)
        kmean = KMeans(n_clusters=kmean_clusters)
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
        origin.columns = ['o' + x for x in origin.columns]
        dest = pd.get_dummies(self.cars['destination'])
        dest.columns = ['d' + x for x in dest.columns]

        self.cars['origin_dest'] = \
            self.cars['customer_country'].apply(lambda x: 'o' + str(x)) + \
            self.cars['destination'].apply(lambda x: 'd' + str(x))

        pairwise_names = []
        for o in origin.columns:
            for d in dest.columns:
                pairwise_names.append(o+d)

        pairwise_matrix = sparse.csc_matrix((96341, 1))
        for o in origin.columns:
            for d in dest.columns:
                t_f = origin[o] * dest[d]
                pairwise_matrix = sparse.hstack([pairwise_matrix,
                                                 sparse.csr_matrix(t_f).T])
        pairwise_matrix = pairwise_matrix[:, 1:]

        self.country_dummies_colnames = \
            list(origin.columns) + list(dest.columns) + pairwise_names

        origin_sparse = sparse.csc_matrix(origin)
        dest_sparse = sparse.csc_matrix(dest)

        self.country_dummies = sparse.hstack([origin_sparse,
                                              dest_sparse,
                                              pairwise_matrix])

    def create_nb_priors(self, flatten=.15):
        '''
        INPUT: flatten parameter.
        OUTPUT: None.

        This fits the prior distribution of car selections to a gaussian
        distribtuion it then creates a new series of priors by flattening the
        distribution. These priors are then used in the Naieve Bayes predict
        proba.
        '''
        def tg(x, a, b, c):
            return gamma(a, loc=b, scale=c).pdf(x)

        vc = self.cars['cluster'].value_counts()
        prefs = vc / sum(vc)
        vals = np.array([x + 1 for x in range(len(prefs))])
        params, pcov = curve_fit(tg, vals, prefs)
        new_priors = tg(vals, params[0] + flatten, params[1], params[2])
        self.new_priors = new_priors / sum(new_priors)

    def fit_nb_cartypes(self, use_weights=False):
        '''
        INPUT: None
        OUTPUT: None

        This runs a naive bayes algorithim on the country_dummies matrix to
        predict the kind of car cluster that that individual preffers. It then
        creates a probability ranking over clusters for each individual in the
        dataset.
        '''
        if use_weights:
            self.nb = BernoulliNB(class_prior=self.new_priors)
        else:
            self.nb = BernoulliNB()

        self.nb.fit(self.country_dummies, self.cars['cluster'])
        self.car_preds = self.nb.predict_proba(self.country_dummies)

    def _eval_(self):
        '''
        INPUT: None
        OUTPUT: None

        This function simply re-runs Naive Bayes, creates a classifier report,
        and a confusion matrix. It also tests for accuracy, creates an accuracy
        score based on overall predictions and the top 2 predicted car types.
        '''
        X_train, X_test, y_train, y_test = \
            train_test_split(self.country_dummies, self.cars['cluster'],
                             test_size=0.2, random_state=42)

        nb = BernoulliNB(class_prior=self.new_priors)
        nb.fit(X_train, y_train)
        pred = nb.predict(X_test)
        self.nb_confusion = confusion_matrix(y_test, pred)
        diag_sum = np.trace(confusion_matrix(y_test, pred))
        total = np.sum(confusion_matrix(y_test, pred))
        self.nb_accuracy = diag_sum/float(total)
        self.nb_reports = classification_report(y_test, pred)
        self.baseline = np.bincount(y_test.T).max() / float(y_test.shape[0])

        self.probs = nb.predict_proba(X_test)
        top2 = np.argsort(self.probs, axis=1)[:, -2:]
        t_f = []
        for i in xrange(y_test.shape[0]):
            t_f.append(np.in1d(y_test[i], top2[i])[0])
        self.nb_top2_accuracy = sum(t_f)/float(len(t_f))

    def fit_kmeans_customer_types(self, clusters=20):
        '''
        INPUT: integer, representing the number of customer types
        OUTPUT: None

        This uses KMeans on the prefference rankings in order to identify
        customer clusters with similar prefferences.
        '''

        km = KMeans(n_clusters=clusters)
        km.fit(self.car_preds)
        self.cars['car_pref_cluster'] = km.predict(self.car_preds)

    def _present_car_cluster(self, i):
        '''
        INPUT: integer, representing a car cluster
        OUTPUT: None.

        Print top 10 most common cars in the specified cluster.
        '''
        car_name = self.cars[self.cars['cluster'] == i]['car_name']
        print list(car_name.value_counts()[0:9].index)
        sipp_code = self.cars[self.cars['cluster'] == i]['sipp_code']
        print list(sipp_code.value_counts()[0:9].index)

    def _present_customer_cluster(self, i):
        '''
        INPUT: integer, representing a customer cluster
        OUTPUT: None

        Print top 6 most common cars for each car cluster in the order
        that the car clusters are preffered as well as the top 5 associated
        origin/destination pairs for that customer cluster.
        '''
        indices = self.cars[self.cars['car_pref_cluster'] == i]['cluster']
        indices = indices.value_counts()[0:6].index
        top_cars = []
        for k in indices:
            car_name = self.cars[self.cars['cluster'] == k]['car_name']
            top_cars.append((list(car_name.value_counts()[0:5].index), k))

        o_d = self.cars[self.cars['car_pref_cluster'] == i]['origin_dest']
        print 'origin/destination pairs: ' + \
              str(list(o_d.value_counts()[0:5].index))
        for j in top_cars:
            print 'car cluster: ' + str(j[1])
            print 'top cars in cluster: ' + str(j[0])

    def _create_d3_data(self, path=None):
        '''
        INPUT: output file directory
        OUTPUT: writes out a javascript hashable arrays for use in d3.
                These conatian information about each cluster.

        This will create an origin/destination dict, i.e. for each country
        there is an entry, and each entry is itself a dict that associates
        all incoming traffic to the original country with sub countries and
        their clusters. E.g.
        {US: {SE:1, GB:2}}
        Means that people coming to the US from SE are in cluster 1 and people
        coming to US from GB are in cluster 2.
        This also writes out information about each prefernece cluster to
        another json array.
        '''
        if path:
            pass
        else:
            path = self.path

        thedict = {}
        origin = np.unique(self.cars['customer_country'])
        origin = [x for x in origin if str(x) != 'nan']
        destination = np.unique(self.cars['destination'])
        destination = [x for x in destination if str(x) != 'nan']

        for d in destination:
            cond_d = self.cars['destination'] == d
            thedict[d] = {}
            for o in origin:
                cond_o = self.cars['customer_country'] == o
                if np.sum(cond_o & cond_d) > 0:
                    vals = self.cars[cond_o & cond_d]['car_pref_cluster']
                    thedict[d][o] = int(vals.value_counts()[0:1].index[0])
        pth = path + 'country_analysis/cc_map/orig_dest.json'
        with open(pth, 'w') as outfile:
            json.dump(thedict, outfile)

        thedict = {}
        for c in np.unique(self.cars['car_pref_cluster']):
            thedict[str(c)] = {}
            cluster = self.cars[self.cars['car_pref_cluster'] == c]['cluster']
            indices = cluster.value_counts()[0:6].index
            for i, k in enumerate(indices):
                car_names = self.cars[self.cars['cluster'] == k]['car_name']
                top_cars = list(car_names.value_counts()[0:1].index)
                thedict[str(c)][str(i)] = "Car Group " + str(k) + ": " + \
                                          ", ".join(top_cars)
        pth = path + 'country_analysis/cc_map/cc_info.json'
        with open(pth, 'w') as outfile:
            json.dump(thedict, outfile)

        thedict = {}
        for d in np.unique(self.cars['destination']):
            thedict[d] = {}
            cond1 = self.cars['destination'] == d
            for c in np.unique(self.cars[cond1]['car_pref_cluster']):
                cond2 = self.cars['car_pref_cluster'] == c
                o = np.unique(self.cars[cond1 & cond2]['customer_country'])
                o = [str(x) for x in o if str(x) != 'nan']
                o = o[:10]
                thedict[d][str(c)] = "To " + d + " from cluster " + str(c) + \
                                     ": " + ", ".join(o)
        pth = path + 'country_analysis/cc_map/incoming_info.json'
        with open(pth, 'w') as outfile:
            json.dump(thedict, outfile)


if __name__ == '__main__':
    cc = CustomerCluster()
    cc.load()
    cc.binary_pca_clustering(pca_components=10, kmean_clusters=10)
    cc.dummy_matrix()
    cc._eval_()
    cc.create_nb_priors(flatten=1)
    cc.fit_nb_cartypes(use_weights=True)
    cc.fit_kmeans_customer_types(clusters=10)
    cc._create_d3_data()
