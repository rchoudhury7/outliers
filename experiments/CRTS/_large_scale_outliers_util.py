import time
import numpy

def compute_outliers(X, algorithm="buffer_kd_tree", outlier_criterion="inverse_densities", algorithm_params=None):
    """ Computes an outlier score (the higher, 
    the more an object is an outlier)
    """

    n_neighbors = algorithm_params['n_neighbors']
    leaf_size = algorithm_params['leaf_size']
    
    verbose = algorithm_params['verbose']

    # nearest neighbors and densities
    if algorithm == "sklearn":
        from sklearn.neighbors import NearestNeighbors as NearestNeighborsSKLEARN
        nbrs = NearestNeighborsSKLEARN(n_neighbors=n_neighbors, algorithm='kd_tree', leaf_size=leaf_size).fit(X)
    else:
        from bufferkdtree.neighbors import NearestNeighbors as NearestNeighbors
        tree_depth = algorithm_params['tree_depth']
        n_jobs = algorithm_params['n_jobs']
        plat_dev_ids = algorithm_params['plat_dev_ids']
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                                algorithm=algorithm, \
                                tree_depth=tree_depth, \
                                leaf_size=leaf_size, \
                                plat_dev_ids=plat_dev_ids, \
                                n_jobs=n_jobs, \
                                verbose=verbose)

    start = time.time()
    nbrs.fit(X)
    end = time.time()
    if verbose > 0:
        print "--------------------------------------------------------------------------------"
        print("Fitting time: %f" % (end - start))
        print "--------------------------------------------------------------------------------"

    start = time.time()
    distances, indices = nbrs.kneighbors(X)
    end = time.time()
    if verbose > 0:
        print "--------------------------------------------------------------------------------"
        print("Querying time: %f" % (end - start))
        print "--------------------------------------------------------------------------------"

    # compute densities
    densities = 1.0 / (distances.sum(axis=1) / n_neighbors)

    # compute density based ranking (reverse order)
    if outlier_criterion == "inverse_densities":
        out_scores = 1.0 / densities
    else:
        raise Exception("Unkown outlier criterion!")
        
    out_ranks = numpy.argsort(out_scores)[::-1]

    return out_ranks, out_scores, densities
