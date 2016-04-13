
import copy
import numpy

def make_binary(X, Y, class1, class2=None, label_class1=1, label_class2=2):
    """ Generates a binary classification task based on the class
    labels provided. In case class2 is None, then ALL remaining
    classes will be assigned to the second class.
    """
    if class2 is None:
        class2 = list(set(Y).difference(set(class1)))

    selector1 = numpy.zeros(len(Y))
    for i in xrange(len(Y)):
        selector1[i] = Y[i] in class1
    selector1 = selector1.astype(bool)

    selector2 = numpy.zeros(len(Y))
    for i in xrange(len(Y)):
        selector2[i] = Y[i] in class2
    selector2 = selector2.astype(bool)

    X1 = X[selector1]
    X2 = X[selector2]
    Y1 = label_class1*numpy.ones(selector1.sum())
    Y2 = label_class2*numpy.ones(selector2.sum())

    X = numpy.concatenate((X1,X2), axis=0)
    Y = numpy.concatenate((Y1,Y2), axis=0)

    perm = numpy.random.permutation(range(len(X)))
    X = X[perm]
    Y = Y[perm]

    return X, Y
    
    
