from sklearn.utils import gen_batches

def minibatch(X, y):
    n_samples = X.shape[0]
    batch_size = 32
    Xb = []
    yb= []

    for sl in gen_batches(n_samples, batch_size):
        Xb.append(safe_indexing(X, sl))
        yb.append(safe_indexing(y, sl))
    
    return Xb, yb

def safe_indexing(X, indices):
    if hasattr(X, 'iloc'):    # pandas
        return X.iloc[indices]
    else:                      # numpy or list
        return X[indices]
