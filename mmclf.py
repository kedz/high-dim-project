import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import scipy.sparse


class ColumnData(object):
    def __init__(self, X):
        self._X = scipy.sparse.csc_matrix(X)

    def get_column(self, col):
        """Iterator over (row, val) tuples for nonzero values in column col"""
        for ii in range(self._X.indptr[col], self._X.indptr[col + 1]):
           yield (self._X.indices[ii],  self._X.data[ii])
      
class SlowLGClassifier(object):
    def __init__(self, max_iter=25):
        self.max_iter = max_iter

    def fit(self, X, y):
        n_features = X.shape[1]
        n_classes = np.unique(y).shape[0]
        n_samples = X.shape[0]


        for n_iter in range(self.max_iter):
            for j in range(n_features):
                for i in range(n_samples):
                    if X[i,j] != 0:
                        1 + 1
            
class LatentGroupClassifier(object):
    def __init__(self, max_iter=25, C=1.0, ls_max_iter=30, alpha=.5, sigma=.01):
        self.max_iter = max_iter
        self.C = C
        
        # line search params   
        self.ls_max_iter = ls_max_iter
        self.sigma = sigma
        self.alpha = alpha

    def score(self, X, y):
        pred = safe_sparse_dot(X, self.coefs_.T)
        y_pred = np.argmax(pred, axis=1)
        n_correct = np.where(y_pred == y)[0].shape[0]
        return n_correct / float(X.shape[0])
         

    def fit(self, X, y):
        n_features = X.shape[1]
        n_classes = np.unique(y).shape[0]
        n_samples = X.shape[0]
        ds = ColumnData(X)
        error = np.ones((n_samples, n_classes))
        self.coefs_ = np.zeros((n_classes, n_features))
        error_weight = 1. / float(n_samples)

        for n_iter in range(self.max_iter):
            for j in range(n_features):
                Gj, hj_max, loss = self._derivatives(
                    n_classes, j, ds, y, error, error_weight) 
                #print "Coordinate loss:",loss, "Hessian row inf norm:", hj_max 
                Gj_norm = np.linalg.norm(Gj, 2)
                Rj = np.linalg.norm(self.coefs_[:,j], 2)
                #print Gj_norm, Rj
                
                Vj = self.coefs_[:,j] - Gj / hj_max
                vnorm = np.linalg.norm(Vj, 2)
                if vnorm != 0:
                    scaling = 1. - self.C / (hj_max * vnorm)
                else:
                    scaling = 1.
                #print scaling
                if scaling < 0:
                    scaling = 0
                D = scaling * Vj - self.coefs_[:,j]
                #print D
                #print "Max delta", 
                dmax = np.max(np.fabs(D))
                if dmax < 1e-12:
                    
                    continue

                ls_cutoff = np.dot(D, Gj)

                D_old = np.zeros((n_classes))    
                step = 1
                while 1:
                    Loss_new = self._update_errors(
                        n_classes, j, ds, y, error, error_weight, D, D_old)    
                    if step >= self.ls_max_iter:
                        #if self.ls_max_iter > 1:
                        #if self.verbose >= 2:
                        #print "Max steps reached during line search..."
                        #recompute = 1
                        break
                    
                    Rj_new = np.linalg.norm(self.coefs_[:,j] + D)        
                    if step == 1:
                        ls_cutoff = self.sigma * self.C * (Rj_new - Rj)
                        ls_cutoff *= self.alpha
                    #if step == self.ls_max_iter:
                    #    print "Warning max steps in line search reached"
                    if Loss_new - loss + self.C * (Rj_new - Rj) <= ls_cutoff:
                        #print "Good break"
                        break
                    
                    ls_cutoff *= self.alpha
                    D_old = D
                    D *= self.alpha 
                    step += 1

                self.coefs_[:,j] += D    
            print self.score(X, y)       
                                         
   # def _loss(self, error
    def _update_errors(self, n_classes, j, ds, y, error, error_weight, D, D_old):
        Loss_new = 0
        for i, Xij in ds.get_column(j):
            tmp = D_old[y[i]] - D[y[i]]
            for r in xrange(n_classes):
                if r != y[i]:
                    
                    new_error = error[i,r] + (tmp - (D_old[r] - D[r])) * Xij
                    error[i,r] = new_error
                    if new_error > 0:
                        Loss_new += 2 * error_weight * new_error * new_error
        return Loss_new                    

    def _derivatives(self, n_classes, j, ds, y, error, error_weight):
        Loss = 0
        Gj = np.zeros((n_classes))
        hj = np.zeros((n_classes))
        for r in xrange(n_classes):
            for i, Xij in ds.get_column(j):
                if y[i] != r and error[i,r] > 0:
                    Loss += error_weight * error[i,r] * error[i,r]
                    Gj[y[i]] += - error[i,r] * Xij
                    Gj[r] += error[i,r] * Xij
                    hj[y[i]] += Xij * Xij
                    hj[r] += Xij * Xij

        Lpp_max = float('-inf')
        for k in xrange(n_classes):
            Gj[k] *= 2 * error_weight
            Lpp_max = max(Lpp_max, hj[k])

        Lpp_max *= 2 * error_weight
        Lpp_max = min(max(Lpp_max, 1.0e-4), 1e9)
        #Lpp_max[0] = min(max(Lpp_max[0], LOWER), UPPER)
        return Gj, Lpp_max, Loss

