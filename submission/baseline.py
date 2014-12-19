import numpy as np
import sys
from sklearn.utils.extmath import safe_sparse_dot
import scipy.sparse

LOWER = 1e-2
UPPER = 1e9

class ColumnData(object):
    def __init__(self, X):
        self._X = scipy.sparse.csc_matrix(X)
    def get_row(self,row):
        return self._X[row,:]
    def get_column(self, col):
        """Iterator over (row, val) tuples for nonzero values in column col"""
        for ii in range(self._X.indptr[col], self._X.indptr[col + 1]):
           yield (self._X.indices[ii],  self._X.data[ii])


class BaselineGroupLasso(object):
    def __init__(self, max_iter=50, tol=1e-3, 
                 alpha=1e-4, beta=.5, sigma=.01, max_steps=30):
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta

    def fit(self, X, y):

        n_samples, n_features = X.shape
        n_classes = np.unique(y).shape[0]

        nv = n_samples * n_classes
        violation_max_old = float('inf')
        violation_init = 0

        ds = ColumnData(X)

        self.coefs_ = np.zeros((n_features, n_classes))
        A = np.ones((n_samples, n_classes))
        error_weight = 1. / n_samples

        inactive = set()
        active_set = range(0, n_features)
        active_size = len(active_set)
        active_size_start = active_size
        for n_iter in xrange(self.max_iter):
            print "iter", n_iter + 1
            violation_max = 0
            violation_sum = 0

            for j in active_set:
                if j in inactive:
                    continue
                Gj, Lpp_max, loss_j = self._derivatives(
                    A, ds, y, j, n_classes, error_weight)
                Gj_norm = np.linalg.norm(Gj, 2)
                Rj = np.linalg.norm(self.coefs_[j,:], 2)

                if Rj == 0:
                    Gj_norm -= self.alpha
                    if Gj_norm > 0:
                        violation = Gj_norm
                    elif Gj_norm + violation_max_old / nv <= 0:
                        # Shrink!
                        if self.verbose >= 2:
                            print "Shrink variable", j
                            inactive.add(j)
                            active_size -= 1
                            continue
                            #return 1
                else:
                    violation = np.fabs(Gj_norm - self.alpha)
                

                # Update violations.
                violation_max = max(violation_max, violation)
                violation_sum += violation
                
                Wj = self.coefs_[j,:]               
                Vj = Wj -  Gj / Lpp_max
                Vj_norm = np.linalg.norm(Vj, 2)
                if Vj_norm != 0:
                    scaling = 1 - self.alpha / (Lpp_max * Vj_norm )
                else: scaing = -float('inf')

                if scaling < 0:
                    scaling = 0
                delta_j = scaling * Vj - Wj
                
                 # Project (proximity operator).
                delta = 0
                dmax = - sys.float_info.max
                delta = np.dot(Gj, delta_j)
                dmax = max(dmax, np.max(np.fabs(delta_j)))

                # Do not bother update if update is too small.
                if dmax < 1e-12:
                    continue

                delta_j_old = np.zeros_like(delta_j)
                step = 1
                while 1:            

                    loss_new = self._update(
                        A, ds, y, delta_j, delta_j_old,
                        j, n_classes, error_weight)
                    if step >= self.max_steps:
                        if self.max_steps > 1:
                            #if self.verbose >= 2:
                            print "Max steps reached during line search..."
                            recompute = 1
                        break
                    Rj_new = 0
                    Rj_new = np.linalg.norm(Wj + delta_j)

                    if step == 1:
                        delta += self.alpha * (Rj_new - Rj)
                        delta *= self.sigma
                    # Check decrease condition
                    if loss_new - loss_j + \
                        self.alpha * (Rj_new - Rj) <= delta:
                        break
                    delta *= self.beta
                    delta_j_old = delta_j.copy()
                    delta_j *= self.beta
                    step += 1

                # Update solution
                self.coefs_[j,:] += delta_j

 
            # Initialize violations.
            if n_iter == 0 and violation_init == 0:
                violation_init = violation_sum

            if violation_sum <= self.tol * violation_init:
                if active_size == active_size_start:
                    print "\nConverged at iteration", n_iter + 1
                    break
                else:
# When shrinking is enabled, we need to do one more outer
# iteration on the entire optimization problem.
                    active_size = active_size_start
                    violation_max_old = sys.float_info.max
                    continue
            violation_max_old = violation_max


    def _update(self,
        A, ds, y, delta_j, delta_j_old, j, n_classes, error_weight):
        loss_new = 0
        for i, Xij, in ds.get_column(j):
            tmp = delta_j_old[y[i]] - delta_j[y[i]]
            for r in xrange(n_classes):
                if r != y[i]:
                    Air_new = A[i,r] + \
                        (tmp - (delta_j_old[r] - delta_j[r])) * Xij
                    A[i,r] = Air_new
                    if Air_new > 0:
                        loss_new += error_weight * Air_new * Air_new
        return loss_new

    def _derivatives(self, A, ds, y, j, n_classes, error_weight):
        Gj = np.zeros((n_classes))
        Hj = np.zeros((n_classes))
        loss_tmp = 0
        for i, Xij, in ds.get_column(j):
            for r in xrange(n_classes):
                if y[i] != r and A[i, r] > 0:
                    loss_ir = A[i,r]         
                    loss_tmp += error_weight * loss_ir * loss_ir
                    tmp = error_weight * Xij 
                    tmp2 = tmp * loss_ir
                    Gj[y[i]] -= tmp2
                    Gj[r] += tmp2
                    tmp2 = tmp * Xij
                    Hj[y[i]] += tmp2
                    Hj[r] += tmp2
        Lpp_max =-1<<31 
     
        for r in xrange(n_classes):
            Gj[r] *= 2
            Lpp_max = max(Lpp_max, Hj[r])
        Lpp_max *= 2
        Lpp_max = min(max(Lpp_max, LOWER), UPPER)
        return Gj, Lpp_max, loss_tmp 

    def score(self, X, y):
        pred = safe_sparse_dot(X, self.coefs_)
        y_pred = np.argmax(pred, axis=1)
        n_correct = np.where(y_pred == y)[0].shape[0]
        return n_correct / float(X.shape[0])

