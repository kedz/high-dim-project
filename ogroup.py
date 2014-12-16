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

    def fit(self, X, y, groups):

        n_samples, n_features = X.shape
        n_classes = np.unique(y).shape[0]

        nv = n_samples * n_classes
        violation_max_old = float('inf')
        violation_init = 0
        violation = 0

        ds = ColumnData(X)

        self.coefs_ = np.zeros((n_features, n_classes))
        A = np.ones((n_samples, n_classes))
        error_weight = 1. / n_samples

        inactive = set()
        active_set = range(0, n_features)
        active_size = len(active_set) * n_classes
        active_size_start = active_size

        for n_iter in xrange(self.max_iter):
            print "iter", n_iter + 1
            violation_max = 0
            violation_sum = 0

            for r in xrange(n_classes): 
                for g_id, group in enumerate(groups):
                    g_start = group[0]
                    g_end = group[1]
                    if (g_id, r) in inactive:
                        continue

                    Ggr = np.zeros((g_end - g_start))
                
            #for j in active_set:
            #    if j in inactive:
            #        continue
                    Lpp_max =-1<<31 
                    for j in xrange(g_start, g_end):
                        Gjr, Hjr, loss_g = self._derivatives_feat(
                            A, ds, y, j, r, n_classes, error_weight)
                        Ggr[j - g_start] = Gjr        
                        Lpp_max = max(Lpp_max, Hjr)
                    Lpp_max = min(max(Lpp_max, LOWER), UPPER)

                    Ggr_norm = np.linalg.norm(Ggr, 2)
                    
#                    Rgr = np.linalg.norm(self.coefs_[g_start:g_end,r], 2)
#
#                    if Rgr == 0:
#                        Ggr_norm -= self.alpha
#                        if Ggr_norm > 0:
#                            violation = Ggr_norm
#                        elif Ggr_norm + violation_max_old / nv <= 0:
#                            # Shrink!
#                            #if self.verbose >= 2:
#                            print "Shrink variable", j
#                            inactive.add((g_id, r))
#                            active_size -= 1
#                            continue
#                                #return 1
#                    else:
#                        violation = np.fabs(Ggr_norm - self.alpha)
#                
#
#                    # Update violations.
#                    violation_max = max(violation_max, violation)
#                    violation_sum += violation
                
                    Wgr = self.coefs_[g_start:g_end,r]               
                    Vgr = Wgr - Ggr / Lpp_max
                    Vgr_norm = np.linalg.norm(Vgr, 2)
                    if Vgr_norm != 0:
                        scaling = 1 - self.alpha / (Lpp_max * Vgr_norm )
                    else: scaling = 0

                    if scaling < 0:
                        scaling = 0

                    # Testing simple thing here
                    Wstar_gr = scaling *Vgr
                    delta_gr = Wstar_gr - Wgr
                    self.coefs_[g_start:g_end,r] = Wstar_gr

                    #delta_gr_old = np.zeros_like(delta_gr)
                    delta_j_old = np.zeros((n_classes))
#                    step = 1
#                    while 1:            
                    delta_j = np.zeros((n_classes))
                    for j in xrange(g_start, g_end):
                        delta_j.fill(0)
                        delta_j[r] = delta_gr[j - g_start]
#                            delta_j_old.fill(0)
#                            delta_j_old[r] = delta_gr_old[j - g_start]
                        loss_new = self._update(
                            A, ds, y, delta_j, delta_j_old,
                            j, n_classes, error_weight)
#

                    
#                    delta_gr = scaling * Vgr - Wgr
#                
#                    delta = 0
#                    dmax = - sys.float_info.max
#                    delta = np.dot(Ggr, delta_gr)
#                    dmax = max(dmax, np.max(np.fabs(delta_gr)))
#
#                    # Do not bother update if update is too small.
#                    if dmax < 1e-12:
#                        continue
#
#                    delta_gr_old = np.zeros_like(delta_gr)
#                    delta_j_old = np.zeros((n_classes))
#                    step = 1
#                    while 1:            
#
#                        delta_j = np.zeros((n_classes))
#                        for j in xrange(g_start, g_end):
#                            delta_j.fill(0)
#                            delta_j[r] = delta_gr[j - g_start]
#                            delta_j_old.fill(0)
#                            delta_j_old[r] = delta_gr_old[j - g_start]
#                            loss_new = self._update(
#                                A, ds, y, delta_j, delta_j_old,
#                                j, n_classes, error_weight)
#                            print loss_new, loss_g
#
#                        if step >= self.max_steps:
#                            if self.max_steps > 1:
#                            #if self.verbose >= 2:
#                                print "Max steps reached during line search...",
#                                print (g_id, r)
#                                recompute = 1
#                            break
#                        Rgr_new = np.linalg.norm(Wgr + delta_gr)
#
#                        if step == 1:
#                            delta += self.alpha * (Rgr_new - Rgr)
#                            delta *= self.sigma
#                        # Check decrease condition
#                        print (g_id, r), loss_new - loss_g + \
#                            self.alpha * (Rgr_new - Rgr), "<=", delta
#                        
#                        if loss_new - loss_g + \
#                            self.alpha * (Rgr_new - Rgr) <= delta:
#                            break
#                        
#                        delta *= self.beta
#                        delta_gr_old = delta_gr.copy()
#                        delta_gr *= self.beta
#                        step += 1
#
#                    # Update solution
#                    self.coefs_[g_start:g_end,r] += delta_gr

 
#            # Initialize violations.
#            if n_iter == 0 and violation_init == 0:
#                violation_init = violation_sum
#
#            if violation_sum <= self.tol * violation_init:
#                if active_size == active_size_start:
#                    print "\nConverged at iteration", n_iter + 1
#                    break
#                else:
## When shrinking is enabled, we need to do one more outer
## iteration on the entire optimization problem.
#                    active_size = active_size_start
#                    violation_max_old = sys.float_info.max
#                    continue
#            violation_max_old = violation_max


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



#    def _derivatives(self, A, ds, y, j, n_classes, error_weight):
#        Gj = np.zeros((n_classes))
#        Hj = np.zeros((n_classes))
#        loss_tmp = 0
#        for i, Xij, in ds.get_column(j):
#            for r in xrange(n_classes):
#                if y[i] != r and A[i, r] > 0:
#                    loss_ir = A[i,r]         
#                    loss_tmp += error_weight * loss_ir * loss_ir
#                    tmp = error_weight * Xij 
#                    tmp2 = tmp * loss_ir
#                    Gj[y[i]] -= tmp2
#                    Gj[r] += tmp2
#                    tmp2 = tmp * Xij
#                    Hj[y[i]] += tmp2
#                    Hj[r] += tmp2
#        Lpp_max =-1<<31 
#     
#        for r in xrange(n_classes):
#            Gj[r] *= 2
#            Lpp_max = max(Lpp_max, Hj[r])
#        Lpp_max *= 2
#        Lpp_max = min(max(Lpp_max, LOWER), UPPER)
#        return Gj, Lpp_max, loss_tmp 

    def _derivatives_feat(self, A, ds, y, j, r_select, n_classes, error_weight):
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

        
        return Gj[r_select] * 2, Hj[r_select] * 2, loss_tmp
#        for r in xrange(n_classes):
#            Gj[r] *= 2
#            Lpp_max = max(Lpp_max, Hj[r])
#        Lpp_max *= 2
#        return Gj, Lpp_max, loss_tmp 

    def score(self, X, y):
        pred = safe_sparse_dot(X, self.coefs_)
        y_pred = np.argmax(pred, axis=1)
        n_correct = np.where(y_pred == y)[0].shape[0]
        return n_correct / float(X.shape[0])

