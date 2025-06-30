import numpy as np
from econml.utilities import cross_product
from econml.grf._base_grf import BaseGRF
from econml.grf.classes import RegressionForest
from econml.utilities import check_inputs
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y
from matplotlib import pyplot as plt
from scipy.stats import truncnorm, norm
import itertools
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
from econml.iv.dr import LinearDRIV, ForestDRIV
from sklearn.linear_model import LogisticRegression
from econml.grf import CausalIVForest
from econml.iv.sieve import SieveTSLS
from econml.iv.dml import DMLIV, OrthoIV
from sklearn.preprocessing import PolynomialFeatures

# Valid instrument
def generate_data_valid(n_samples, seed):
    np.random.seed(seed)
    # Generate baseline covariates X
    X = np.random.uniform(0, 1, size=(n_samples, 5))

    # Functional transformation of X
    X_star = 1 / (1 + np.exp(-20 * (X - 0.5)))

    # Generate unmeasured confounder U
    # Compute the variance for each sample based on X_star
    variance = 0.25 + 0.5 * X_star[:, 0] + 0.15 * X_star[:, 1] - 0.1 * X_star[:, 2] - 0.1 * X_star[:, 3] + 0.1 * X_star[:, 4]
    # Compute the standard deviation (square root of variance)
    std_dev = np.sqrt(variance)
    # Truncation limits (-0.5 to 0.5)
    lower, upper = -0.5, 0.5
    # Generate U values from a truncated normal distribution
    U = truncnorm.rvs(a=(lower - 0) / std_dev, b=(upper - 0) / std_dev, loc=0, scale=std_dev, size=X_star.shape[0], random_state=seed)
    
    # Generate invalid instrument Z
    Z_prob = 1 / (1 + np.exp(-0.8 - 1 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] +0.2 * X_star[:, 3] - 0.1 * X_star[:, 4]))
    Z = np.random.binomial(1, Z_prob, size=n_samples)

    # Generate treatment A
    kappa1 = 0.1
    A_prob = 1 / (1 + np.exp(2 - 1.5 * Z - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U
    A = np.random.binomial(1, A_prob, size=n_samples)
    A_prob0 = 1 / (1 + np.exp(2 - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U
    A_prob1 = 1 / (1 + np.exp(0.5 - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U

    # Generate outcome Y
    kappa2 = 1
    
    Y_mean = -2 + (2 * X_star[:, 0] + 0.5 * X_star[:, 1] + 0.5 * X_star[:, 2]) * A + 2 * X_star[:, 0] + 0.5 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] + 0.1 * X_star[:, 4] + kappa2 * U
    Y = Y_mean + np.random.normal(0, 1, size=n_samples)

    # Target parameter gamma
    gamma = 2 * X_star[:, 0] + 0.5 * X_star[:, 1] + 0.5 * X_star[:, 2]
    
    return X_star, Y, A, Z, gamma, Z_prob, A_prob, A_prob0, A_prob1


# Invalid instrument with linear treatment effect
def generate_data_invalid(n_samples, seed):
    np.random.seed(seed)
    # Generate baseline covariates X
    X = np.random.uniform(0, 1, size=(n_samples, 5))

    # Functional transformation of X
    X_star = 1 / (1 + np.exp(-20 * (X - 0.5)))

    # Generate unmeasured confounder U
    # Compute the variance for each sample based on X_star
    variance = 0.25 + 0.5 * X_star[:, 0] + 0.15 * X_star[:, 1] - 0.1 * X_star[:, 2] - 0.1 * X_star[:, 3] + 0.1 * X_star[:, 4]
    # Compute the standard deviation (square root of variance)
    std_dev = np.sqrt(variance)
    # Truncation limits (-0.5 to 0.5)
    lower, upper = -0.5, 0.5
    # Generate U values from a truncated normal distribution
    U = truncnorm.rvs(a=(lower - 0) / std_dev, b=(upper - 0) / std_dev, loc=0, scale=std_dev, size=X_star.shape[0], random_state=seed)

    # Generate invalid instrument Z
    Z_prob = 1 / (1 + np.exp(-0.8 - 1 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] +0.2 * X_star[:, 3] - 0.1 * X_star[:, 4]))
    Z = np.random.binomial(1, Z_prob, size=n_samples)

    # Generate treatment A
    kappa1 = 0.1
    A_prob = 1 / (1 + np.exp(2 - 1.5 * Z - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U
    A = np.random.binomial(1, A_prob, size=n_samples)
    A_prob0 = 1 / (1 + np.exp(2 - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U
    A_prob1 = 1 / (1 + np.exp(0.5 - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U

    # Generate outcome Y
    kappa2 = 1
    
    Y_mean = -2 + (2 * X_star[:, 0] + 0.5 * X_star[:, 1] + 0.5 * X_star[:, 2]) * A + 2 * X_star[:, 0] + 0.5 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] + 0.1 * X_star[:, 4] - 2*Z + kappa2 * U
    Y = Y_mean + np.random.normal(0, 1, size=n_samples)

    # Target parameter gamma
    gamma = 2 * X_star[:, 0] + 0.5 * X_star[:, 1] + 0.5 * X_star[:, 2]
    
    return X_star, Y, A, Z, gamma, Z_prob, A_prob, A_prob0, A_prob1


# Invalid instrument with nonlinear treatment effect
def generate_data_nonlinear(n_samples, seed):
    np.random.seed(seed)
    # Generate baseline covariates X
    X = np.random.uniform(0, 1, size=(n_samples, 5))

    # Functional transformation of X
    X_star = 1 / (1 + np.exp(-20 * (X - 0.5)))

    # Generate unmeasured confounder U
    # Compute the variance for each sample based on X_star
    variance = 0.25 + 0.5 * X_star[:, 0] + 0.15 * X_star[:, 1] - 0.1 * X_star[:, 2] - 0.1 * X_star[:, 3] + 0.1 * X_star[:, 4]
    # Compute the standard deviation (square root of variance)
    std_dev = np.sqrt(variance)
    # Truncation limits (-0.5 to 0.5)
    lower, upper = -0.5, 0.5
    # Generate U values from a truncated normal distribution
    U = truncnorm.rvs(a=(lower - 0) / std_dev, b=(upper - 0) / std_dev, loc=0, scale=std_dev, size=X_star.shape[0], random_state=seed)

    # Generate invalid instrument Z
    Z_prob = 1 / (1 + np.exp(-0.8 - 1 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] +0.2 * X_star[:, 3] - 0.1 * X_star[:, 4]))
    Z = np.random.binomial(1, Z_prob, size=n_samples)

    # Generate treatment A
    kappa1 = 0.1
    A_prob = 1 / (1 + np.exp(2 - 1.5 * Z - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U
    A = np.random.binomial(1, A_prob, size=n_samples)
    A_prob0 = 1 / (1 + np.exp(2 - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U
    A_prob1 = 1 / (1 + np.exp(0.5 - 0.6 * X_star[:, 0] + 0.2 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] - 0.1 * X_star[:, 4])) + kappa1 * U

    # Generate outcome Y
    kappa2 = 1
    
    Y_mean = -2 + (2*X_star[:, 0]**2 + 0.5 * X_star[:, 1]) * A + 2 * X_star[:, 0] + 0.5 * X_star[:, 1] + 0.2 * X_star[:, 2] + 0.1 * X_star[:, 3] + 0.1 * X_star[:, 4] - 2*Z + kappa2 * U
    Y = Y_mean + np.random.normal(0, 1, size=n_samples)

    # Target parameter gamma
    gamma = 2*X_star[:, 0]**2 + 0.5 * X_star[:, 1]
    
    return X_star, Y, A, Z, gamma, Z_prob, A_prob, A_prob0, A_prob1


# IRF module inherited from BaseGRF
class CustomGRF(BaseGRF):
    def __init__(self,
                 n_estimators=600, *,
                 criterion="het",
                 muE=None,
                 piE=None,
                 betaE=None,
                 tauE=None,
                 rhoE=None,
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=40,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 max_features='auto',
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.15,
                 honest=True,
                 inference=True,
                 fit_intercept=False,
                 subforest_size=5,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf, min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features, min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples, min_balancedness_tol=min_balancedness_tol,
                         honest=honest, inference=inference, fit_intercept=fit_intercept,
                         subforest_size=subforest_size, n_jobs=n_jobs, random_state=random_state, verbose=verbose,
                         warm_start=warm_start)
        self.muE = muE
        self.piE = piE
        self.betaE = betaE
        self.tauE = tauE
        self.rhoE = rhoE
        
    def _get_alpha_and_pointJ(self, X, T, y, *, Z):
        XZ1 = np.hstack([X, np.ones((X.shape[0], 1))])
        XZ0 = np.hstack([X, np.zeros((X.shape[0], 1))])
        XZ = np.hstack([X, Z.reshape(-1, 1)])
        muz0 = self.muE.predict(XZ0)
        muz1 = self.muE.predict(XZ1)
        varz0 = (1-muz0)*muz0
        varz1 = (1-muz1)*muz1
        varA = varz1 - varz0

        T = T.ravel()
        Z = Z.ravel()
        y = y.ravel()
                
        beta = self.betaE.predict(X).ravel()
        tau = self.tauE.predict(XZ).ravel()
        rho = self.rhoE.predict(X).ravel()

        fZ = (2*Z-1)/(Z * self.piE.predict(X).ravel() + (1 - Z) * (1 - self.piE.predict(X).ravel()))
        eps = T - self.muE.predict(XZ).ravel()
      
        J = np.ones((X.shape[0], 1))
        A = ((y - beta*T - tau) * eps - rho) * fZ / varA.ravel() + beta
        return A.reshape(-1, 1), J

    def _get_n_outputs_decomposition(self, X, T, y, *, Z):
        n_relevant_outputs = 1
        n_outputs = 1
        if self.fit_intercept:
            n_outputs += 1
        return n_outputs, 1
    

# Single simulation function    
def one_exp(param_dict, seed, datagen):
    # Generate training data
    if datagen == 1:  # model 1
        X_train, Y_train, A_train, Z_train, gamma_train, pi_train, mu_train, mu_train0, mu_train1 = generate_data_valid(n_train_samples, seed=seed)
        X_test, Y_test, A_test, Z_test, gamma_test, pi_test, mu_test, mu_test0, mu_test1 = generate_data_valid(n_test_samples, seed=seed+1)
    elif datagen == 2:  # model 2
        X_train, Y_train, A_train, Z_train, gamma_train, pi_train, mu_train, mu_train0, mu_train1 = generate_data_invalid(n_train_samples, seed=seed)
        X_test, Y_test, A_test, Z_test, gamma_test, pi_test, mu_test, mu_test0, mu_test1 = generate_data_invalid(n_test_samples, seed=seed+1)
    elif datagen == 3:  # model 3
        X_train, Y_train, A_train, Z_train, gamma_train, pi_train, mu_train, mu_train0, mu_train1 = generate_data_nonlinear(n_train_samples, seed=seed)
        X_test, Y_test, A_test, Z_test, gamma_test, pi_test, mu_test, mu_test0, mu_test1 = generate_data_nonlinear(n_test_samples, seed=seed+1)

    # Initialization of nuisance parameter estimators
    muE = RegressionForest(n_estimators=500, min_samples_leaf=5, min_samples_split=20, max_depth=None,
                        min_impurity_decrease=0.0, max_samples=0.75, min_balancedness_tol=0.45,
                        warm_start=False, inference=False, subforest_size=5, min_weight_fraction_leaf = 0.0,
                        honest=True, verbose=0, n_jobs=-1, random_state=1235)
    piE = RegressionForest(n_estimators=500, min_samples_leaf=5, min_samples_split=20, max_depth=None,
                        min_impurity_decrease=0.0, max_samples=0.75, min_balancedness_tol=0.45,
                        warm_start=False, inference=False, subforest_size=4,
                        honest=True, verbose=0, n_jobs=-1, random_state=1235)
    betaE = RegressionForest(n_estimators=500, min_samples_leaf=8, min_samples_split=15, max_depth=None,
                        min_impurity_decrease=0.0, max_samples=0.75, min_balancedness_tol=0.45,
                        warm_start=False, inference=False, subforest_size=5, min_weight_fraction_leaf = 0.0,
                        honest=True, verbose=0, n_jobs=-1, random_state=1235)
    tauE = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, min_samples_split=6, n_jobs=-1, random_state=1235)
    rhoE = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, min_samples_split=2, n_jobs=-1, random_state=1235)

    XZ_train = np.hstack([X_train, Z_train.reshape(-1, 1)])
    XZ_test = np.hstack([X_test, Z_test.reshape(-1, 1)])
    muE.fit(X=XZ_train, y=A_train)
    piE.fit(X=X_train, y=Z_train)

    XZ1 = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    XZ0 = np.hstack([X_train, np.zeros((X_train.shape[0], 1))])
    muz0 = muE.predict(XZ0)
    muz1 = muE.predict(XZ1)
    varz0 = (1-muz0)*muz0
    varz1 = (1-muz1)*muz1
    varA = varz1 - varz0
    eps = A_train.ravel() - muE.predict(XZ_train).ravel()
    piZ = Z_train.ravel() * piE.predict(X_train).ravel() + (1 - Z_train.ravel()) * (1 - piE.predict(X_train).ravel())
    
    phi1 = (2*Z_train-1)*eps*Y_train/(piZ*varA.ravel())
    betaE.fit(X=X_train, y=phi1)
    beta = betaE.predict(X_train).ravel()
    beta_test = betaE.predict(X_test).ravel()
    
    tauY = Y_train.ravel() - beta*A_train.ravel()
    tauE.fit(X=XZ_train, y=tauY)
    
    rhoY = eps*(Y_train.ravel()-beta*A_train.ravel())
    rhoE.fit(X=X_train, y=rhoY)
    
    point_CATE_1 = beta
    mse_CATE_1 = np.mean((beta - gamma_train) ** 2)
    point_CATE_test = beta_test
    mse_CATE_test = np.mean((beta_test - gamma_test) ** 2)
        
    est_CATE_final_1 = CustomGRF(criterion='het', muE=muE, piE=piE, betaE=betaE,
                                 tauE=tauE, rhoE=rhoE, **param_dict)
    est_CATE_final_1.fit(X_train, A_train, Y_train, Z=Z_train)
    point_CATE_final_1 = est_CATE_final_1.predict(X_train)
    mse_CATE_final_1 = np.mean((point_CATE_final_1[:, 0] - gamma_train) ** 2)

    point_CATE_final_test = est_CATE_final_1.predict(X_test)
    mse_CATE_final_test = np.mean((point_CATE_final_test[:, 0] - gamma_test) ** 2)

    if (datagen == 1) | (datagen == 2):  # model 1 & model 2
        gamma_detm = 2 * X_detm[:, 0] + 0.5 * X_detm[:, 1] + 0.5 * X_detm[:, 2]
    elif datagen == 3:  # model 3
        gamma_detm = 2 * X_detm[:, 0]**2 + 0.5 * X_detm[:, 1]
    
    point_CATE_detm, lb_CATE_detm, ub_CATE_detm = est_CATE_final_1.predict(X_detm, interval=True, alpha=0.05)
    CI = ub_CATE_detm[:, 0] - lb_CATE_detm[:, 0]
    # print(gamma_detm.ravel())
    # print(point_CATE_detm.ravel())
    # print(lb_CATE_detm.ravel())
    # print(ub_CATE_detm.ravel())
    cover = [1 if ((gamma_detm[i] <= ub_CATE_detm[:, 0][i]) & (gamma_detm[i] >= lb_CATE_detm[:, 0][i])) else 0 for i in range(len(gamma_detm))]

    results = {
        "gamma_train": gamma_train,
        "mean_gamma_train": np.mean(gamma_train),
        "gamma_test": gamma_test,
        "mean_gamma_test": np.mean(gamma_test),

        "pred_CATE_1": point_CATE_1,
        "pred_CATE_final": point_CATE_final_1[:, 0],
        "mean_pred_CATE_1": np.mean(point_CATE_1),
        "mean_pred_CATE_final": np.mean(point_CATE_final_1[:, 0]),
        "mse_CATE_1": mse_CATE_1,
        "mse_CATE_final": mse_CATE_final_1,

        "pred_CATE_test": point_CATE_test,
        "pred_CATE_final_test": point_CATE_final_test[:, 0],
        "mean_pred_CATE_test": np.mean(point_CATE_test),
        "mean_pred_CATE_final_test": np.mean(point_CATE_final_test[:, 0]),
        "mse_CATE_test": mse_CATE_test,
        "mse_CATE_final_test": mse_CATE_final_test,

        "CI": CI,
        "cover": cover,
        
        "true_mu": mu_test,
        "pred_mu": muE.predict(XZ_test),
        "true_pi": pi_test,
        "pred_pi": piE.predict(X_test),
        "true_tau": Y_test.ravel() - gamma_test*A_test.ravel(),
        "pred_tau": tauE.predict(XZ_test),
        "true_rho": (A_test.ravel() - mu_test)*(Y_test.ravel()-gamma_test*A_test.ravel()),
        "pred_rho": rhoE.predict(X_test),
        "true_eps": A_test.ravel() - mu_test,
        "pred_eps": A_test.ravel() - muE.predict(XZ_test).ravel()

    }
    return results


if __name__ == "__main__":
    n_test_samples = 50000  # Define based on your requirements
    n_estimators = 1000
    n_monte = 100
    X_detm = np.array([0.1]*5+[0.2]*5+[0.3]*5+[0.4]*5+[0.5]*5+[0.6]*5+[0.7]*5+[0.8]*5+[0.9]*5).reshape(9, 5)  # 测试点
    # Hyperparameters
    params = {'n_estimators': 1000, 'min_samples_leaf': 10, 'min_samples_split': 25,
               'max_depth': None, 'min_impurity_decrease': 0.0, 'max_samples': 0.5, 'min_balancedness_tol': 0.45,
               'subforest_size': 4, 'max_features': 'auto', 'min_var_leaf_on_val': False, 'inference': True,
               'warm_start': False, 'min_var_fraction_leaf': None}
    
    with open('GRF_paper_test.txt', 'w') as f:
        
        for n_train_samples in [5000, 10000, 15000]:
            # Example 1
            f.write(f'Running n={n_train_samples} & invalidIV (Model 2)\n')
            biasATE_list1 = []
            PEHE_list1 = []
            biasATE_list2 = []
            PEHE_list2 = []
            CI_list = []
            cover_list = []
            time_start = time.time()
            for i in range(n_monte):
                print(f"Running experiment {i + 1}/100...")
                f.write(f"Running experiment {i + 1}/100...\n")
                
                result = one_exp(params, seed=i+1234, datagen=2)
                
                bias1 = abs(result['mean_pred_CATE_test']-result['mean_gamma_test'])
                PEHE1 = result['mse_CATE_test']
                PEHE_list1.append(PEHE1)
                biasATE_list1.append(bias1)
                
                bias2 = abs(result['mean_pred_CATE_final_test']-result['mean_gamma_test'])
                PEHE2 = result['mse_CATE_final_test']
                PEHE_list2.append(PEHE2)
                biasATE_list2.append(bias2)
                
                CI = result['CI']
                cover = result['cover']
                CI_list.append(CI)
                cover_list.append(cover)
                
                print(f"{i}\t{bias1}\t{np.sqrt(PEHE1)}\t{bias2}\t{np.sqrt(PEHE2)}\t{CI}\t{cover}")
                f.write(f"{i}\t{bias1}\t{np.sqrt(PEHE1)}\t{bias2}\t{np.sqrt(PEHE2)}\t{CI}\t{cover}\n")
                
            time_end = time.time() 
            time_sum = time_end - time_start
            print(f'Results of n={n_train_samples} & invalidIV (Model 2)')
            print('time: %d s' % time_sum)
            print('time: %.2f min' % (time_sum/60))
            
            print('First Stage - Absolute Error of ATE: %.4f' %(np.mean(np.array(biasATE_list1))))
            print('First Stage - PEHE: %.4f' %(np.sqrt(np.mean(np.array(PEHE_list1)))))
            print('First Stage - RMSE of ATE: %.4f' %(np.sqrt(np.mean(np.array(biasATE_list1)**2))))
            
            print('Second Stage - Absolute Error of ATE: %.4f' %(np.mean(np.array(biasATE_list2))))
            print('Second Stage - PEHE: %.4f' %(np.sqrt(np.mean(np.array(PEHE_list2)))))
            print('Second Stage - RMSE of ATE: %.4f' %(np.sqrt(np.mean(np.array(biasATE_list2)**2))))
            
            CI_mean = []
            for i in range(len(CI_list[0])):
                CI_mean.append(round(np.mean([x[i] for x in CI_list]), 4))
            
            prop_mean = []
            for i in range(len(CI_list[0])):
                prop_mean.append(round(np.mean([x[i] for x in cover_list]), 4))
            print('Average length of CI: ', CI_mean)
            print('Cover prop of CI: ', prop_mean)
            
            f.write(f'Results of n={n_train_samples} & invalidIV (Model 2)\n')
            f.write(f'time: {round(time_sum)} s\n')
            f.write(f'time: {round(time_sum)/60} min\n')
            f.write(f'First Stage - Absolute Error of ATE: {round(np.mean(np.array(biasATE_list1)), 4)}\n')
            f.write(f'First Stage - PEHE: {round(np.sqrt(np.mean(np.array(PEHE_list1))), 4)}\n')
            f.write(f'First Stage - RMSE of ATE: {round(np.sqrt(np.mean(np.array(biasATE_list1)**2)), 4)}\n')
            f.write(f'Second Stage - Absolute Error of ATE: {round(np.mean(np.array(biasATE_list2)), 4)}\n')
            f.write(f'Second Stage - PEHE: {round(np.sqrt(np.mean(np.array(PEHE_list2))), 4)}\n')
            f.write(f'Second Stage - RMSE of ATE: {round(np.sqrt(np.mean(np.array(biasATE_list2)**2)), 4)}\n')
            f.write(f'Average length of CI: {CI_mean}\n')
            f.write(f'Cover prop of CI: {prop_mean}\n')
            f.write('\n')

            # Example 2
            f.write(f'Running n={n_train_samples} & invalidIV_nonlinear (Model 3)\n')
            biasATE_list1 = []
            PEHE_list1 = []
            biasATE_list2 = []
            PEHE_list2 = []
            CI_list = []
            cover_list = []
            time_start = time.time()
            for i in range(n_monte):
                print(f"Running experiment {i + 1}/100...")
                f.write(f"Running experiment {i + 1}/100...\n")
                
                result = one_exp(params, seed=i+1234, datagen=3)
                
                bias1 = abs(result['mean_pred_CATE_test']-result['mean_gamma_test'])
                PEHE1 = result['mse_CATE_test']
                PEHE_list1.append(PEHE1)
                biasATE_list1.append(bias1)
                
                bias2 = abs(result['mean_pred_CATE_final_test']-result['mean_gamma_test'])
                PEHE2 = result['mse_CATE_final_test']
                PEHE_list2.append(PEHE2)
                biasATE_list2.append(bias2)
                
                CI = result['CI']
                cover = result['cover']
                CI_list.append(CI)
                cover_list.append(cover)
                
                print(f"{i}\t{bias1}\t{np.sqrt(PEHE1)}\t{bias2}\t{np.sqrt(PEHE2)}\t{CI}\t{cover}")
                f.write(f"{i}\t{bias1}\t{np.sqrt(PEHE1)}\t{bias2}\t{np.sqrt(PEHE2)}\t{CI}\t{cover}\n")
                
            time_end = time.time()
            time_sum = time_end - time_start
            print(f'Results of n={n_train_samples} & invalidIV_nonlinear (Model 3)')
            print('time: %d s' % time_sum)
            print('time: %.2f min' % (time_sum/60))
            
            print('First Stage - Absolute Error of ATE: %.4f' %(np.mean(np.array(biasATE_list1))))
            print('First Stage - PEHE: %.4f' %(np.sqrt(np.mean(np.array(PEHE_list1)))))
            print('First Stage - RMSE of ATE: %.4f' %(np.sqrt(np.mean(np.array(biasATE_list1)**2))))
            
            print('Second Stage - Absolute Error of ATE: %.4f' %(np.mean(np.array(biasATE_list2))))
            print('Second Stage - PEHE: %.4f' %(np.sqrt(np.mean(np.array(PEHE_list2)))))
            print('Second Stage - RMSE of ATE: %.4f' %(np.sqrt(np.mean(np.array(biasATE_list2)**2))))
            
            CI_mean = []
            for i in range(len(CI_list[0])):
                CI_mean.append(round(np.mean([x[i] for x in CI_list]), 4))
            
            prop_mean = []
            for i in range(len(CI_list[0])):
                prop_mean.append(round(np.mean([x[i] for x in cover_list]), 4))
            print('Average length of CI: ', CI_mean)
            print('Cover prop of CI: ', prop_mean)
            
            f.write(f'Results of n={n_train_samples} & invalidIV_nonlinear (Model 3)\n')
            f.write(f'time: {round(time_sum)} s\n')
            f.write(f'time: {round(time_sum)/60} min\n')
            f.write(f'First Stage - Absolute Error of ATE: {round(np.mean(np.array(biasATE_list1)), 4)}\n')
            f.write(f'First Stage - PEHE: {round(np.sqrt(np.mean(np.array(PEHE_list1))), 4)}\n')
            f.write(f'First Stage - RMSE of ATE: {round(np.sqrt(np.mean(np.array(biasATE_list1)**2)), 4)}\n')
            f.write(f'Second Stage - Absolute Error of ATE: {round(np.mean(np.array(biasATE_list2)), 4)}\n')
            f.write(f'Second Stage - PEHE: {round(np.sqrt(np.mean(np.array(PEHE_list2))), 4)}\n')
            f.write(f'Second Stage - RMSE of ATE: {round(np.sqrt(np.mean(np.array(biasATE_list2)**2)), 4)}\n')
            f.write(f'Average length of CI: {CI_mean}\n')
            f.write(f'Cover prop of CI: {prop_mean}\n')
            f.write('\n')