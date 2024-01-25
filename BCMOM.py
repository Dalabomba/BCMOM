# Packages requiries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from rich.progress import track
from tqdm import tqdm
from scipy.stats import norm
from multiprocessing import Pool
from sympy import symbols, Eq, solve



# Data Generation
def generate_multi_lognormal_data(n_workers=10, n_samples_eachworker=1000, dim=10, mu=0, sigma=2, alpha=0, bzmean=0, bzsd=np.sqrt(200)):
    data = np.random.lognormal(mean=mu, sigma=sigma, size=(n_workers, n_samples_eachworker, dim))
    bzmachine = np.floor((n_workers-1)*alpha).astype(int)
    data[1:bzmachine+1] = np.random.normal(loc=bzmean, scale=bzsd, size=(bzmachine, n_samples_eachworker, dim))
    true_mean = np.repeat(np.exp(mu+sigma**2/2), dim)
    true_sd = np.repeat(np.sqrt((np.exp(sigma**2)-1) * np.exp(2*mu+ sigma**2)), dim)
    return data, true_mean, true_sd


def generate_multi_gamma_data(n_workers=10, n_samples_eachworker=1000, dim=10, shape=1, scale=1, alpha=0, bzmean=0, bzsd=np.sqrt(200)):
    # 生成一个m*n*1的三维数组，注意不是二维的！
    data = np.random.gamma(shape=shape, scale=scale, size=(n_workers, n_samples_eachworker, dim))
    bzmachine = np.floor((n_workers-1)*alpha).astype(int)
    data[1:bzmachine+1] = np.random.normal(loc=bzmean, scale=bzsd, size=(bzmachine, n_samples_eachworker, dim))
    true_mean = np.repeat(shape*scale, dim)
    true_sd = np.repeat(np.sqrt(shape*scale**2), dim)
    return data, true_mean, true_sd

def generate_multi_normal_data(n_workers=10, n_samples_eachworker=1000, dim=10, mean=0, sd=1, alpha=0, bzmean=0, bzsd=np.sqrt(200)):
    # 生成一个m*n*1的三维数组，注意不是二维的！
    data = np.random.normal(scale=1, size=(n_workers, n_samples_eachworker, dim))
    bzmachine = np.floor((n_workers-1)*alpha).astype(int)
    data[1:bzmachine+1] = np.random.normal(loc=bzmean, scale=bzsd, size=(bzmachine, n_samples_eachworker, dim))
    mu = np.repeat(mean, dim)
    true_sd = np.repeat(np.sqrt(sd**2), dim)
    return data, mu ,true_sd



# Define a class to calculate MOM, VRMOM and BCMOM
class AllEstimators:
    def __init__(self, data):
        self.data = data
        self.n_workers, self.n_samples_eachworker, self.dim = self.data.shape
        self.initial_estimation = self.get_MOM_estimation()

    def get_edgeworth_pdf(self, data):
        f_new = norm.pdf(data)*(1+2/(np.sqrt(self.n_samples_eachworker)*3)*self.skewness*data) - 1/np.sqrt(self.n_samples_eachworker*2*np.pi)*np.exp(-data**2/2)*data*(2*(data**2)+1)/6*self.skewness
        return f_new
    
    def get_edgeworth_expansion(self, x):
        F_x = norm.cdf(x) + 1/np.sqrt(self.n_samples_eachworker) * norm.pdf(x)*(2* (x**2) +1)/6*self.skewness
        return F_x
    
    def get_to_edgeworth_expansion(self, x):
        F_x = self.get_edgeworth_expansion(x) - 1/self.n_samples_eachworker*x*norm.pdf(x)*(self.skewness**2/18*(x**4 + 2*x**2 -3) - self.kurtosis/12*(x**2-3)+1/4*(x**2+3))
        return F_x
    
    def get_third_order_edgeworth_pdf(self, data):
        return norm.pdf(data) * (1 + data / np.sqrt(self.n_samples_eachworker) * self.skewness * (1/2 - data**2/3)
                            + 1 / np.sqrt(self.n_samples_eachworker) * (
                                (data**2-1) * (self.skewness**2/18*(data**4+2*data**2-3) - (self.kurtosis)/12*(data**2-3) + 0.25*(data**2+3))
                                - data * (2/9*self.skewness**2*(data**3+data) - ((self.kurtosis)/6 - 0.5)*data)
                            )
                            )
    
    def get_third_order_cornish_fisher_expanison_for_t_stat(self, q):
        x_norm = norm.ppf(q)
        X = x_norm -1/np.sqrt(self.n_samples_eachworker)*(2*(x_norm**2)+1)/6*self.skewness\
            + 1/self.n_samples_eachworker * (x_norm*((self.kurtosis+3)/4 - (5*self.skewness**2)/72) + (x_norm**3) * (1/2 + (11*self.skewness**2)/36 - (self.kurtosis+3)/12) + (x_norm**5) * ((5*self.skewness**2)/18) )
        return X
    
    def get_cornish_fisher_expansion(self, q):
        x_norm = norm.ppf(q)
        X = x_norm -1/np.sqrt(self.n_samples_eachworker)*(2*(x_norm**2)+1)/6*self.skewness
        return X

    def get_MOM_estimation(self):
        MOM_estimation = np.mean(self.data, axis=1)
        return np.median(MOM_estimation, axis=0)
    
    def get_VRMOM_estimation(self, K=10, iter=False, tolerance=1e-5, max_iter=100):
        VRMOM_estimation = self.initial_estimation
        sigma_hat = np.std(self.data[0], axis=0, ddof=0)
        list_k = np.array(list(range(1, K+1)))/(K+1)
        delta_k = norm.ppf(list_k)
        pdf = norm.pdf(delta_k)
        h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
        if not iter:
            max_iter = 1
        for i in range(max_iter):
            initial_estimation = VRMOM_estimation
            # sigma_hat = np.mean((self.data[0]-initial_estimation)**2)
            data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
            VRMOM_estimation = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*norm.cdf(data_new)), axis=0)
            if np.linalg.norm(VRMOM_estimation-initial_estimation) < tolerance:
                break
            # if i == max_iter-1:
            #     print("Reach max iteration!!!")
        # self.VRMOM_estimation = VRMOM_estimation
        return VRMOM_estimation
    
    def get_EVRMOM_estimation(self, K=10, iter=False, tolerance=1e-5, max_iter=100, setting = [1,1]):
        EVRMOM = self.initial_estimation
        _ = self.data - np.mean(self.data, axis=1, keepdims=True)
        sigma_hat = np.median(np.sqrt((_**2).mean(axis=1).squeeze()), axis=0)
        mu_3_hat = np.median((_**3).mean(axis=1).squeeze(), axis=0)
        mu_4_hat = np.median((_**4).mean(axis=1).squeeze(), axis=0)
        self.skewness = mu_3_hat/sigma_hat**3
        self.kurtosis = mu_4_hat/sigma_hat**4 - 3
        list_k = np.array(list(range(1, K+1)))/(K+1)

        if setting[0] == 0:
            delta_k = norm.ppf(list_k)
            pdf = norm.pdf(delta_k)
        elif setting[0] == 1:
            delta_k = self.get_cornish_fisher_expansion(list_k)
            pdf = self.get_edgeworth_pdf(delta_k)
        elif setting[0] == 2:
            delta_k = self.get_third_order_cornish_fisher_expanison_for_t_stat(list_k)
            pdf = self.get_third_order_edgeworth_pdf(delta_k)

        if not iter:
            max_iter = 1
        for i in range(max_iter):
            initial_estimation = EVRMOM
            if setting[1] == 0:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*norm.cdf(data_new)), axis=0)
            elif setting[1] == 1:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*self.get_edgeworth_expansion(data_new)), axis=0)
            elif setting[1] == 2:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*self.get_to_edgeworth_expansion(data_new)), axis=0)
            
            if np.linalg.norm(EVRMOM-initial_estimation) < tolerance:
                break
        return EVRMOM 
    
    def get_EVRMOM_estimation_QC(self, K=10, iter=False, tolerance=1e-5, max_iter=100, setting = [1,1], ql=0.1, qu=0.9, th=np.inf):
        EVRMOM = self.initial_estimation
        w = np.abs(self.data)< th
        _ = self.data - np.mean(self.data, axis=1, keepdims=True)
        sigma_hat = np.median(np.sqrt((_**2).mean(axis=1).squeeze()), axis=0)
        sigma_new = np.std(self.data, where=w, axis=1)
        mu_3_hat = np.median((_**3).mean(axis=1).squeeze(), axis=0)
        mu_4_hat = np.median((_**4).mean(axis=1).squeeze(), axis=0)
        self.skewness = mu_3_hat/sigma_hat**3
        self.kurtosis = mu_4_hat/sigma_hat**4 - 3
        list_k = ql + (qu-ql)*np.array(list(range(1, K+1)))/(K+1)

        if setting[0] == 0:
            delta_k = norm.ppf(list_k)
            pdf = norm.pdf(delta_k)
        elif setting[0] == 1:
            delta_k = self.get_cornish_fisher_expansion(list_k)
            pdf = self.get_edgeworth_pdf(delta_k)
        elif setting[0] == 2:
            delta_k = self.get_third_order_cornish_fisher_expanison_for_t_stat(list_k)
            pdf = self.get_third_order_edgeworth_pdf(delta_k)

        if not iter:
            max_iter = 1
        for i in range(max_iter):
            initial_estimation = EVRMOM
            if setting[1] == 0:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K*(1-(ql+qu)/2)+1-np.ceil((K+1)*(norm.cdf(data_new)-ql)/(qu-ql)), axis=0)
            elif setting[1] == 1:
                h_x = 1/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_new
                EVRMOM = initial_estimation - h_x * np.sum(sigma_hat*(K/2+1-np.ceil((K+1)*self.get_edgeworth_expansion(data_new))), axis=0)
            elif setting[1] == 2:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K*(1-(ql+qu)/2)+1-np.ceil((K+1)*(self.get_to_edgeworth_expansion(data_new)-ql)/(qu-ql)), axis=0)

            if np.linalg.norm(EVRMOM-initial_estimation) < tolerance:
                break
        return EVRMOM 

        
    def get_EVRMOM_estimation_new(self, K=10, iter=False, tolerance=1e-5, max_iter=100, setting = [1,1], th = np.inf):
        EVRMOM = self.initial_estimation
        w = np.abs(self.data)< th
        _ = self.data - np.mean(self.data, axis=1, keepdims=True)
        sigma_hat = np.median(np.sqrt((_**2).mean(axis=1).squeeze()), axis=0)
        sigma_new = np.std(self.data, where=w, axis=1)
        mu_3_hat = np.median((_**3).mean(axis=1).squeeze(), axis=0)
        mu_4_hat = np.median((_**4).mean(axis=1).squeeze(), axis=0)
        self.skewness = mu_3_hat/sigma_hat**3
        self.kurtosis = mu_4_hat/sigma_hat**4 - 3
        list_k = np.array(list(range(1, K+1)))/(K+1)

        if setting[0] == 0:
            delta_k = norm.ppf(list_k)
            pdf = norm.pdf(delta_k)
        elif setting[0] == 1:
            delta_k = self.get_cornish_fisher_expansion(list_k)
            pdf = self.get_edgeworth_pdf(delta_k)
        elif setting[0] == 2:
            delta_k = self.get_third_order_cornish_fisher_expanison_for_t_stat(list_k)
            pdf = self.get_third_order_edgeworth_pdf(delta_k)

        if not iter:
            max_iter = 1
        for i in range(max_iter):
            initial_estimation = EVRMOM
            if setting[1] == 0:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*norm.cdf(data_new)), axis=0)
            elif setting[1] == 1:
                h_x = 1/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_new
                EVRMOM = initial_estimation - h_x * np.sum(sigma_hat*(K/2+1-np.ceil((K+1)*self.get_edgeworth_expansion(data_new))), axis=0)
            elif setting[1] == 2:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_new
                EVRMOM = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*self.get_to_edgeworth_expansion(data_new)), axis=0)
            
            if np.linalg.norm(EVRMOM-initial_estimation) < tolerance:
                break
        return EVRMOM
    
    
    def get_EVRMOM_estimation_attack(self, K=10, iter=False, tolerance=1e-5, max_iter=100, setting = [1,1], th = np.inf):
        EVRMOM = self.initial_estimation
        w = self.data< th
        _ = self.data - np.mean(self.data, axis=1, keepdims=True)
        sigma_hat = np.median(np.sqrt((_**2).mean(axis=1).squeeze()), axis=0)
        sigma_new = np.std(self.data, where=w, axis=1)
        mu_3_hat = np.median((_**3).mean(axis=1).squeeze(), axis=0)
        mu_4_hat = np.median((_**4).mean(axis=1).squeeze(), axis=0)
        self.skewness = mu_3_hat/sigma_hat**3
        self.kurtosis = mu_4_hat/sigma_hat**4 - 3
        list_k = np.array(list(range(1, K+1)))/(K+1)

        if setting[0] == 0:
            delta_k = norm.ppf(list_k)
            pdf = norm.pdf(delta_k)
        elif setting[0] == 1:
            delta_k = self.get_cornish_fisher_expansion(list_k)
            pdf = self.get_edgeworth_pdf(delta_k)
        elif setting[0] == 2:
            delta_k = self.get_third_order_cornish_fisher_expanison_for_t_stat(list_k)
            pdf = self.get_third_order_edgeworth_pdf(delta_k)

        if not iter:
            max_iter = 1
        for i in range(max_iter):
            initial_estimation = EVRMOM
            if setting[1] == 0:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*norm.cdf(data_new)), axis=0)
            elif setting[1] == 1:
                h_x = 1/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_new
                EVRMOM = initial_estimation - h_x * np.sum(sigma_hat*(K/2+1-np.ceil((K+1)*self.get_edgeworth_expansion(data_new))), axis=0)
            elif setting[1] == 2:
                h_x = sigma_hat/(self.n_workers*np.sqrt(self.n_samples_eachworker)*np.sum(pdf))
                data_new = np.sqrt(self.n_samples_eachworker)*(self.data.mean(axis=1)-initial_estimation)/sigma_hat
                EVRMOM = initial_estimation - h_x * np.sum(K/2+1-np.ceil((K+1)*self.get_to_edgeworth_expansion(data_new)), axis=0)
            
            if np.linalg.norm(EVRMOM-initial_estimation) < tolerance:
                break
        return EVRMOM

        

def find_inverse_binary(f, y, x_min, x_max, tolerance=1e-6, max_iterations=100):
    if f(x_min) > y or f(x_max) < y:
        return None

    for _ in range(max_iterations):
        x_mid = (x_min + x_max) / 2
        y_mid = f(x_mid)

        if abs(y_mid - y) < tolerance:
            return x_mid

        if y_mid < y:
            x_min = x_mid
        else:
            x_max = x_mid

    return None

def get_mean_of_lognormal(mu,sigma):
    return np.exp(mu+sigma**2/2)

def get_variance_of_lognormal(mu,sigma):
    return (np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2)

def get_skewness_of_lognormal(sigma):
    return (np.exp(sigma**2)+2)*np.sqrt(np.exp(sigma**2)-1)

def get_kurtosis_of_lognormal(sigma):
    return np.exp(4*sigma**2)+2*np.exp(3*sigma**2)+3*np.exp(2*sigma**2)-6

def get_mean_of_gamma(shape, scale):
    return shape*scale

def get_variance_of_gamma(shape, scale):
    return shape*scale**2

def get_skewness_of_gamma(shape):
    return 2/np.sqrt(shape)

def get_kurtosis_of_gamma(shape):
    return 6/shape
