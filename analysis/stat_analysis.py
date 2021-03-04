#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

import numpy as np
import scipy.stats as ss

def jonckheere(data):
    """
    Implements Jonckheere's trend test. It will test for a positive or
    negative trend in the number of treatments. The result is a statistic z
    for which significance testing is done against the standard normal
    distribution. This function requires the same number of samples for each 
    treatment.
    
    Parameters
    ----------
    data : num_samples x num_treatments numpy array
        The observed data with rows corresponding to independent samples and 
        columns corresponding to the ordered treatments that are being
        tested for a trend.
    
    Returns
    -------
    z : float
        The test statistic, normalized to be approximately distributed
        standard normal
    p : float
        The p-value for a one-sided tail test of z under the assumption of a
        standard normal distribution
    """
    
    num_samples, num_treatments = data.shape
    # Compute S via the direct method
    P = 0
    Q = 0
    for i in range(num_samples):
        for j in range(num_treatments-1):
            P += np.sum(data[:, (j+1):] >= data[i, j])
            Q += np.sum(data[:, (j+1):] < data[i, j])
    S = P-Q
    # Normalize and correct S
    var = (2 * ((num_samples * num_treatments)**3 - num_treatments * num_samples**3) + 3 * ((num_samples * num_treatments)**2 - num_treatments * num_samples**2)) / 18
    z = (np.abs(S) - 1) / np.sqrt(var)
    p = ss.norm.sf(z)
    return z, p, S

# # Example implementation of Jonckheere test
# results = np.load('../RelativeErrorVsHorizon.npz')
# ic_hzn, true_risk_param, pred_risk_param, nll = results['ic_hzn'], results['true_risk_param'], results['pred_risk_param'], results['nll']
# pred_risk_param = pred_risk_param[:, :, 0]
# z, p = jonckheere(np.abs(pred_risk_param-true_risk_param))

def unbiased(data):
    """
    Implements a one-sample t-test for each of several ordered treatments in
    order to assess unbiasedness. The result is a test statistic and p-value 
    for each treatment. This function requires an equal number of samples for
    each treatment.

    Parameters
    ----------
    data : num_samples x num_treatments numpy array
        The observed data with rows corresponding to independent samples and 
        columns corresponding to the ordered treatments that are 
        independently being tested for unbiasedness

    Returns
    -------
    t : num_treatments numpy array
        The resulting test statistic for each treatment
    p : num_treatments numpy array
        The p-value for each treatment 
    """
    
    t, p = ss.stats.ttest_1samp(data, 0, axis=0) # columns are each horizon and rows are each true risk sample
    return t, p

# # Example implementation of unbiasedness test
# results = np.load('../RelativeErrorVsHorizon.npz')
# ic_hzn, true_risk_param, pred_risk_param, nll = results['ic_hzn'], results['true_risk_param'], results['pred_risk_param'], results['nll']
# pred_risk_param = pred_risk_param[:, :, 0]
# t, p = unbiased(pred_risk_param-true_risk_param)      



results = np.load('Expanded_Domain_Horizon-20210114.npz')
ic_hzn, true_risk_param, pred_risk_param, nll = results['ic_hzn'], results['true_risk_param'], results['pred_risk_param'], results['nll']

pred_risk_param = np.transpose(np.array(pred_risk_param))
print("Expanded_Domain pred_risk_param: ")
print(pred_risk_param)

true_risk_param = np.transpose(np.array(true_risk_param))
print("Expanded_Domain true_risk_param: ")
print(true_risk_param)

z, p, S = jonckheere(np.abs(pred_risk_param-true_risk_param))
print("Expanded_Domain jonckheere: " + "z: "+ str(z) + ",p: "+ str(p) + ",S: "+ str(S))

t, p = unbiased(pred_risk_param-true_risk_param)        
print("Expanded_Domain unbiased: " + "t: "+ str(t) + ",p: "+ str(p))  



# results = np.load('Idealized_Domain_Horizon_5_30.npz')
# ic_hzn, true_risk_param, pred_risk_param, nll = results['ic_hzn'], results['true_risk_param'], results['pred_risk_param'], results['nll']

# pred_risk_param = np.transpose(np.array(pred_risk_param))[0]
# print("Idealized_Domain pred_risk_param: ")
# print(pred_risk_param)

# true_risk_param = np.transpose(np.array(true_risk_param))
# print("Idealized_Domain true_risk_param: ")
# print(true_risk_param)

# z, p, S = jonckheere(np.abs(pred_risk_param-true_risk_param))
# print("Idealized_Domain jonckheere: " + "z: "+ str(z) + ",p: "+ str(p) + ",S: "+ str(S))

# t, p = unbiased(pred_risk_param-true_risk_param)        
# print("Idealized_Domain unbiased: " + "t: "+ str(t) + ",p: "+ str(p))  
