import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = np.load('PrudentVsMyopic_horizon_30.npz')

true_risk = data['true_risk']
#print(true_risk)

# (param_pp[0],param_pm[0],param_mp[0],param_mm[0])
pred_risk = data['pred_risk']
#print(pred_risk)

hzn = data['hzn']
nll = data['nll']
#print(hzn)

myopic_bi=0
prudent_bi = 1

pp_df = pd.DataFrame()
pp_df['true'] = true_risk
pp_df['pred'] = [x[0] for x in pred_risk]
pp_df['hzn'] = hzn
pp_df['nll'] = [x[0] for x in nll]
pp_df['model_sim_traj'] = [prudent_bi for x in range(len(hzn))]
pp_df['model_sel_agent'] = [prudent_bi for x in range(len(hzn))]
pp_df['raw_error'] = pp_df['pred']-pp_df['true']

pm_df = pd.DataFrame()
pm_df['true'] = true_risk
pm_df['pred'] = [x[1] for x in pred_risk]
pm_df['hzn'] = hzn
pm_df['nll'] = [x[1] for x in nll]
# pm_df['type'] = ["prudent,myopic" for x in range(len(hzn))]
pm_df['model_sim_traj'] = [prudent_bi for x in range(len(hzn))]
pm_df['model_sel_agent'] = [myopic_bi for x in range(len(hzn))]
pm_df['raw_error'] = pm_df['pred']-pm_df['true']

mp_df = pd.DataFrame()
mp_df['true'] = true_risk
mp_df['pred'] = [x[2] for x in pred_risk]
mp_df['hzn'] = hzn
mp_df['nll'] = [x[2] for x in nll]
# mp_df['type'] = ["myopic,prudent" for x in range(len(hzn))]
mp_df['model_sim_traj'] = [myopic_bi for x in range(len(hzn))]
mp_df['model_sel_agent'] = [prudent_bi for x in range(len(hzn))]
mp_df['raw_error'] = mp_df['pred']-mp_df['true']

mm_df = pd.DataFrame()
mm_df['true'] = true_risk
mm_df['pred'] = [x[3] for x in pred_risk]
mm_df['hzn'] = hzn
mm_df['nll'] = [x[3] for x in nll]
# mm_df['type'] = ["myopic,myopic" for x in range(len(hzn))]
mm_df['model_sim_traj'] = [myopic_bi for x in range(len(hzn))]
mm_df['model_sel_agent'] = [myopic_bi for x in range(len(hzn))]
mm_df['raw_error'] = mm_df['pred']-mm_df['true']


df_full = pd.DataFrame()
df_full = pd.concat([pp_df,pm_df,mp_df,mm_df])
df_full

df_full.to_csv('horizon_30_raw_error_PM.csv',index=False)
