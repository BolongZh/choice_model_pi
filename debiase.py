from prediction import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
pd.set_option("display.precision", 4)
from scipy.stats import norm
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn'



def moment_func(x_1, x_2, model, delta):
    # apply price change 
    new_x_1 = x_1.clone()
    new_x_1[:,:,-1] = new_x_1[:,:,-1] * (1+delta)  ## in data generation, the last feature is price
    
    # calculate moment
    a1 = model(x_2, new_x_1)
    a0 = model(x_2, x_1)
    moment = a1 - a0
    
    return moment, a0
    
    
def alpha_loss(x_1, x_2, model, delta): 
    moment, alpha = moment_func(x_1, x_2, model, delta)
    return (alpha**2 - moment * 2).mean()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_alpha(data, delta):
    K = data['K']
    x_1, x_2 = x_transform_mm(data)
    model = SmallDeepSet(x_d = K+1)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    losses = []
    x_1, x_2 = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda()

    for _ in range(5000):
        loss = alpha_loss(x_1, x_2, model, delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if loss < 1e-10:
            break
        #print(loss)
    return model, losses


def pred_theta(f_model, alpha_model, data, delta):
    x_1, x_2 = x_transform_mm(data) 
    y = data['Y']
    x_1, x_2, y = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda(), torch.from_numpy(y).float().cuda()
    
    ## the effect of price change on market share estimated using f_model 
    new_x_1 = x_1.clone()
    new_x_1[:,:,-1] = new_x_1[:,:,-1] * (1+delta)  ## in data generation, the last feature is price
    # calculate change in market share
    y1 = f_model(x_2, new_x_1)
    y0 = f_model(x_2, x_1)
    m_est = y1-y0
    ## alpha hat 
    alpha_hat = alpha_model(x_2, x_1) 
    theta = m_est + alpha_hat * (y-y0)
    
    return m_est, theta, alpha_hat, y0


def Inference(J, M, K, seed, dg, params,  delta = 0.01):
    
    setup = [J, K, M, str(dg).split()[1], seed]
    ### step 1: data generation 
    datax = data_generation(params, J, K, M, seed, dg)
    
    data_f1, data_f2 = split_train_test(datax, p = 0.5)
    
    ### step 2: estimation by each fold of data 
    ## estimation of fold 2
    ## train on the left out data
    f_model, loss_f = train_deep(data_f1)
    alpha_model, loss_alpha = train_alpha(data_f1, delta)
    ## apply the trained f and alpha to the fold
    m_est1, theta1, alpha_hat1, f_hat1 = pred_theta(f_model, alpha_model, data_f2, delta)
    
    ## estimation of fold 1
    ## train on the left out data
    f_model, loss_f = train_deep(data_f2)
    alpha_model, loss_alpha = train_alpha(data_f2, delta)
    ## apply the trained f and alpha to the fold
    m_est2, theta2, alpha_hat2, f_hat2 = pred_theta(f_model, alpha_model, data_f1, delta)
    
    ### step 3: report the estimation result
    theta = torch.cat((theta1, theta2), dim=0)
    theta_hat = theta.mean()
    theta_sd = theta.std()
    
    return theta_hat.cpu().detach().numpy(), theta_sd.cpu().detach().numpy()


def true_theta(J, M, K, seed, dg, params,  delta = 0.01): 
    
    data = data_generation(params, J, K, M * 20, seed, dg)
    
    large_record = pd.DataFrame()

    for prod_id in range(J):
        record = cal_true_share_change(data, dg, prod_id, delta, seed)
        record['i'] = prod_id
        record['j'] = record.index % data['J']
        large_record = pd.concat([large_record, record]).reset_index(drop=True)

    true = large_record.loc[large_record.i == large_record.j,'true_change'].mean()
    
    return true

def PlugIn_theta(J, M, K, seed, dg, params, delta = 0.01):
    
    setup = [J, K, M, str(dg).split()[1], seed]
    ### step 1: data generation 
    datax = data_generation(params, J, K, M, seed, dg)
    
    ### step 2: train
    m1_deep, losses_deep = train_deep(datax)
    
    ### step 3: plug-in
    theta_deep = pred_theta_nc(datax, pred_deep, seed,  m1_deep,  delta)
    
    return [theta_deep] 




def pred_theta_nc(data, pred_method, seed, new_params, delta =0.01):
    large_record = pd.DataFrame()
    J = data['J']
    old_pred = pred_method(data, new_params)

    for prod_id in range(J):        
        X = data['X'].copy()
        M = data['M']

        ## update price
        record = pd.DataFrame({'old_price' : X['price'].copy()})

        for m in range(M):
            new_id = prod_id + m * J
            X['price'].iloc[new_id] = (1 + delta) * X['price'].iloc[new_id]

        new_data = data.copy()
        new_data['X'] = X

        ## pred y 
        new_pred = pred_method(new_data,new_params)

        record['new_pred_share'] = new_pred
        record['old_pred_share'] = old_pred

        record['pred_change'] = record['new_pred_share'] - record['old_pred_share']

        record['i'] = prod_id
        record['j'] = record.index % data['J']
        large_record = pd.concat([large_record, record]).reset_index(drop=True)

    true = large_record.loc[large_record.i == large_record.j,'pred_change'].mean()
    true_sd = large_record.loc[large_record.i == large_record.j,'pred_change'].std()
    
    return [true, true_sd]


def cal_cover(true, theta_hat, theta_sd, J, M):
    ci_l = theta_hat - 1.96 * theta_sd / np.sqrt(J * M)
    ci_u = theta_hat + 1.96 * theta_sd / np.sqrt(J * M)
    cover = (ci_l < true) & (ci_u >  true)
    significant = ci_u < 0
    bias = np.abs(theta_hat - true)
    
    return np.mean(bias), np.mean(cover), np.mean(significant)