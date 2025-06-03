import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn

# =============================================================
# 1.  DATA‑GENERATION UTILITIES
# =============================================================

def feature_generation(J, K, M, seed):
    # J is number of products per markets (assume each market has the same number of products but different products)
    # K is number of features (excluding price)
    # M is the number of markets (assume each market has different sets of products)
    np.random.seed(seed)

    Total_J = J * M
    x_1 = np.random.normal(loc=0, scale=1, size=(Total_J , K))
    price = np.random.uniform(0,4,size = Total_J)
    #price = np.abs(np.random.normal(loc=0, scale=1, size = Total_J))
    
    X = pd.DataFrame(x_1)
    X['price'] = price

    return X 

def market_id_gen(J, M):
    Total_J = J * M
    J_list = [J] * M
    ## create a column as market id
    market_id = np.ones(Total_J)
    start = 0 
    for m in range(M):
        j_m = J_list[m]
        market_id[start:start + j_m] = m
        start = start + j_m
        
    return market_id

def mnl(X, b, J, K, M, seed):
    # seed is not used in this function
    # b are the parameters 
    Total_J = J * M
    
    ## get the share of each product (y)
    u = b[0] * np.ones(Total_J)
    for i in range(1,K+2):
        u = u + b[i] * X.iloc[:,i-1] 

    X['u'] = np.exp(u)    
    market_id = market_id_gen(J, M)
    X['market_id'] = market_id
    X['sum_u'] = X.groupby(['market_id'])['u'].transform('sum') 

    #Y = np.log(X['u'] / (X['sum_u'] + 1))
    Y = X['u'] / (X['sum_u'] + 1)

    del X['u']
    del X['market_id']
    del X['sum_u']

    data = {'X': X, 'Y': Y.to_numpy(),  'M': M, "J": J, "K": K, 'params':b, 'generation_seed': seed,'market_id':market_id}

    return data

# --- Random‑coefficient logit (rcl) ---------------------------------------

def rcl(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    st1 = time.time()
    for i in range(len(b)):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    st2 = time.time()
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    #u_i_max = np.max(u_i, axis=0, keepdims=True)  # shape: (1, N)
    #u_i_stable = u_i - u_i_max  
    exp_u_i = np.exp(u_i)  
    u_m = exp_u_i.reshape(M, J, N)
    sum_u_m = np.sum(u_m, axis=1, keepdims=True) 
    ccp_m = u_m / (sum_u_m + 1 )
    
    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}

    return data

def rcl_regenerate(X, data, N=10000):
    J = data['J']
    K = data['K']
    M = data['M']
    b_random = data['b_random']
    
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    #u_i = np.max(u_i, axis=0, keepdims=True)  # shape: (1, N)
    #u_i_stable = u_i - u_i_max  # subtract max per individual
    exp_u_i = np.exp(u_i)
    
    Y = np.zeros((M* J))
    
    # Reshape exp_u_i from (M*J, N) to (M, J, N)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    

    return Y


def data_generation(params, J, K, M, seed, x_to_y):
    X = feature_generation(J, K, M, seed)
    data = x_to_y(X, params, J, K, M, seed)
    #X_test = feature_generation(J, K, M, 1234)
    #data_test = x_to_y(X_test, params, J, K, M, 1234)
    
    #data['X_test'] = data_test['X'].copy()
    #data['Y_test'] = data_test['Y'].copy()
    return data


# =============================================================
# 2.  PRE‑PROCESSING & NN ARCHITECTURE
# =============================================================

def x_transform_mm(data):
    ### Two inputs for the nn
    X = data['X']
    J = data['J']
    M = data['M']
    K = data['K']
    ## x_1: the focal product's features; shape should be (number of products * number of markets, 1, number of features)
    Total_J = J * M
    X_1 = np.ones((Total_J,1,K+1))
    for i in range(Total_J):
        X_1[i][:] = np.array(X.iloc[i,]).astype(np.float32)
        X_1 = X_1.astype(np.float32)  

    ## x_2: Other products' features (permutation invariant) within the same market; 
    ## shape should be (number of products * number of markets, number of products-1, number of features)
    X_2 = np.ones((Total_J,J-1,K+1))
    for m in range(M):
        for j in range(J):
            i = m * J + j
            ### select other products in the same market 
            X_2[i][:] = np.array(X.loc[(X.index != i) & (X.index >= m * J) & (X.index < (m+1) * J),]).astype(np.float32)

    return X_1, X_2 



class SmallDeepSet(nn.Module):
    def __init__(self, x_d, pool="sum"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=x_d, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.share_enc = nn.Sequential(
            nn.Linear(in_features=x_d, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=64),                                                                                    
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            #nn.Sigmoid()
        )
        self.pool = pool

    def forward(self, shares, x):
        x = self.enc(x)
        shares = self.share_enc(shares)
        x = x.sum(dim=1) + shares.sum(dim=1)
        x = self.dec(x)
        return x.squeeze()    
    

# -------------------------------------------------------------

def train_deep(data):
    K = data['K']
    x_1, x_2 = x_transform_mm(data)
    # y = np.log(data['Y'])
    y = data['Y']
    model = SmallDeepSet(x_d = K+1)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #criterion = nn.BCELoss().cuda()
    criterion = nn.MSELoss().cuda()
    losses = []
    x_1, x_2, y = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda(), torch.from_numpy(y).float().cuda()
    iteration=0
    for _ in range(5000):
        loss = criterion(model(x_2, x_1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


def pred_deep(data, model):
    K = data['K']
    x_1, x_2 = x_transform_mm(data)
    x_1, x_2 = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda()
    y_pred = model(x_2, x_1)
    
    return y_pred.cpu().detach().numpy() #np.exp(y_pred.cpu().detach().numpy())



# =============================================================
# 3.  HELPERS 
# =============================================================

def split_train_test(data, p = 0.8):
    M = data['M']
    J = data['J']
    train_size = int(M * J * p)
    
    
    data_train = {
        'X': data['X'].iloc[0: train_size],
        'Y': data['Y'][0: train_size],
        'M': int(M * p),
        'J': J, 
        'K': data['K'], 
        'params': data['params'], 
        'generation_seed': data['generation_seed'], 
        'market_id' : data['market_id'][0: train_size]}
        
    data_test = {
        'X': data['X'].iloc[train_size: ,].reset_index(drop = True),
        'Y': data['Y'][train_size: ],
        'M': M - int(M*p),
        'J': J, 
        'K': data['K'], 
        'params': data['params'], 
        'generation_seed': data['generation_seed'], 
        'market_id' : data['market_id'][train_size:]}
    
    #print(data_test['M'])
    if 'b_random' in data.keys():
        data_train['b_random'] = [arr[0:train_size,:] for arr in data['b_random']]
        data_test['b_random'] = [arr[train_size:,:] for arr in data['b_random']]
    
    return data_train, data_test




def cal_true_share_change(data, from_x_to_y, prod_id, delta, seed):
    
    X = data['X'].copy()
    b = data['params']
    J = data['J']
    K = data['K']
    M = data['M']
    
    record = pd.DataFrame({'old_price' : X['price'].copy()})
    
    for m in range(M):
        new_id = prod_id + m * J
        X['price'].iloc[new_id] = (1 + delta) * X['price'].iloc[new_id]

    ## get the true elasticity 
    if from_x_to_y == mnl :
        y_true_new = from_x_to_y(X, b, J, K, M, seed)['Y']
    elif from_x_to_y == rcl:
        y_true_new = rcl_regenerate(X, data)
        
    record['new_true_share'] = y_true_new
    record['old_true_share'] = data['Y']
    
    record['true_change'] = (record['new_true_share'] - record['old_true_share'])    
    
    return record
