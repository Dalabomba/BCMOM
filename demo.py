# %%
from BCMOM import *

# %% [markdown]
#  # change the skewness

# %%
# 1 change the skewness
set_skewness = [20*i for i in range(1,5)]

# 2 lognormal distribution
sigma_lognorm = [find_inverse_binary(get_skewness_of_lognormal, skew, 0.1, 2, tolerance=1e-6, max_iterations=1000) for skew in set_skewness]
mu_lognorm = [(np.log(1/(np.exp(sigma**2)-1))-sigma**2)/2 for sigma in sigma_lognorm]


n_simulations = 500 # number of simulations
n_workers = 100 # m+1
n_samples_eachworker = 500 # sample size per worker
K = 5 # number of quantiles
dim = 1 # dimension of data
alpha = 0 # fraction of Byzantine machines
setting = [1,1]
iter = False
dist = 'lognormal'
# dist = 'gamma'
if dist == 'lognormal':
    param_1 = mu_lognorm
    param_2 = sigma_lognorm
if dist == 'gamma':
    param_1 = shape_gamma
    param_2 = scale_gamma

list_TRUE_mu = []
list_TRUE_sd = []
list_MOM = []
list_VRMOM = []
list_BCMOM = []

for i in tqdm(range(5)):
    i = i-1
    list_mu = []
    list_mu_mom = []
    list_mu_vrmom = []
    list_mu_bcmom = []
    for simulation in range(n_simulations): # 500 Monte Carlo simulations
        np.random.seed(10086+simulation)
        if dist == 'lognormal':
            data, mu, true_sd = generate_multi_lognormal_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker,
                                                          dim=dim, mu=param_1[i],sigma=param_2[i], alpha=alpha, bzmean=0, bzsd=np.sqrt(200))
        if dist == 'gamma':
            data, mu, true_sd = generate_multi_gamma_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker,
                                                          dim=dim, shape=param_1[i], scale=param_2[i], alpha=0, bzmean=0, bzsd=np.sqrt(200))
        if i == -1: # Lognormal degrades to normal when skew=0
            data, mu, true_sd = generate_multi_normal_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker,
                                                        dim=dim, mean=0, sd=1, alpha=0, bzmean=0, bzsd=np.sqrt(200))
        model = AllEstimators(data)
        list_mu.append(mu)
        list_mu_mom.append(model.initial_estimation)
        list_mu_vrmom.append(model.get_VRMOM_estimation(K=K, iter=iter))
        list_mu_bcmom.append(model.get_EVRMOM_estimation_new(K=K, iter=iter, setting=setting, th=5))
    list_TRUE_mu.append(list_mu)
    list_TRUE_sd.append(true_sd)
    list_MOM.append(list_mu_mom)
    list_VRMOM.append(list_mu_vrmom)
    list_BCMOM.append(list_mu_bcmom)

list_TRUE_mu = np.array(list_TRUE_mu).squeeze()
list_MOM = np.array(list_MOM).squeeze()
list_VRMOM = np.array(list_VRMOM).squeeze()
list_BCMOM = np.array(list_BCMOM).squeeze()


bias_mom = (list_MOM - list_TRUE_mu)
bias_vrmom = (list_VRMOM - list_TRUE_mu)
bias_bcmom = (list_BCMOM - list_TRUE_mu)

df_result = pd.DataFrame([np.mean(bias_mom, axis=1), np.mean(bias_vrmom, axis=1), np.mean(bias_bcmom, axis=1)], index=['MOM', 'VRMOM', 'TOEVRMOM'], columns=[0] + set_skewness)
print(df_result)



# %%
# Plotting
x1 = np.arange(1,20,4)
x2 = x1+1
x3 = x2+1
plt.style.use('default')
plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots(figsize=(5,4))
x = np.linspace(0,20,10)
y = np.repeat(0,10)
ax.plot(x,y,'--',color='black',linewidth=2)
box1 = ax.boxplot(bias_mom.T,positions=x1,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#8ECFC9",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box2 = ax.boxplot(bias_vrmom.T,positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#82B0D2",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box3 = ax.boxplot(bias_bcmom.T,positions=x3,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#FA7F6F",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                          )

plt.xticks(x2,[([0] + set_skewness)[i] for i in range(5)],
        fontsize=12)

ax.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3['boxes'][0]],labels=['MOM', 'VRMOM','BCMOM'],loc='upper right',fontsize=10)
plt.title('(a)', fontsize=16)
plt.xlabel('Skewness',fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.ylim(-0.03,0.03)
plt.tight_layout()
plt.savefig('SELECT YOUR OWN DIRECTORY/ChangeSkewness.png', dpi=300, bbox_inches='tight')


# %% [markdown]
#  # change the Byzanitne machines

# %%
# 2 change the Byzanitne machines
skewness = 80
sigma_lognorm = find_inverse_binary(get_skewness_of_lognormal, skewness, 0.1, 2, tolerance=1e-6, max_iterations=1000)
mu_lognorm = (np.log(1/(np.exp(sigma_lognorm**2)-1))-sigma_lognorm**2)/2

n_simulations = 500 # number of simulations
n_workers = 100 # m+1
n_samples_eachworker = 500 # sample size per worker
K = 5 # number of quantiles
dim = 1 # dimension of data
alpha = [0.02, 0.04, 0.06, 0.08, 0.1] # fraction of Byzantine machines

setting = [1,1]
iter = False
dist = 'lognormal'
# dist = 'gamma'
if dist == 'lognormal':
    param_1 = mu_lognorm
    param_2 = sigma_lognorm
if dist == 'gamma':
    param_1 = shape_gamma
    param_2 = scale_gamma

list_TRUE_mu = []
list_TRUE_sd = []
list_MOM = []
list_VRMOM = []
list_BCMOM = []

for i in tqdm(range(5)):
    list_mu = []
    list_mu_mom = []
    list_mu_vrmom = []
    list_mu_bcmom = []
    for simulation in range(n_simulations): # 500 Monte Carlo simulations
        np.random.seed(10086+simulation)
        if dist == 'lognormal':
            data, mu, true_sd = generate_multi_lognormal_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker,
                                                          dim=dim, mu=param_1,sigma=param_2, alpha=alpha[i], bzmean=0, bzsd=np.sqrt(200))
        if dist == 'gamma':
            data, mu, true_sd = generate_multi_gamma_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker,
                                                          dim=dim, shape=param_1[i], scale=param_2[i], alpha=0, bzmean=0, bzsd=np.sqrt(200))
        model = AllEstimators(data)
        list_mu.append(mu)
        list_mu_mom.append(model.initial_estimation)
        list_mu_vrmom.append(model.get_VRMOM_estimation(K=K, iter=iter))
        list_mu_bcmom.append(model.get_EVRMOM_estimation_new(K=K, iter=iter, setting=setting, th=5))
    list_TRUE_mu.append(list_mu)
    list_TRUE_sd.append(true_sd)
    list_MOM.append(list_mu_mom)
    list_VRMOM.append(list_mu_vrmom)
    list_BCMOM.append(list_mu_bcmom)

list_TRUE_mu = np.array(list_TRUE_mu).squeeze()
list_MOM = np.array(list_MOM).squeeze()
list_VRMOM = np.array(list_VRMOM).squeeze()
list_BCMOM = np.array(list_BCMOM).squeeze()


bias_mom = (list_MOM - list_TRUE_mu)
bias_vrmom = (list_VRMOM - list_TRUE_mu)
bias_bcmom = (list_BCMOM - list_TRUE_mu)


df_result_bias = pd.DataFrame([np.mean(bias_mom, axis=1), np.mean(bias_vrmom, axis=1), np.mean(bias_bcmom, axis=1)], index=['MOM', 'VRMOM', 'TOEVRMOM'], columns=alpha)
print(df_result_bias)
df_result_sd = pd.DataFrame([np.std(bias_mom, axis=1), np.std(bias_vrmom, axis=1), np.std(bias_bcmom, axis=1)], index=['MOM', 'VRMOM', 'TOEVRMOM'], columns=alpha)
print(df_result_sd)
df_result_rmse = pd.DataFrame([np.mean(bias_mom**2, axis=1), np.mean(bias_vrmom**2, axis=1), np.mean(bias_bcmom**2, axis=1)], index=['MOM', 'VRMOM', 'TOEVRMOM'], columns=alpha)
print(df_result_rmse)


# %%
# Plotting
x1 = np.arange(1,20,4)
x2 = x1+1
x3 = x2+1
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize=(10,10))
plt.style.use('default')
plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots(figsize=(5,4))
x = np.linspace(0,20,10)
y = np.repeat(0,10)
ax.plot(x,y,'--',color='black',linewidth=2)
box1 = ax.boxplot(bias_mom.T,positions=x1,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#8ECFC9",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box2 = ax.boxplot(bias_vrmom.T,positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#82B0D2",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box3 = ax.boxplot(bias_bcmom.T,positions=x3,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#FA7F6F",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                          )


plt.xticks(x2,[alpha[i] for i in range(5)],
        fontsize=12)


ax.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3['boxes'][0]],labels=['MOM', 'VRMOM','BCMOM'],loc='upper right',fontsize=10)
plt.xlabel(r'Byzantine Ratio $\alpha_m$',fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.ylim(-0.03,0.03)
plt.title('(b)', fontsize=16)
plt.tight_layout()
plt.savefig('SELECT YOUR OWN DIRECTORY/ChangeAlpha.png', dpi=300, bbox_inches='tight')


# %% [markdown]
#  # change n

# %%
# 3 change n
skewness = 80
sigma_lognorm = find_inverse_binary(get_skewness_of_lognormal, skewness, 0.1, 2, tolerance=1e-6, max_iterations=1000)
mu_lognorm = (np.log(1/(np.exp(sigma_lognorm**2)-1))-sigma_lognorm**2)/2

n_simulations = 500 # number of simulations
n_workers = 100 # m+1
n_samples_eachworker = [300, 600, 900, 1200, 1500] # sample size per worker
K = 5 # number of quantiles
dim = 1 # dimension of data
alpha = 0 # fraction of Byzantine machines
setting = [1,1]
iter = False
dist = 'lognormal'
# dist = 'gamma'
if dist == 'lognormal':
    param_1 = mu_lognorm
    param_2 = sigma_lognorm
if dist == 'gamma':
    param_1 = shape_gamma
    param_2 = scale_gamma

list_TRUE_mu = []
list_TRUE_sd = []
list_MOM = []
list_VRMOM = []
list_BCMOM = []

for i in tqdm(range(5)):
    list_mu = []
    list_mu_mom = []
    list_mu_vrmom = []
    list_mu_bcmom = []
    for simulation in range(n_simulations): # 500 Monte Carlo simulations
        np.random.seed(10086+simulation)
        if dist == 'lognormal':
            data, mu, true_sd = generate_multi_lognormal_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker[i],
                                                          dim=dim, mu=param_1,sigma=param_2, alpha=alpha, bzmean=0, bzsd=np.sqrt(200))
        if dist == 'gamma':
            data, mu, true_sd = generate_multi_gamma_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker,
                                                          dim=dim, shape=param_1, scale=param_2, alpha=0, bzmean=0, bzsd=np.sqrt(200))

        model = AllEstimators(data)
        list_mu.append(mu)
        list_mu_mom.append(model.initial_estimation)
        list_mu_vrmom.append(model.get_VRMOM_estimation(K=K, iter=iter))
        list_mu_bcmom.append(model.get_EVRMOM_estimation_new(K=K, iter=iter, setting=setting, th=5))
    list_TRUE_mu.append(list_mu)
    list_TRUE_sd.append(true_sd)
    list_MOM.append(list_mu_mom)
    list_VRMOM.append(list_mu_vrmom)
    list_BCMOM.append(list_mu_bcmom)

list_TRUE_mu = np.array(list_TRUE_mu).squeeze()
list_MOM = np.array(list_MOM).squeeze()
list_VRMOM = np.array(list_VRMOM).squeeze()
list_BCMOM = np.array(list_BCMOM).squeeze()


bias_mom = (list_MOM - list_TRUE_mu)
bias_vrmom = (list_VRMOM - list_TRUE_mu)
bias_bcmom = (list_BCMOM - list_TRUE_mu)


df_result = pd.DataFrame([np.mean(bias_mom, axis=1), np.mean(bias_vrmom, axis=1), np.mean(bias_bcmom, axis=1)], index=['MOM', 'VRMOM', 'TOEVRMOM'], columns=n_samples_eachworker)
print(df_result)



# %%
# Plotting
x1 = np.arange(1,20,4)
x2 = x1+1
x3 = x2+1
plt.style.use('default')
plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots(figsize=(5,4))
x = np.linspace(0,20,10)
y = np.repeat(0,10)
ax.plot(x,y,'--',color='black',linewidth=2)
box1 = ax.boxplot(bias_mom.T,positions=x1,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#8ECFC9",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box2 = ax.boxplot(bias_vrmom.T,positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#82B0D2",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box3 = ax.boxplot(bias_bcmom.T,positions=x3,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#FA7F6F",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                          )


plt.xticks(x2,[n_samples_eachworker[i] for i in range(5)],
        fontsize=12)


ax.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3['boxes'][0]],labels=['MOM', 'VRMOM','BCMOM'],loc='upper right',fontsize=10)
plt.xlabel(r'Sample Size on Each Machine $n$',fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.ylim(-0.03,0.03)
plt.title('(c)', fontsize=16)
plt.tight_layout()
plt.savefig('SELECT YOUR OWN DIRECTORY/ChangeN.png', dpi=300, bbox_inches='tight')


# %% [markdown]
#  # change m

# %%
# 4 change m
skewness = 80
sigma_lognorm = find_inverse_binary(get_skewness_of_lognormal, skewness, 0.1, 2, tolerance=1e-6, max_iterations=1000)
mu_lognorm = (np.log(1/(np.exp(sigma_lognorm**2)-1))-sigma_lognorm**2)/2

n_simulations = 500 # number of simulations
n_workers = [100, 200, 300, 400, 500] # m+1
n_samples_eachworker = 500 # sample size per worker
K = 5 # number of quantiles
dim = 1 # dimension of data
alpha = 0 # fraction of Byzantine machines
setting = [1,1]
iter = False
dist = 'lognormal'
# dist = 'gamma'
if dist == 'lognormal':
    param_1 = mu_lognorm
    param_2 = sigma_lognorm
if dist == 'gamma':
    param_1 = shape_gamma
    param_2 = scale_gamma

list_TRUE_mu = []
list_TRUE_sd = []
list_MOM = []
list_VRMOM = []
list_BCMOM = []

for i in tqdm(range(5)):
    list_mu = []
    list_mu_mom = []
    list_mu_vrmom = []
    list_mu_bcmom = []
    for simulation in range(n_simulations): # 500 Monte Carlo simulations
        np.random.seed(10086+simulation)
        if dist == 'lognormal':
            data, mu, true_sd = generate_multi_lognormal_data(n_workers=n_workers[i]+1, n_samples_eachworker=n_samples_eachworker,
                                                          dim=dim, mu=param_1,sigma=param_2, alpha=alpha, bzmean=0, bzsd=np.sqrt(200))
        if dist == 'gamma':
            data, mu, true_sd = generate_multi_gamma_data(n_workers=n_workers+1, n_samples_eachworker=n_samples_eachworker,
                                                          dim=dim, shape=param_1, scale=param_2, alpha=0, bzmean=0, bzsd=np.sqrt(200))

        model = AllEstimators(data)
        list_mu.append(mu)
        list_mu_mom.append(model.initial_estimation)
        list_mu_vrmom.append(model.get_VRMOM_estimation(K=K, iter=iter))
        list_mu_bcmom.append(model.get_EVRMOM_estimation_new(K=K, iter=iter, setting=setting, th=5))
    list_TRUE_mu.append(list_mu)
    list_TRUE_sd.append(true_sd)
    list_MOM.append(list_mu_mom)
    list_VRMOM.append(list_mu_vrmom)
    list_BCMOM.append(list_mu_bcmom)

list_TRUE_mu = np.array(list_TRUE_mu).squeeze()
list_MOM = np.array(list_MOM).squeeze()
list_VRMOM = np.array(list_VRMOM).squeeze()
list_BCMOM = np.array(list_BCMOM).squeeze()


bias_mom = (list_MOM - list_TRUE_mu)
bias_vrmom = (list_VRMOM - list_TRUE_mu)
bias_bcmom = (list_BCMOM - list_TRUE_mu)


df_result = pd.DataFrame([np.mean(bias_mom, axis=1), np.mean(bias_vrmom, axis=1), np.mean(bias_bcmom, axis=1)], index=['MOM', 'VRMOM', 'TOEVRMOM'], columns=n_workers)
print(df_result)



# %%
# Plotting
x1 = np.arange(1,20,4)
x2 = x1+1
x3 = x2+1
plt.style.use('default')
plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots(figsize=(5,4))
x = np.linspace(0,20,10)
y = np.repeat(0,10)
ax.plot(x,y,'--',color='black',linewidth=2)
box1 = ax.boxplot(bias_mom.T,positions=x1,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#8ECFC9",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box2 = ax.boxplot(bias_vrmom.T,positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#82B0D2",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                       )
box3 = ax.boxplot(bias_bcmom.T,positions=x3,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#FA7F6F",
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5},
            flierprops={'marker':'.',}
                          )


plt.xticks(x2,[n_workers[i] for i in range(5)],
        fontsize=12)


ax.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3['boxes'][0]],labels=['MOM', 'VRMOM','BCMOM'],loc='upper right',fontsize=10)
plt.xlabel(r'Number of Machines $m$',fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.ylim(-0.03,0.03)
plt.title('(d)', fontsize=16)
plt.tight_layout()
plt.savefig('SELECT YOUR OWN DIRECTORY/ChangeM.png', dpi=300, bbox_inches='tight')





