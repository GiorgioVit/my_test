# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:18:55 2017

@author: Giorgio
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from envs.dvahedging import DVAHedging
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from modules.modules import plot_results
import time
from datetime import datetime
import os
import lasagne.nonlinearities as NL
from rllab.algos.trpo import TRPO
from rllab.algos.npo import NPO
from rllab.algos.vpg import VPG
from rllab.algos.nop import NOP
from rllab.algos.cem import CEM
#from rllab.algos.cma_es import CMAES
from rllab.algos.ddpg import DDPG

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from modules.collect_trajectories import plot_PL_RM, DVA_episodes, DVA_episode, DVA_trajectories
import shutil


# -----------------------------------------------------------------------------
# INPUT parameters ------------------------------------------------------------
input_parameters = {'n_itr': 2, 'N': 2, 'horizon': 7*96, 'n_neurons': 5, \
                    'empty_training': 0, 'empty_test': 1}
# CLASS parameters ------------------------------------------------------------
class_parameters = {'risk_factor': 1.00, 'dva_nominal': 400e6, 'window_offset': 1, \
                    'window_offset_day': 1, 'window_offset_week': 0,  'low_bound': 6}
# -----------------------------------------------------------------------------
step_size_list   = [0.00008]
risk_factor_list = [1]
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# start timer 
tic = time.time()

# set the parameters
n_itr       = input_parameters['n_itr']          # number of iterations
horizon     = input_parameters['horizon']        # each trajectory will have at most horizon steps
N           = input_parameters['N']              # we will collect N trajectories per iteration
n_neurons   = input_parameters['n_neurons']      # number neurons in one layer neural network
empty_train = input_parameters['empty_training'] # 1 se nella fase di train, il portafoglio iniziale sarà vuoto 
empty_test  = input_parameters['empty_test']     # 1 se nella fase di test, il portafoglio iniziale sarà vuoto 

dva_nominal        = class_parameters['dva_nominal']
window_offset      = class_parameters['window_offset']
window_offset_day  = class_parameters['window_offset_day']
window_offset_week = class_parameters['window_offset_week']
low_bound          = class_parameters['low_bound']  

# save the test description in a .txt -----------------------------------------
s = "Input Parameters  \n"
s = s + "step_size_list : " + str(step_size_list) 
s = s + "\nrisk_factor_list : " + str(risk_factor_list)
for p in input_parameters.keys():
    s = s + str(p) + ":  " + str(input_parameters[str(p)]) + "  \n"
s = s + "\nClass Parameters\n"
for p in class_parameters.keys():
    s = s + str(p) + ":  " + str(class_parameters[str(p)]) + " \n"
s = s + "\n"


directory = '../results/results_NPO_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'/'
if os.path.exists(directory):
    shutil.rmtree(directory)
    
if not os.path.exists(directory):
    os.makedirs(directory)
f = open(directory + 'test_description.txt','w')
f.write(s)
f.close()
# -----------------------------------------------------------------------------

#    
#    # load data -------------------------------------------------------------------
filename = '../dataset/DatasetDVA_2017-10-09_cleaned.csv'
data = pd.read_csv(filename, sep=';')

# ----------------------------------------------------------------------------
bsl = ['btp', 'iTraxx', 'btp-iTraxx']

DVAepisodes_train  = DVA_episodes()  # save in here all the trajectories
DVAepisodes_bsl_train = []
for b in range(len(bsl)):
    DVAepisodes_bsl_train.append(DVA_episodes())

DVAepisodes  = DVA_episodes()  # save in here all the trajectories
DVAepisodes_bsl = []
for b in range(len(bsl)):
    DVAepisodes_bsl.append(DVA_episodes())

selu = NL.SELU(scale=1.0507009873554804934193349852946, scale_neg=1.6732632423543772848170429916717)
for idx in range(len(risk_factor_list)):
    risk_factor = risk_factor_list[idx]
    step_size   = step_size_list[idx]

    env = normalize(DVAHedging(market_datafile=filename, horizon=horizon, dva_nominal=dva_nominal,training=True,risk_factor=risk_factor, \
                     window_offset=window_offset, window_offset_day=window_offset_day,window_offset_week=window_offset_week, \
                     empty_allocation=empty_train, verbose=0,low_bound=low_bound, sep=';'))
    
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(n_neurons,n_neurons ),
        std_hidden_nonlinearity= NL.selu,
        hidden_nonlinearity= NL.selu,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    algo = NPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size= N*horizon,
        max_path_length=horizon,
        n_itr=n_itr,
        discount=1,
    )

    returns = algo.train()

    path_training = plot_results(directory+'train/',[returns], xlabel = 'Number Iteration', ylabel = 'Average Return ', title = 'Training Phase', name = 'training_return', risk_factor=risk_factor)
    
    # --------------- test ----------------------------------------------------------
    print("Starting test on train data....")

    env.wrapped_env.empty_allocation = empty_test
    env.wrapped_env.training = True

    DVAepisode = DVA_episode(risk_factor)

    DVAepisode_bsl = []
    for b in range(len(bsl)):
        DVAepisode_bsl.append(DVA_episode(risk_factor))

    env_bsl = []
    for b in range(len(bsl)):
        env_bsl.append(DVAHedging(market_datafile=filename, horizon=horizon,training=True,dva_nominal=dva_nominal, window_offset=window_offset,risk_factor=risk_factor, \
                      window_offset_day=window_offset_day, window_offset_week=window_offset_week, empty_allocation=1,verbose=0, low_bound=low_bound,sep=';'))

    rows = data.shape[0]-low_bound-1
    l_bound = 96*7
    u_bound  = round(3/4*rows)

    n_test_train = 1 # keep track of the number of trajectories per episode

    while (l_bound + horizon*n_test_train)< u_bound:
        # select start point in test set
        start_point = l_bound + horizon*(n_test_train-1)

        for b in range(len(bsl)):
            env_bsl[b].start_point = start_point

        # select the state, and the first action to hedge the portfolio, use the baseline
        for b in range(len(bsl)):
            _ = env_bsl[b].reset()
            action_bsl      = env_bsl[b].baseline_action(flag=bsl[b])
            _, _, _, _ = env_bsl[b].step(action_bsl)
            # if bsl[b]=='btp':
            #     env_btp.wrapped_env.start_point = start_point
            #     observation = env_btp.wrapped_env.reset()
            #     observation, reward, terminal, _ = env_btp.wrapped_env.step(action_bsl)

        # reset anche l'ambiente test
        env.wrapped_env.start_point = start_point + 1
        observation = env.wrapped_env.reset()


        # salvare le traiettorie
        DVAtrajectories = DVA_trajectories()
        DVAtrajectories.append(env.wrapped_env.allocation_tot, env.wrapped_env.total_PL)
        # baseline
        DVAtrajectories_bsl = []
        for b in range(len(bsl)):
            DVAtrajectories_bsl.append(DVA_trajectories())
            DVAtrajectories_bsl[b].append(env_bsl[b].allocation_tot, env_bsl[b].total_PL)

        # a inizio traittoria, inizializzare il P&L giornaliero a 0 -----------
        daily_PL     = 0
        daily_PL_bsl = np.zeros(len(bsl))

        for i in range(horizon):
            # step polivy -----------------------------------------------------
            _, policy_param = algo.policy.get_action(observation)
            action = policy_param['mean']
            next_observation, reward, terminal, _ = env.wrapped_env.step(action)
            observation = next_observation

            # step baseline ---------------------------------------------------
            for b in range(len(bsl)):
                action_bsl = env_bsl[b].baseline_action(flag=bsl[b])
                _, _, _, _ = env_bsl[b].step(action_bsl)

            # update del P&L giornaliero --------------------------------------
            if env.wrapped_env.chiusura == 0:
                daily_PL += env.wrapped_env.total_PL
                for b in range(len(bsl)):
                    daily_PL_bsl[b] += env_bsl[b].total_PL
            else:
                DVAtrajectories.add_daily_PL(daily_PL)
                daily_PL = 0

                for b in range(len(bsl)):
                    DVAtrajectories_bsl[b].add_daily_PL(daily_PL_bsl[b])
                    daily_PL_bsl[b] = 0

            # aggiungere lo step alle traiettorie -----------------------------
            DVAtrajectories.append(env.wrapped_env.allocation_tot, env.wrapped_env.total_PL)

            for b in range(len(bsl)):
                DVAtrajectories_bsl[b].append(env_bsl[b].allocation_tot, env_bsl[b].total_PL)

        # salvavare le traiettorie nell' oggetto episodio ---------------------
        DVAepisode.append(DVAtrajectories)

        for b in range(len(bsl)):
            DVAepisode_bsl[b].append(DVAtrajectories_bsl[b])

        # mandare avanti di 1 il contatore del numero di triettorie di test ---
        n_test_train += 1

    # salvare gli episodi -----------------------------------------------------
    DVAepisodes_train.append(DVAepisode)
    for b in range(len(bsl)):
        DVAepisodes_bsl_train[b].append(DVAepisode_bsl[b])

    # -------------------------------------------------------------------------
    print("Real test....")
    # -------------------------------------------------------------------------
    
    env.wrapped_env.empty_allocation = empty_test
    env.wrapped_env.training = False

    DVAepisode = DVA_episode(risk_factor)
    
    DVAepisode_bsl = []
    for b in range(len(bsl)):
        DVAepisode_bsl.append(DVA_episode(risk_factor))
    
    #   creare l' ambiente per le baseline
    env_bsl = []
    for b in range(len(bsl)):
        env_bsl.append(DVAHedging(market_datafile=filename, horizon=horizon,training=True,dva_nominal=dva_nominal, window_offset=window_offset,risk_factor=risk_factor, \
                      window_offset_day=window_offset_day, window_offset_week=window_offset_week, empty_allocation=1,verbose=0, low_bound=low_bound,sep=';'))

    l_bound = round(3/4*rows)+1
    u_bound  = data.shape[0] 
    
    n_test = 1 # keep track of the number of trajectories per episode
       
    while (l_bound + horizon*n_test)< u_bound:
        # select start point in test set
        start_point = l_bound + horizon*(n_test-1)
    
        for b in range(len(bsl)):
            env_bsl[b].start_point = start_point
        
        # select the state, and the first action to hedge the portfolio, use the baseline
        for b in range(len(bsl)):
            _ = env_bsl[b].reset()   
            action_bsl      = env_bsl[b].baseline_action(flag=bsl[b])
            _, _, _, _ = env_bsl[b].step(action_bsl)
#            if bls[b]=='btp':
                # env.wrapped_env.start_point = start_point
                # observation = env.wrapped_env.reset()
                # observation, reward, terminal, _ = env.wrapped_env.step(action_bsl)

        # reset anche l'ambiente test
        env.wrapped_env.start_point = start_point + 1
        observation = env.wrapped_env.reset()

        # salvare le traiettorie
        DVAtrajectories = DVA_trajectories()
        DVAtrajectories.append(env.wrapped_env.allocation_tot, env.wrapped_env.total_PL)
        # baseline
    
        DVAtrajectories_bsl = []
        for b in range(len(bsl)):
            DVAtrajectories_bsl.append(DVA_trajectories())
            DVAtrajectories_bsl[b].append(env_bsl[b].allocation_tot, env_bsl[b].total_PL)
        
        # a inizio traittoria, inizializzare il P&L giornaliero a 0 -----------
        daily_PL     = 0
        daily_PL_bsl = np.zeros(len(bsl))

        for i in range(horizon):
            # step polivy -----------------------------------------------------
            _, policy_param = algo.policy.get_action(observation)
            action = policy_param['mean']
            next_observation, reward, terminal, _ = env.wrapped_env.step(action)
            observation = next_observation
    
            # step baseline ---------------------------------------------------
            for b in range(len(bsl)):
                action_bsl = env_bsl[b].baseline_action(flag=bsl[b])
                _, _, _, _ = env_bsl[b].step(action_bsl)
    
            # update del P&L giornaliero --------------------------------------
            if env.wrapped_env.chiusura == 0:
                daily_PL += env.wrapped_env.total_PL
                for b in range(len(bsl)):
                    daily_PL_bsl[b] += env_bsl[b].total_PL
            else:
                DVAtrajectories.add_daily_PL(daily_PL)
                daily_PL = 0
    
                for b in range(len(bsl)):
                    DVAtrajectories_bsl[b].add_daily_PL(daily_PL_bsl[b])
                    daily_PL_bsl[b] = 0
                              
            # aggiungere lo step alle traiettorie -----------------------------
            DVAtrajectories.append(env.wrapped_env.allocation_tot, env.wrapped_env.total_PL)
    
            for b in range(len(bsl)):
                DVAtrajectories_bsl[b].append(env_bsl[b].allocation_tot, env_bsl[b].total_PL)
        
        # salvavare le traiettorie nell' oggetto episodio ---------------------
        DVAepisode.append(DVAtrajectories)
    
        for b in range(len(bsl)):
            DVAepisode_bsl[b].append(DVAtrajectories_bsl[b])
        
        # mandare avanti di 1 il contatore del numero di triettorie di test ---
        n_test += 1
    
    # salvare gli episodi -----------------------------------------------------    
    DVAepisodes.append(DVAepisode)
    for b in range(len(bsl)):
        DVAepisodes_bsl[b].append(DVAepisode_bsl[b])
    
    ## -------------------- plot trajectories --------------------------------- 
    path_sx7e = plot_results(directory+'test/',DVAepisode.sx7e, xlabel = 'Step', ylabel = 'lotti SX7E',  name = 'SX7E_Policy', risk_factor=risk_factor)
    path_btp = plot_results(directory+'test/',DVAepisode.btp, xlabel = 'Step', ylabel = 'lotti BTP',  name = 'BTP_Policy', risk_factor =risk_factor)
    path_iTraxx = plot_results(directory+'test/',DVAepisode.itraxx, xlabel = 'Step', ylabel = 'lotti ITRAXX',  name = 'ITRAXX_Policy', risk_factor=risk_factor)
    path_bank_account = plot_results(directory+'test/',DVAepisode.bank_account, xlabel = 'Step', ylabel = 'bank_account',  name = 'bank_account', risk_factor=risk_factor)
       


# plottare la frontiera P&L - Misura del Rischio ------------------------------    
plot_PL_RM(directory+'train/', DVAepisodes_train, DVAepisodes_bsl_train, bsl, n_test_train, flag = 'VaR')
plot_PL_RM(directory+'train/', DVAepisodes_train, DVAepisodes_bsl_train, bsl, n_test_train, flag = 'ES')
plot_PL_RM(directory+'train/', DVAepisodes_train, DVAepisodes_bsl_train, bsl, n_test_train, flag = 'L2')

plot_PL_RM(directory+'test/', DVAepisodes, DVAepisodes_bsl, bsl, n_test, flag = 'VaR')
plot_PL_RM(directory+'test/', DVAepisodes, DVAepisodes_bsl, bsl, n_test, flag = 'ES')
plot_PL_RM(directory+'test/', DVAepisodes, DVAepisodes_bsl, bsl, n_test, flag = 'L2')

# running time ----------------------------------------------------------------
toc = time.time() 
elapsed = (toc - tic)/60
print("computation time (min):  %3.2f" %elapsed)

s = "\n# hidden layers: " + str(algo.policy._cached_param_shapes[()][1])
s = s + "\n\ncomputation time : "+str(elapsed) + " min \n"
# s = s + "P&L :  " + str(DVAepisodes.PL_m) + str(DVAepisodes.PL_sigma) + \
#         "\nVaR :  " + str(DVAepisodes.VaR_m) + str(DVAepisodes.VaR_sigma) + \
#         "\nES :  " +   str(DVAepisodes.ES_m) + str(DVAepisodes.ES_sigma) + \
#          "\nL2 :" + str(DVAepisodes.l2_daily_PL_m) + str(DVAepisodes.l2_daily_PL_sigma)
f = open(directory + 'test_description.txt', 'a')
f.write(s)
f.close()

print(" ... End!")    


