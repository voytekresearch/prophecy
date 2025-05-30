"""MAIN prophecy analysis simulating hypothesized LPFs and comparing to raw iEEG data"""

# Import necessary modules
import numpy as np


import pandas as pd
from scipy import signal, stats

import mne
from neurodsp import sim, plts, filt, spectral, timefrequency, utils
#import matplotlib.pyplot as plt


from prophecy_functions import *

save_path = '/labs/bvoyteklab/Smith/prophecy/02-results/main/'
#save_path = '/Users/sydneysmith/Projects/PrOPHEcy/PrOPHEcy/03-results/ieeg/cluster_test/TIMING/'

similarity_metrics = ['r', 'rho'] # pearson, spearman
compared_sigs = ['top', 'btm', 'comb', 'SAM_y', 'SAM_e', 'SAM_i'] # top-down, bottom-up, combined, SAM module
freq_ranges = ['raw']


subjs = ['1002', '1005', '1007', '1008', '1009', '1010', '1014']

for subj in subjs:
    print('running subject # ' + subj)


    epochs = load_fif_epo(subj, control=False) 
    epochs = epochs.resample(1000)
    epochs = epochs.pick_types(seeg=True)
    epoch_all_data = epochs.get_data(copy=False)

    n_trials = len(epochs)
    channels = epochs.info['ch_names']
    fs = 1000
    n_samples = 2500 # length of resampled segment
    similarity_df_list = []

    print('running all trials...')

    for trial in range(0,n_trials): #range(0,3): 
        
        if trial%5==0:
            print('running trial ' + str(trial))
        else:
            pass
        
        # get trial info
        trial_info = epochs.metadata.iloc[trial]
        epoch_trial_data = epoch_all_data[trial]
        trial_length = epoch_trial_data.shape[1] # total length of trial in ms

        # define start & end of segment
        tmin = 700 # WHAT IS THIS?
        tmax = int(700 + trial_info['SOA']*trial_info['Condition'] + trial_info['SOA Jitter'] + 400) # WHERE DO 700 AND 400 COME FROM???


        # simulate top-down, bottom-up, & combined LFP model
        tp_lfp, btm_lfp, combined_lfp = simulate_trial_lfps(trial_info, trial_length)
        
        # simulate sensory anticipation module LFP (adapted from Egger, Le, & Jazayeri 2020)
        sam_lfp_y, sam_lfp_e, sam_lfp_i = simulate_sam_lfp(trial_info, trial_length) 
       
        # raw simulated signals
        resamp_sim_sigs_raw = get_raw_sim_sigs([tp_lfp, btm_lfp, combined_lfp, sam_lfp_y, sam_lfp_e, sam_lfp_i], tmin, tmax, n_samples)

        # empty array for similarity metrics per channel
        similarity_vals_trial_array = np.zeros(shape=[len(channels), 12]) #12 = 2 sim_metrics x 6 signals

        # empty array for pac metrics per channel
        #pac_trial_array = np.zeros(shape=[len(channels)])

        for i, channel in enumerate(channels):

            similarity_vals_chan_list = [] 

            ieeg_dat = epoch_trial_data[i]
            norm_ieeg = normalize_waveform(ieeg_dat)[tmin:tmax]
            
            # compile iEEG signals for resampling
            eeg_sigs = [norm_ieeg] #delta_ieeg, beta_ieeg]

            resamp_eeg_sigs = []

            # resample iEEG signals
            for sig in eeg_sigs:
                resamp_eeg_sigs.append(signal.resample(sig, n_samples))

            # SANITY CHECK PLOT
            # if i==0 and trial==2:
            #     plt.figure(figsize=(15,5))
            #     plt.plot(resamp_eeg_sigs[0], label='ieeg')
            #     plt.plot(resamp_sim_sigs_raw[0], label='top-down', alpha=0.5)
            #     plt.plot(resamp_sim_sigs_raw[1], label='bottom-up', alpha=0.5)
            #     plt.plot(resamp_sim_sigs_raw[2], label='combined', alpha=0.5)
            #     plt.plot(resamp_sim_sigs_raw[3], label='SAM_y', alpha=0.5)
            #     plt.plot(resamp_sim_sigs_raw[4], label='SAM_e', alpha=0.5)
            #     plt.plot(resamp_sim_sigs_raw[5], label='SAM_i', alpha=0.5)
            #     plt.legend()
            #     plt.savefig('/Users/sydneysmith/Projects/PrOPHEcy/PrOPHEcy/03-results/ieeg/model_comparison/subj_'+subj+'sanity_check.pdf')


            # compute similarity metrics RAW
            for sim_sig in resamp_sim_sigs_raw: # raw simulated sigs (top_down, bottom_up, combined, SAM_y, SAM_e, SAM_i)
                eeg_sig = resamp_eeg_sigs[0]
                r_val = compute_r(sim_sig, eeg_sig)
                rho_val = compute_rho(sim_sig, eeg_sig)
                #euc_val = compute_euc(sim_sig, eeg_sig)

                # add to list of metrics for this channel
                similarity_vals_chan_list.extend([r_val, rho_val])#, euc_val])


            similarity_vals_trial_array[i,:] = np.asarray(similarity_vals_chan_list)

        col_names_trial = name_cols(freq_ranges, compared_sigs, similarity_metrics)    
        similarity_dict = dict(zip(col_names_trial, similarity_vals_trial_array.T))
        similarity_dict['trial'] = np.tile(trial,len(channels))
        similarity_dict['channel'] = channels

        similarity_df = pd.DataFrame(similarity_dict)
        similarity_df_list.append(similarity_df)


    similarity_df_all_trials = pd.concat(similarity_df_list)
    print('saving...')


    similarity_df_all_trials.to_csv(save_path+'subj_'+subj+'_model_comparisonTIMING.csv')
