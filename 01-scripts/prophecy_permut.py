"""PERMUTATION prophecy analysis simulating hypothesized LPFs and comparing to raw iEEG data"""

# Import necessary modules
import numpy as np
import random
import pandas as pd
from scipy import signal, stats

import mne
from neurodsp import sim, plts, filt, spectral, timefrequency, utils
#import matplotlib.pyplot as plt


from prophecy_functions import *

save_path = '/labs/bvoyteklab/Smith/prophecy/02-results/permut/n_1000_pvals/'

# PERMUTATION SETTINGS
N_RUNS = 1000
SEED = 0

similarity_metrics = ['r', 'rho'] # pearson, spearman
compared_sigs = ['top', 'btm', 'comb', 'SAM_y', 'SAM_e', 'SAM_i'] # top-down, bottom-up, combined, SAM module
freq_ranges = ['raw']

subjs = ['1008'] #['1002', '1005', '1007', '1008', '1009', '1010', '1014']

for subj in subjs:
    print('running subject # ' + subj)

    epochs = load_fif_epo(subj, control=False) 
    epochs = epochs.resample(1000)
    epochs = epochs.pick_types(seeg=True)
    epoch_all_data = epochs.get_data(copy=False)

    n_trials = len(epochs)
    trial_list = list(range(0, n_trials))
    random.seed(SEED)
    shuffled_trials = random.sample(trial_list, len(trial_list))

    channels = epochs.info['ch_names']
    n_channels = len(channels)
    fs = 1000
    n_samples = 2500 # length of resampled segment

    # Make empty arrays for df values
    similarity_vals_subject_array = np.zeros(shape=[N_RUNS, n_trials, n_channels, 12]) # array for single subject (1000 runs)
    perm_n_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    shuff_ISI_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    shuff_SOA_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    shuff_n_tones_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    trial_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    real_ISI_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    real_SOA_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    real_n_tones_array = np.zeros(shape=[N_RUNS, n_trials, n_channels])
    channel_array = np.zeros(shape=[N_RUNS, n_trials, n_channels], dtype="<U20")

    info_col_names = ['perm_n', 'shuff_ISI', 'shuff_SOA', 'shuff_n_tones', 'trial', 'real_ISI', 'real_SOA', 'real_n_tones', 'channel']

    print('running permutations')

    for run in range(0, N_RUNS):

        if run%50==0:
            print('running permutation '+ str(run)+' / 1000')

        for trial, rand_trial in zip(range(0,n_trials), shuffled_trials): #trial_list
            
            # if trial%50==0:
            #     print('running trial ' + str(trial))
            # else:
            #     pass
            
            # get trial info
            trial_info = epochs.metadata.iloc[rand_trial]
            epoch_trial_data = epoch_all_data[trial]
            trial_length = epoch_trial_data.shape[1] # total length of trial in ms

            # get trial info
            trial_info = epochs.metadata.iloc[rand_trial] #randomized for permutation test
            real_trial_info = epochs.metadata.iloc[trial]

            # define start & end of segment
            tmin = 700 
            tmax = int(700 + real_trial_info['SOA']*real_trial_info['Condition'] + real_trial_info['SOA Jitter'] + 400) 


            # simulate top-down, bottom-up, & combined LFP model
            tp_lfp, btm_lfp, combined_lfp = simulate_trial_lfps(trial_info, trial_length)
            
            # simulate sensory anticipation module LFP (adapted from Egger, Le, & Jazayeri 2020)
            sam_lfp_y, sam_lfp_e, sam_lfp_i = simulate_sam_lfp(trial_info, trial_length) 
           
            # raw simulated signals
            resamp_sim_sigs_raw = get_raw_sim_sigs([tp_lfp, btm_lfp, combined_lfp, sam_lfp_y, sam_lfp_e, sam_lfp_i], tmin, tmax, n_samples)


            for i, channel in enumerate(channels):

                similarity_vals_chan_list = [] 

                ieeg_dat = epoch_trial_data[i] 
                norm_ieeg = normalize_waveform(ieeg_dat)[tmin:tmax]
                
                # compile iEEG signals for resampling
                eeg_sigs = [norm_ieeg]

                resamp_eeg_sigs = []

                # resample iEEG signals
                for sig in eeg_sigs:
                    resamp_eeg_sigs.append(signal.resample(sig, n_samples))


                # compute similarity metrics RAW
                for sim_sig in resamp_sim_sigs_raw: # raw simulated sigs (top_down, bottom_up, combined, SAM_y, SAM_e, SAM_i)
                    eeg_sig = resamp_eeg_sigs[0]
                    r_val = compute_r(sim_sig, eeg_sig)
                    rho_val = compute_rho(sim_sig, eeg_sig)

                    # add to list of metrics for this channel
                    similarity_vals_chan_list.extend([r_val, rho_val])#, euc_val])

                similarity_vals_subject_array[run,trial,i,:] = np.asarray(similarity_vals_chan_list)
                perm_n_array[run,trial,i] = run
                shuff_ISI_array[run,trial,i] = trial_info['SOA']
                shuff_SOA_array[run,trial,i] = trial_info['SOA Jitter']
                shuff_n_tones_array[run,trial,i] = trial_info['Standard #'] + 1
                trial_array[run,trial,i] = trial
                real_ISI_array[run,trial,i] = real_trial_info['SOA']
                real_SOA_array[run,trial,i] = real_trial_info['SOA Jitter']
                real_n_tones_array[run,trial,i] = real_trial_info['Standard #'] + 1
                channel_array[run,trial,i] = channel


    similarity_col_names = name_cols(freq_ranges, compared_sigs, similarity_metrics)
    reshaped_similarity_vals_array = np.reshape(similarity_vals_subject_array, newshape=[N_RUNS*n_trials*n_channels, 12])
    similarity_dict = dict(zip(similarity_col_names, reshaped_similarity_vals_array.T))

    info_dat = [perm_n_array, 
                shuff_ISI_array,
                shuff_SOA_array,
                shuff_n_tones_array,
                trial_array,
                real_ISI_array,
                real_SOA_array,
                real_n_tones_array,
                channel_array]

    for column, dat in zip(info_col_names, info_dat):
        reshaped_dat = dat.flatten()
        similarity_dict[column] = reshaped_dat



    similarity_df_subj = pd.DataFrame(similarity_dict)

    # compute median r and rho values

    features = ['raw_top_r', 'raw_top_rho',
                'raw_btm_r', 'raw_btm_rho', 
                'raw_comb_r', 'raw_comb_rho',
                'raw_SAM_y_r', 'raw_SAM_y_rho',
                'raw_SAM_e_r', 'raw_SAM_e_rho', 
                'raw_SAM_i_r', 'raw_SAM_i_rho']

    median_perm_df = similarity_df_subj.groupby(['perm_n', 'channel'], as_index=False)[features].median()


    print('saving...')


    median_perm_df.to_csv(save_path+'subj_'+subj+'_permut_median.csv')




