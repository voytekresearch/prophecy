"""runs a parameter sweep of LFP simulations in prophecy analysis"""
import numpy as np
from scipy import signal, stats
import pandas as pd
from neurodsp import sim

import mne

import sys 
sys.path.insert(1, '/Users/sydneysmith/Projects/PrOPHEcy/PrOPHEcy/01-scripts')
from param_sweep_functions import *

subjs = ['1002', '1005', '1007', '1008', '1009', '1010', '1014']
similarity_metrics = ['rho'] # pearson, spearman
compared_sigs = ['top', 'btm', 'SAM_y'] # top-down, bottom-up, combined, SAM module
freq_ranges = ['raw']

# param conditions
epsp_params = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.002, 0.002] # kernel decay timescale in seconds (last two are standard because synaptic kernel unused)
conditions = ['ts_1', 'ts_2', 'ts_5', 'ts_10', 'ts_50', 'ts_100', 'ts_200', 'no_kernel', 'gauss']

#######################################################################################################################################
############################################################## FUNCTIONS ##############################################################
#######################################################################################################################################

def simulate_trial_lfps(trial_info, trial_length, tau_d=float):
    """simulate lfps from given iEEG trial
    Parameters
    ----------
    trial_info : dict
        dictionary of trial info from iEEG metadata
    trial_length : int
        length of trial, in ms
    tau_d : float
        timescale of kernel decay (in seconds)
        
    Returns
    -------
    tp_lfp : 1d array
        spike probability signal convlved with spiking kernel
    tp_lfp : 1d array
        spike probability signal convlved with spiking kernel
    tp_lfp : 1d array
        spike probability signal convlved with spiking kernel
    """
    # simulate LFPs
    n_tones = trial_info['Standard #'] + 1 #number of tones in the trial
    stim_length = 200 #length of auditory stimulus
    isi_length = trial_info['SOA'] #interstimulus interval of tone sequence in ms
    soa_length = int(round(trial_info['SOA Jitter'] - isi_length)) #soa
    fs = 1000 # sampling frequency
    edge_length = 800 #time before trial start and end of tones
    #n_tones = 5 #number of tones in the trial

    n_neurons = 100 #neurons in each population converging
    #n_trials = 60 #number of identical trials to simulate

    fr_rest = 0.05 #minimum probability of neuron spiking at any point
    fr_stim = 0.5  #maximum probability of neuron spiking at any point

    tau_r = 0.0001 #rise timescale of synaptic kernel (s)
    #tau_d = 0.002 #decay timescale of synaptic kernel (s)
    t_ker = 0.5 #total duration of synaptic kernel (s)

    adapt_rate = 0.2 #adaptation rate (bottom-up) <-- removed from sim as of Aug 2024
    sens_rate = 0.2 #sensitization rate (top-down, height of gauss)
    learning_rate = 0.15 #learning rate (top-down, width of gauss)
    
    # generate tone onsets and kernel
    tone_onsets_tp = get_tone_onsets_tp(n_tones, stim_length, isi_length, soa_length, edge_length)
    tone_onsets_btm = get_tone_onsets_btm(n_tones, stim_length, isi_length, soa_length, edge_length)
    ker = sim.sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    
    # simualte signals
    combined_lfps = np.zeros(trial_length)
    tp_lfps = np.zeros(trial_length)
    btm_lfps = np.zeros(trial_length)

    tp_neurons = np.zeros(shape=[n_neurons,trial_length])
    btm_neurons = np.zeros(shape=[n_neurons,trial_length])

    for neuron in range(0,n_neurons):
        tp_prob, tp_signal = sim_top_down(tone_onsets_tp, fs, stim_length, learning_rate, trial_length,
                                          fr_rest, fr_stim, sens_rate, ker)
        tp_neurons[neuron,:] = tp_signal

        btm_prob, btm_signal = sim_bottom_up(tone_onsets_btm, fs, stim_length, trial_length,
                                             fr_rest, fr_stim, adapt_rate, ker)
        btm_neurons[neuron,:] = btm_signal

    tp_lfp = tp_neurons.sum(0)
    btm_lfp = btm_neurons.sum(0)
    #combined_lfp = np.sum([btm_neurons.sum(0), tp_neurons.sum(0)], axis=0)
    
    return tp_lfp, tp_prob, btm_lfp, btm_prob


def simulate_sam_lfp(trial_info, trial_length, tau_d=float):
    """simulates a LFP by interpolating and convilving EPSC kernels with a SAM spiking probability signal

    Parameters
    ----------

    Returns
    -------
    sam_lfp : 1d array
        local field potetial from 100 neurons spiking according to the SAM module on one trial

    """
    n_tones = trial_info['Standard #'] + 1 #number of tones in the trial
    stim_length = 200 #length of auditory stimulus
    isi_length = trial_info['SOA'] #interstimulus interval of tone sequence in ms
    soa_length = int(round(trial_info['SOA Jitter'] - isi_length)) #soa
    fs = 1000 # sampling frequency
    edge_length = 800 #time before trial start and end of tones
    #n_tones = 5 #number of tones in the trial

    n_neurons = 100 #neurons in each population converging
    n_trials = 60 #number of identical trials to simulate

    fr_rest = 0.05 #minimum probability of neuron spiking at any point
    fr_stim = 0.5 #maximum probability of neuron spiking at any point

    tau_r = 0.0001 #rise timescale of synaptic kernel (s)
    #tau_d = 0.002 #decay timescale of synaptic kernel (s)
    t_ker = 0.5 #total duration of synaptic kernel (s)

    i_tau_r = 0.0005 #rise of inhibitory synaptic kernel (s)
    i_tau_d = 0.01 #decay of inhibitory synaptic kernel (s)
    i_t_ker = 0.05 #total duration of synaptic kernel (s)

    ker = sim.sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    i_ker = sim.sim_synaptic_kernel(i_t_ker, fs, i_tau_r, i_tau_d)*-1 # inverted polarity

    e_sig, i_sig, y_sig = sim_sam_trial(trial_info, trial_length)
    sam_lfp_y = sim_lfp(y_sig, ker, n_neurons, trial_length, fr_rest, fr_stim)
    #sam_lfp_e = sim_lfp(e_sig, ker, n_neurons, trial_length, fr_rest, fr_stim)
    #sam_lfp_i = sim_lfp(i_sig, i_ker, n_neurons, trial_length, fr_rest, fr_stim)

    return sam_lfp_y, y_sig #, sam_lfp_e, sam_lfp_i

def sim_lfp(p_spike, ker, n_neurons, trial_length, fr_rest, fr_stim):
    """simulates a LFP by interpolating and convilving EPSC kernels with a SAM spiking probability signal

    Parameters
    ----------
    y_sig : 1d array
        array of excitatory (comparator) population activity

    Returns
    -------
    sam_lfp : 1d array
        local field potetial from 100 neurons spiking according to the SAM module on one trial

    """
    #ker = sim.sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)

    sam_sigs_all = np.zeros([n_neurons, trial_length])

    # fix convolution delay
    conv_delay = 250
    p_spike = np.pad(p_spike, (conv_delay,0), 'edge')
    p_spike = p_spike[:-250]


    for n in range(0, n_neurons):
        norm_spikes = normalize_waveform(p_spike)
        spike_prob = get_spike_prob(fr_rest, fr_stim, norm_spikes)
        sam_sig = sim_spikes(spike_prob, ker, trial_length)
        sam_sigs_all[n,:] = sam_sig

    lfp = normalize_waveform(sam_sigs_all.sum(0))

    return lfp


##################################################################################################################################
################################################## MASTER ANALYSIS SCRIPT ########################################################
##################################################################################################################################



df_list = []

for subj in subjs: 

    epochs = load_fif_epo(subj, control=False)
    epochs = epochs.resample(1000)
    epochs_seeg = epochs.copy().pick_types(seeg=True)
    epochs_stim = epochs.copy().pick_types(stim=True)

    #dc1 = epochs_stim.get_data(picks='DC1')
    epoch_all_data = epochs_seeg.get_data(picks='seeg')
    
    n_trials = len(epochs)
    trial_list = list(range(0,n_trials)) 
    
    channels = epochs_seeg.info['ch_names']
    n_channels = len(channels)
    fs = 1000
    n_samples = 2500
    
    # make empty arrays for df values
    subjs_array = np.zeros(shape=[n_trials, n_channels], dtype='<U5')
    trial_array = np.zeros(shape=[n_trials, n_channels])
    ISI_array = np.zeros(shape=[n_trials, n_channels])
    SOA_array = np.zeros(shape=[n_trials, n_channels])
    n_tones_array = np.zeros(shape=[n_trials, n_channels])
    channel_array = np.zeros(shape=[n_trials, n_channels], dtype='<U20')
    similarity_vals_subj_array = np.zeros(shape=[n_trials, n_channels, 9, 3]) # 9 is number of conditions in parameter sweep, 3 is number of models
    
    col_names = ['subj', 'trial', 'ISI', 'SOA', 'n_tones', 'channel',
                 'top_ts_1_rho',      'btm_ts_1_rho',      'sam_ts_1_rho', 
                 'top_ts_2_rho',      'btm_ts_2_rho',      'sam_ts_2_rho',
                 'top_ts_5_rho',      'btm_ts_5_rho',      'sam_ts_5_rho',
                 'top_ts_10_rho',     'btm_ts_10_rho',     'sam_ts_10_rho', 
                 'top_ts_50_rho',     'btm_ts_50_rho',     'sam_ts_50_rho', 
                 'top_ts_100_rho',    'btm_ts_100_rho',    'sam_ts_100_rho', 
                 'top_ts_200_rho',    'btm_ts_200_rho',    'sam_ts_200_rho', 
                 'top_no_kernel_rho', 'btm_no_kernel_rho', 'sam_no_kernel_rho',
                 'top_gauss_rho',     'btm_gauss_rho',     'sam_gauss_rho']
    
    for trial in trial_list:
    
        # get trial info
        trial_info = epochs.metadata.iloc[trial]
        epoch_trial_data = epoch_all_data[trial]
        trial_length = epoch_trial_data.shape[1] # total length of trial in ms
    
        # define start & end of segment
        tmin = 700 
        tmax = int(700 + trial_info['SOA']*trial_info['Condition'] + trial_info['SOA Jitter'] + 400)

        # loop through kernel decays
        for j, (condition, decay) in enumerate(zip(conditions, epsp_params)):

            if condition[0]=='t': #timescale decay conditions

                # simulate top-down, bottom-up, & combined LFP model
                tp_lfp, tp_prob, btm_lfp, btm_prob = simulate_trial_lfps(trial_info, trial_length, tau_d=decay)
                
                # simulate sensory anticipation module LFP (adapted from Egger, Le, & Jazayeri 2020)
                sam_lfp_y, sam_y_prob = simulate_sam_lfp(trial_info, trial_length, tau_d=decay) 

                top_sim = tp_lfp
                btm_sim = btm_lfp
                sam_sim = sam_lfp_y

            if condition[0]=='n': #no kernel condition

                # simulate top-down, bottom-up, & combined LFP model
                tp_lfp, tp_prob, btm_lfp, btm_prob = simulate_trial_lfps(trial_info, trial_length, tau_d=decay)
                
                # simulate sensory anticipation module LFP (adapted from Egger, Le, & Jazayeri 2020)
                sam_lfp_y, sam_y_prob = simulate_sam_lfp(trial_info, trial_length, tau_d=decay)

                top_sim = tp_prob
                btm_sim = btm_prob
                sam_sim = sam_y_prob


            if condition[0]=='g': #gaussian kernel condition

                # simulate top-down, bottom-up, & combined LFP model
                _, tp_prob, _, btm_prob = simulate_trial_lfps(trial_info, trial_length, tau_d=decay)
                
                # simulate sensory anticipation module LFP (adapted from Egger, Le, & Jazayeri 2020)
                _, sam_y_prob = simulate_sam_lfp(trial_info, trial_length, tau_d=decay) 

                # simulate lfp with gaussian convolution kernel
                n_samples = 50
                x = np.linspace(-3, 3, n_samples)
                mean = 0
                std_dev = 1
                ker = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

                tp_lfp = sim_lfp(tp_prob, ker, n_neurons=100, trial_length=trial_length, fr_rest=0.05, fr_stim=0.5)
                btm_lfp = sim_lfp(btm_prob, ker, n_neurons=100, trial_length=trial_length, fr_rest=0.05, fr_stim=0.5)
                sam_lfp = sim_lfp(sam_y_prob, ker, n_neurons=100, trial_length=trial_length, fr_rest=0.05, fr_stim=0.5)

                top_sim = tp_lfp
                btm_sim = btm_lfp
                sam_sim = sam_lfp

        
            for i, channel in enumerate(channels):
                
                # get ieeg data from channel
                ieeg_dat = epoch_trial_data[i]
        
                # normalize ieeg and dc1 data
                norm_ieeg = normalize_waveform(ieeg_dat)[tmin:tmax]
                norm_top_sim = normalize_waveform(top_sim)[tmin:tmax]
                norm_btm_sim = normalize_waveform(btm_sim)[tmin:tmax]
                norm_sam_sim = normalize_waveform(sam_sim)[tmin:tmax]
                
                # resample signals
                ieeg_sig = signal.resample(norm_ieeg, n_samples)
                top_sig = signal.resample(norm_top_sim, n_samples)
                btm_sig = signal.resample(norm_btm_sim, n_samples)
                sam_sig = signal.resample(norm_sam_sim, n_samples)
        
                # compute rho
                # rho_val = compute_rho(ieeg_sig, other_sig)

                similarity_vals_subj_array[trial, i, j, 0] = compute_rho(ieeg_sig, top_sig)
                similarity_vals_subj_array[trial, i, j, 1] = compute_rho(ieeg_sig, btm_sig)
                similarity_vals_subj_array[trial, i, j, 2] = compute_rho(ieeg_sig, sam_sig)


        
                subjs_array[trial, i] = subj
                trial_array[trial, i] = trial
                ISI_array[trial, i] = trial_info['SOA']
                SOA_array[trial, i] = trial_info['SOA Jitter']
                n_tones_array[trial, i] = trial_info['Standard #'] +1
                channel_array[trial, i] = channel
            #similarity_vals_subj_array[trial, i] = rho_val
    
    col_vals = [subjs_array.flatten(), trial_array.flatten(), ISI_array.flatten(), SOA_array.flatten(),
                n_tones_array.flatten(), channel_array.flatten(), 
                similarity_vals_subj_array[:,:,0,0].flatten(), similarity_vals_subj_array[:,:,0,1].flatten(), similarity_vals_subj_array[:,:,0,2].flatten(),
                similarity_vals_subj_array[:,:,1,0].flatten(), similarity_vals_subj_array[:,:,1,1].flatten(), similarity_vals_subj_array[:,:,1,2].flatten(),
                similarity_vals_subj_array[:,:,2,0].flatten(), similarity_vals_subj_array[:,:,2,1].flatten(), similarity_vals_subj_array[:,:,2,2].flatten(),
                similarity_vals_subj_array[:,:,3,0].flatten(), similarity_vals_subj_array[:,:,3,1].flatten(), similarity_vals_subj_array[:,:,3,2].flatten(),
                similarity_vals_subj_array[:,:,4,0].flatten(), similarity_vals_subj_array[:,:,4,1].flatten(), similarity_vals_subj_array[:,:,4,2].flatten(),
                similarity_vals_subj_array[:,:,5,0].flatten(), similarity_vals_subj_array[:,:,5,1].flatten(), similarity_vals_subj_array[:,:,5,2].flatten(),
                similarity_vals_subj_array[:,:,6,0].flatten(), similarity_vals_subj_array[:,:,6,1].flatten(), similarity_vals_subj_array[:,:,6,2].flatten(),
                similarity_vals_subj_array[:,:,7,0].flatten(), similarity_vals_subj_array[:,:,7,1].flatten(), similarity_vals_subj_array[:,:,7,2].flatten()
                ]
    
    similarity_dict = dict(zip(col_names, col_vals))
    similarity_df = pd.DataFrame(similarity_dict)    
    df_list.append(similarity_df)

master_df = pd.concat(df_list)
master_df.to_csv('/Users/sydneysmith/Projects/PrOPHEcy/PrOPHEcy/03-results/ieeg/model_comparison/parameter_sweep/parameter_sweep2.csv')

print('analysis finished and CSV saved')
