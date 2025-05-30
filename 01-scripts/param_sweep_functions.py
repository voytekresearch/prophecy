"""CONTAINS ALL FUNCTIONS FOR PROPHECY ANALYSIS"""


import numpy as np
from scipy import stats, signal
import pandas as pd
from neurodsp import sim, filt

#from scipy.stats import zscore
#from scipy.signal import medfilt
#from scipy.signal import resample

#from neurodsp.timefrequency import amp_by_time

import mne
from neurodsp import filt 


####### iEEG Functions #######

def load_fif_epo(subj, control=False):
    """load iEEG epo.fif file
    
    Parameters:
    -----------
    subj : str
        subject ID number
    ref : str
        reference type, 'avg' or 'bipolar'
        
    Returns:
    --------
    epo : object
        instance of MNE Epochs 
    """

    if control:
        print('loading RANDOMLY epoched data...')
        path = '/Volumes/bvoytek/Smith/UCSD_iEEG/projects/PrOPHECy/' + subj + '/' + subj + '_control_epo.fif'
    else:
        #print('loading 1st tone epoched data...')
        #path = '/Volumes/bvoytek/Smith/UCSD_iEEG/projects/PrOPHECy/' + subj + '/' + subj + '_BIP_epo.fif'
        path = '/Users/sydneysmith/Projects/PrOPHEcy/PrOPHEcy/00-data/epo_fifs/' + subj + '_BIP_epo.fif'
        #path = '/labs/bvoyteklab/Smith/prophecy/00-data/' + subj + '_BIP_epo.fif'
    epo = mne.read_epochs(path, preload=True)
    
    return epo


####### ANALYSIS HELPER FUNCTIONS #######
def name_cols(list1, list2, list3):
    """generate column names as combination of listed features
    """
    names1 = []
    names2 = []
    for a in list1:
        for b in list2:
            names1.append('_'.join([a,b]))

    for c in names1:
        for d in list3:
            names2.append('_'.join([c,d]))
            
    return names2


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def CenterBidirectional(data):
    return data/2 + 0.5


def longest_continuous_true(arr):
    """finds the length of the longest continuous instance of True in a boolean array
    Parameters:
    -----------
    arr : array-like
        boolean array input
    
    Returns
    -------
    max_length : int
        length of longest continuous True in arr
    """
    
    max_length = 0
    current_length = 0
    
    for value in arr:
        if value:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    
    return max_length


####### SIMULATION FUNCTIONS ########

def get_raw_sim_sigs(signals, tmin, tmax, n_samples):
    """get resampled, normalized, unfiltered segment of simulated signal
    
    Parameters
    ----------
    signals : list
        list of 1d arrays [tp_lfp, btm_lfp, combined_lfp]
    tmin : int
        start of segment in ms (samples)
    n_samples : int
        number of samples for resampled data segment
    
    Returns
    -------
    resamp_sim_sigs_raw : list of 1d arrays
        resampled, normalized simulated signals, same order as input
    
    """

    sim_sigs_raw = []

    for sig in signals:
        norm_sig = normalize_waveform(sig)[tmin:tmax]
        sim_sigs_raw.append(norm_sig)
    
    resamp_sim_sigs_raw = []
    
    for sig in sim_sigs_raw:
        resamp_sim_sigs_raw.append(signal.resample(sig, n_samples))
    
    return resamp_sim_sigs_raw
    
########## TOP-DOWN BOTTOM-UP FUNCTIONS ###########
def get_tone_onsets_btm(n_tones, stim_length, isi_length, soa_length, edge_length):
    """get dictionary of tone onsets based on stim length and isi
    Parameters
    ----------
    n_tones : int
        number of tones in trial
    stim_length : int
        length of auditory stimulus, in ms
    isi_length : int
        length of interstimulus interval (ISI), in ms
    soa_length : int
        length of stimulus onset asynchrony (SOA), in ms
    edge_length : int
        length of signal before first tone onset, in ms
        
    Returns
    -------
    tone_onsets : dict
        dictionary with tone numbers & onsets, in ms
    """
    
    keys = ['tone 1']
    vals = [edge_length]
    
    adj_isi_length = isi_length - stim_length
    
    for n in range(2, n_tones):
        tone_n = vals[-1] + stim_length + adj_isi_length
        vals.append(tone_n)
        keys.append('tone '+ str(n))
    
    tone_n = vals[-1] + stim_length + (isi_length - stim_length + soa_length)
    vals.append(tone_n)
    keys.append('tone' + str(n+1))
    
    tone_onsets = dict(zip(keys, vals))
    
    return tone_onsets

def get_tone_onsets_tp(n_tones, stim_length, isi_length, soa_length, edge_length):
    """get dictionary of predicted tone onsets based on stim length and isi
    Parameters
    ----------
    n_tones : int
        number of tones in trial
    stim_length : int
        length of auditory stimulus, in ms
    isi_length : int
        length of interstimulus interval (ISI), in ms
    soa_length : int
        length of stimulus onset asynchrony (SOA), in ms
    edge_length : int
        length of signal before first tone onset, in ms
        
    Returns
    -------
    tone_onsets : dict
        dictionary with tone numbers & onsets, in ms
    """
    
    keys = ['tone 1']
    vals = [edge_length]
    
    adj_isi_length = isi_length - stim_length
    
    for n in range(2, n_tones+1):
        tone_n = vals[-1] + stim_length + adj_isi_length
        vals.append(tone_n)
        keys.append('tone '+ str(n))
    
    tone_onsets = dict(zip(keys, vals))
    
    return tone_onsets


def normalize_waveform(signal):
    """Normalize waveform (for simulated cycles)
    Paramters
    ---------
    signal : np array
        simulated signal
    Returns
    -------
    signal: 1d array
        normalized signal"""
    
    signal = signal - np.min(signal)
    signal = signal / np.max(signal)
    
    return signal

def sim_cycle_aud_nerve(stim_length, i, adapt_rate):
    """Simulate a single cyce of an auditory nerve response, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3931124/
    Parameters
    ----------
    stim_length : int
        length of auditory stimulus, in ms
    i : int
        position of cycle in sequence
    adapt_rate : float
        rate of adaptation per cycle in sequence (fraction of decreased amplitude)
    
    Returns
    -------
    cycle: 1d array
        simmulated cycle of an auditory nerve response"""
    
    amp_r = 1.0
    #amp_r = amp_r-amp_r*i*adapt_rate
    tau_r = 10.0
    amp_st = 0.25
    tau_st = 100.0
    amp_base = 0.5
    #amp_base = amp_base-amp_base*i*adapt_rate
    
    times = np.arange(0, stim_length)
    cycle = amp_r*np.exp(-times/tau_r) + amp_st*np.exp(-times/tau_r) + amp_base
    
    return cycle

def sim_cycle_tp_dwn(stim_length, std, amp):
    """simulate single cycle of gaussian top-down exciatory prediction
    Parameters
    ----------
    stim_length: int
        length of auditory stimulus, in ms
    std : float
        width, as standard dev, of gaussian
    amp : float
        amplitude of simulated cycle
        
    Returns
    -------
    cycle: 1d array
        simulated cycle of a top-down prediction"""
    
    center=stim_length/2
    
    times = np.arange(0, stim_length)
    cycle = amp*np.exp(-(times-center)**2 / (2*std**2))
    
    return cycle

def get_spike_prob(fr_rest, fr_stim, signal):
    """get array of spiking probability between rest and sitmulus response based on simulated stimulus sequence
    Parameters
    ----------
    fr_rest : float
        minimum probability of neuron spiking at any point
    fr_stim : float
        maximum probability of neuron spiking at any point
    signal : 1d array
        simulated top-down or bottom-up probability signal 
        
    Returns
    -------
    spike_prob: 1d array
        probability of spiking based on simulated stimulus sequence"""
    
    spike_prob = np.interp(signal,
                          (signal.min(), signal.max()), (fr_rest, fr_stim))
    
    return spike_prob

def sim_bottom_up(tone_onsets, fs, stim_length, trial_length, fr_rest, fr_stim, adapt_rate, ker):
    """simulate bottom-up signal to predictive timing task
    Parmeters
    ---------
    tone_onsets : dict
        dictionary of tone_onsets
    fs : int
        sampling rate of simulated signal (1000 Hz is best)
    stim_length : int
        length of auditory stimulus, in ms
    trial_length : int
        length of trial, in ms
    fr_rest : float
        minimum probability of neuron spiking at any point
    fr_stim : float
        maximum probability of neuron spiking at any point
    adapt_rate : float
        rate of precise temporal prection in bottom-up signal (fraction of cycle width decrease)
    ker : 1d array
        synaptic kernel to convolve with spiking array
    
    Returns
    -------
    btm_up_sig : 1d array
        simulated bottom-up signal"""
    
    pa_latency = 25 #ms of earliest cortical response (Katz, 2002)

    btm_up_spikes = np.zeros(trial_length)
    
    for i, tone_n in enumerate(tone_onsets):
        tone = int(tone_onsets[tone_n] + pa_latency) 
        if i == len(tone_onsets)-1:
            btm_up_spikes[tone:tone+stim_length] = sim_cycle_aud_nerve(stim_length, 0, adapt_rate)
        else:
            btm_up_spikes[tone:tone+stim_length] = sim_cycle_aud_nerve(stim_length, i, adapt_rate)
        
    
    # normalize
    norm_spikes = normalize_waveform(btm_up_spikes)
    
    # get overall spike probabily between rest and baseline
    btm_up_spike_prob = get_spike_prob(fr_rest, fr_stim, norm_spikes)

    # fix convolution delay
    conv_delay = 250
    btm_up_spike_prob = np.pad(btm_up_spike_prob, (conv_delay,0), 'edge')
    btm_up_spike_prob = btm_up_spike_prob[:-250]
    
    # get lfp
    btm_up_sig = sim_spikes(btm_up_spike_prob, ker, trial_length)
    
    return btm_up_spikes, btm_up_sig


def sim_top_down(tone_onsets, fs, stim_length, learning_rate, trial_length, fr_rest,
                 fr_stim, sens_rate, ker):
    """
    Parameters
    ----------
    tone_onsets : dict
        dictionary of tone_onsets
    fs : int
        sampling rate of simulated signal (1000 Hz is best)
    stim_length : int
        length of auditory stimulus, in ms
    learning_rate : float
        fraction of sensitization (increase in amplitude) with each subsequent cycle
    trial length : float
        length of trial, in ms
    n_neurons : int
        number of neurons in simulated population
    fr_rest : float
        minimum probability of neuron spiking at any point
    fr_stim : float
        maximum probability of neuron spiking at any point
    sens_rate : float
        rate of sensitization in each subsequent cycle (amplitude increase)
    ker : 1d array
        simulated synaptic kernel to convolve with spiking array    
    
    Returns
    -------
    tp_down_spikes : 1d array
        array of simulated spiking neurons
    tp_dwn_sign : 1d array
        array of simulated lfp
    
    """
    conv_delay = 250 #ms of bias induced by convolution

    tp_dwn_spikes = np.zeros(trial_length)
        
    stim_window=stim_length*2.5
    
    stim_ind = int(stim_window/2)
    
    for i, tone_n in enumerate(tone_onsets):
        if i == 0:
            pass
        else:
            tone = int(tone_onsets[tone_n])
            tp_dwn_spikes[tone-stim_ind:tone+stim_ind] = sim_cycle_tp_dwn(stim_window, 
                                                                          std=100-100*i*learning_rate,
                                                                          amp=1*i*sens_rate)
    
    # normalize
    norm_spikes = normalize_waveform(tp_dwn_spikes)
    
    #get overall spike probability between rest & baseline
    tp_dwn_spike_prob = get_spike_prob(fr_rest, fr_stim, norm_spikes)

    # fix convolution delay
    conv_delay = 250
    tp_dwn_spike_prob = np.pad(tp_dwn_spike_prob, (conv_delay,0), 'edge')
    tp_dwn_spike_prob = tp_dwn_spike_prob[:-250]
    
    # get lfp
    tp_dwn_sig = sim_spikes(tp_dwn_spike_prob, ker, trial_length)
    
    return tp_dwn_spikes, tp_dwn_sig

def sim_spikes(spike_prob, ker, trial_length):
    """simulate lfp from given spiking probability
    Parameters
    ----------
    spike_prob : 1d array
        simulated arrray of spiking probability
    ker : 1d array
        simulated synaptic kernel to convolve with spikes
    trial_length : int
        length of trial, in ms
        
    Returns
    -------
    lfp: 1d array
        spike probability signal convlved with spiking kernel
    """
    
    # simulate spikes at probability in signal
    random_array = np.random.uniform(low=0.0, high=1.0, size=trial_length)
    spiking_array = spike_prob > random_array
    
    #convolve spiking with synaptic kernel
    lfp = np.convolve(spiking_array, ker, 'same') / sum(ker)
    
    return lfp


def simulate_trial_lfps(trial_info, trial_length):
    """simulate lfps from given iEEG trial
    Parameters
    ----------
    trial_info : dict
        dictionary of trial info from iEEG metadata
    trial_length : int
        length of trial, in ms
        
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
    tau_d = 0.002 #decay timescale of synaptic kernel (s)
    t_ker = 0.05 #total duration of synaptic kernel (s)

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
    combined_lfp = np.sum([btm_neurons.sum(0), tp_neurons.sum(0)], axis=0)
    
    return tp_lfp, btm_lfp, combined_lfp


################# SAM SIMULATION ####################

# settings
PARAMS_DICT = {
    "Wut": 6,
    "Wuv": 6,
    "Wvt": 6,
    "Wvu": 6,
    "dt": 10,
    "tau": 100,
    "y0": 0.7,
    "IF": 50,
    "uinit": 0.7,
    "vinit": 0.2,
    "yinit": 0.5,
    "first_duration": 800,
    "regime": 0,
}

n_neurons = 100

def trial_onset_signal(trial_info, trial_length):
    """generates array of zeros with ones for 10ms at tone onset
    
    Parameters:
    -----------
    trial_info: dict
        trial metadata from epochs object
        
    Returns:
    --------
    trial_onset_signal: 1d array
        array of zeros with ones at tone onsets for one trial
    
    
    """
    # extract trial info
    n_tones = trial_info['Standard #'] + 1 #number of tones in the trial
    
    isi_length = trial_info['SOA'] #interstimulus interval of tone sequence in ms
    soa_length = int(round(trial_info['SOA Jitter'] - isi_length)) #soa
    stim_length = 200
    edge_length = 800
    
    # stimulus info
    tone_onsets = get_tone_onsets_btm(n_tones, stim_length, isi_length, soa_length, edge_length)
    #trial_length = get_trial_length(n_tones, stim_length, isi_length, soa_length, edge_length)
    #print(trial_length)
    # model parameters (set from Jazayeri 2020)

    duration = isi_length
    dt = PARAMS_DICT['dt']
    
    # generate fake signal with ones where each tone is
    siglen = trial_length
    signal = np.zeros([int(siglen / dt), 1])
    #signal = np.zeros([int(siglen),1])
    for t in tone_onsets:
        ind = tone_onsets[t]+25
        signal[int(ind/dt)] = 1
        #signal[int(ind)] = 1

    return signal


def thresh_exp(x):
    """Sigmoid non-linearity."""
    return 1 / (1 + np.exp(-x))


def start_simulation_parallel(state_init, params, K, sigma, niter, regime, signal=None):
    """Run simulation over multiple trials for a period.

    Inputs:
    state_init: an array which includes:
        * I: initial current
        * u: initial state of u
        * v: initial state of v
        * y: initial state of y (readout neuron)
        * sig: state indicator (0 or 1)

    params: a dictionary of relevant parameters of the network, see PARAMS_DICT
    niter: number of iterations
    signal (2d array): an array the length of niter containing the entirety of s-values
        for the run. Must be 2D even if there is only one signal.

    Outputs: each list contains niter elements
    u_lst: list of u activities
    v_lst: list of v activities
    y_lst: list of y activities
    I_lst: list of I activities
    sig_lst: list of sig in this simulation

    regime: 0 = intermediate I, 1 = low I regime, 2 = high I regime

    """
    # signal is the entire time course of s, must be 0 or 1
    if signal is not None:
        if niter is None:
            niter = len(signal)
        assert np.in1d(np.unique(signal), np.array([0, 1])).all()
        assert len(signal) == niter
        # save K value elsewhere so you can reset K to zero and back
        K_orig = K
        K = K_orig * np.ones(signal.shape[1])
    # Unpack parameters of the simulation
    Wut = params["Wut"]
    Wuv = params["Wuv"]
    Wvt = params["Wvt"]
    Wvu = params["Wvu"]
    dt = params["dt"]
    tau = params["tau"]
    IF = params["IF"]
    y0 = params["y0"]  # The target (threshold) value of y

    # Unpack variables
    I, u, v, y, sig = state_init
    ntrials = len(I)

    I = I.copy()
    u = u.copy()
    v = v.copy()
    y = y.copy()

    sig_lst = []
    u_lst = []
    v_lst = []
    y_lst = []
    I_lst = []

    sig_counter = 0

    for i in range(niter):
        if signal is not None:
            sig = signal[i]
            sig_counter += sig
            #print(sig_counter)
            # set K to zero on first input as first time is random.
            K[sig_counter == 1] = 0
            K[sig_counter > 1] = K_orig
        # Update I, u, v and y
        if regime == 0 or regime == 2:
            I += (sig * K * (y - y0)) / tau * dt
        elif regime == 1:
            I -= (sig * K * (y - y0)) / tau * dt

        if regime == 0 or regime == 1:
            u += (
                (
                    -u
                    + thresh_exp(
                        Wut * I - Wuv * v - sig * IF + np.random.randn(ntrials) * sigma
                    )
                )
                / tau
                * dt
            )
            v += (
                (
                    -v
                    + thresh_exp(
                        Wvt * I - Wvu * u + sig * IF + np.random.randn(ntrials) * sigma
                    )
                )
                / tau
                * dt
            )
            y += (-y + u - v + np.random.randn(ntrials) * sigma) / tau * dt
        elif regime == 2:
            u += (
                (
                    -u
                    + thresh_exp(
                        Wut * I - Wuv * v + sig * IF + np.random.randn(ntrials) * sigma
                    )
                )
                / tau
                * dt
            )
            v += (
                (
                    -v
                    + thresh_exp(
                        Wvt * I - Wvu * u - sig * IF + np.random.randn(ntrials) * sigma
                    )
                )
                / tau
                * dt
            )
            y += (-y + u - v + np.random.randn(ntrials) * sigma) / tau * dt

        v_lst.append(v.copy())
        u_lst.append(u.copy())
        y_lst.append(y.copy())
        I_lst.append(I.copy())
        sig_lst.append(sig)

    return u_lst, v_lst, y_lst, I_lst, sig_lst



def simulate_trial(ntrials, duration, nstages, sigma, K, initI, signal=None, y0=None):
    """Simulate a complete trial."""
    # Initial run
    first_duration = PARAMS_DICT[
        "first_duration"
    ]  # duration in ms of first duration (500 ms + exponential with mean 250)
    regime = PARAMS_DICT["regime"]

    nbin = int(duration / PARAMS_DICT["dt"])
    nbinfirst = int(first_duration / PARAMS_DICT["dt"])

    uinit = PARAMS_DICT["uinit"]
    vinit = PARAMS_DICT["vinit"]
    yinit = PARAMS_DICT["yinit"]

    state_init = [
        np.ones(ntrials) * initI,
        np.ones(ntrials) * uinit,
        np.ones(ntrials) * vinit,
        np.ones(ntrials) * yinit,
        0.0,
    ]

    if signal is None:
        ulst, vlst, ylst, Ilst, siglst = start_simulation_parallel(
            state_init, PARAMS_DICT, 0, sigma, nbinfirst, regime
        )

        # For subsequent runs, flip the state every 100 trials
        for k in range((nstages - 2) * 2):

            # acoefs = 1 - acoefs
            state_init = [
                Ilst[-1],
                ulst[-1],
                vlst[-1],
                ylst[-1],
                (state_init[4] + 1.0) % 2,
            ]
            # print('k = ', k, 'state_init[4] =', state_init[4])
            if state_init[4] == 0.0:
                ulst2, vlst2, ylst2, Ilst2, siglst2 = start_simulation_parallel(
                    state_init, PARAMS_DICT, K, sigma, nbin, regime
                )
            else:
                if k == 0:
                    # No update for first flash
                    ulst2, vlst2, ylst2, Ilst2, siglst2 = start_simulation_parallel(
                        state_init, PARAMS_DICT, 0, sigma, 1, regime
                    )
                else:
                    ulst2, vlst2, ylst2, Ilst2, siglst2 = start_simulation_parallel(
                        state_init, PARAMS_DICT, K, sigma, 1, regime
                    )

            ulst += ulst2
            vlst += vlst2
            ylst += ylst2
            Ilst += Ilst2
            siglst += siglst2

        if nstages > 1:
            state_init = [
                Ilst[-1],
                ulst[-1],
                vlst[-1],
                ylst[-1],
                (state_init[4] + 1.0) % 2,
            ]

            if nstages == 2:
                Keff = 0
            else:
                Keff = K

            # For the last run, produce the behavior when the threshold is reached
            ulst2, vlst2, ylst2, Ilst2, siglst2 = start_simulation_parallel(
                state_init, PARAMS_DICT, Keff, sigma, 1, regime
            )

            ulst += ulst2
            vlst += vlst2
            ylst += ylst2
            Ilst += Ilst2
            siglst += siglst2

            state_init = [
                Ilst[-1],
                ulst[-1],
                vlst[-1],
                ylst[-1],
                (state_init[4] + 1.0) % 2,
            ]
            # For the last run, produce the behavior when the threshold is reached
            ulst2, vlst2, ylst2, Ilst2, siglst2 = start_simulation_parallel(
                state_init, PARAMS_DICT, K, sigma, nbin * 2, regime
            )

            ulst += ulst2
            vlst += vlst2
            ylst += ylst2
            Ilst += Ilst2

            siglst2[nbin] = 1
            siglst += siglst2
        else:
            print(len(siglst))
            siglst[-1] = 1
            ylst2 = ylst
    else:
        if y0 is not None:
            PARAMS_DICT["y0"] = y0
        ulst, vlst, ylst, Ilst, siglst = start_simulation_parallel(
            state_init, PARAMS_DICT, K, sigma, len(signal), regime, signal
        )
        ylst2 = ylst

    return ulst, vlst, ylst, Ilst, siglst, ylst2


def sim_sam_trial(trial_info, trial_length):
    """simulates a ramping signal for a trial based on Jazayeri 2020 model
    
    Parameters:
    -----------
    trial_info: dict
        trial metadata from epochs object
        
    Returns:
    --------
    e_sig: 1d array
        array of excitatory population acitivty (u population)
    i_sig: 1d array
        array of inhibitory population activity (v popualtion)
    y_sig: 1d array
        array of excitatory (comparator) population activity (y population)
    
    """
    signal =trial_onset_signal(trial_info, trial_length)

    duration = trial_info['SOA']
    n_tones = trial_info['Standard #'] +1

    sigma = 0.005
    initI = 0.78
    K = 6.0
    
    # run simulation
    ulst, vlst, ylst, Ilst, siglst, ylst2 = simulate_trial(ntrials=1,
                                                           nstages=n_tones,
                                                           duration=duration,
                                                           sigma=sigma,
                                                           K=K,
                                                           initI=initI,
                                                           signal=signal)   
    
    # interpolate simualted signals to match length of trial
    interp_sigs = []
    xvals = np.arange(0,trial_length) # new x-axis for interpolation
    
    for sig in [ulst, vlst, ylst]:
        sigarr = np.asarray(sig).flatten()
        xorig = np.linspace(0,trial_length,len(sig)) # original x-axis to interpolate
        interp_sig = np.interp(xvals, xorig, sigarr)
        interp_sigs.append(interp_sig)
    
    e_sig = interp_sigs[0]
    i_sig = interp_sigs[1]
    y_sig = interp_sigs[2]
    
    return e_sig, i_sig, y_sig # y sig is the comparator signal we want

def sim_lfp(y_sig, ker, n_neurons, trial_length, fr_rest, fr_stim):
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
    y_sig = np.pad(y_sig, (conv_delay,0), 'edge')
    y_sig = y_sig[:-250]


    for n in range(0, n_neurons):
        norm_spikes = normalize_waveform(y_sig)
        spike_prob = get_spike_prob(fr_rest, fr_stim, norm_spikes)
        sam_sig = sim_spikes(spike_prob, ker, trial_length)
        sam_sigs_all[n,:] = sam_sig

    sam_lfp = normalize_waveform(sam_sigs_all.sum(0))

    return sam_lfp



def simulate_sam_lfp(trial_info, trial_length):
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
    tau_d = 0.002 #decay timescale of synaptic kernel (s)
    t_ker = 0.05 #total duration of synaptic kernel (s)

    i_tau_r = 0.0005 #rise of inhibitory synaptic kernel (s)
    i_tau_d = 0.01 #decay of inhibitory synaptic kernel (s)
    i_t_ker = 0.05 #total duration of synaptic kernel (s)

    ker = sim.sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    i_ker = sim.sim_synaptic_kernel(i_t_ker, fs, i_tau_r, i_tau_d)*-1 # inverted polarity

    e_sig, i_sig, y_sig = sim_sam_trial(trial_info, trial_length)
    sam_lfp_y = sim_lfp(y_sig, ker, n_neurons, trial_length, fr_rest, fr_stim)
    sam_lfp_e = sim_lfp(e_sig, ker, n_neurons, trial_length, fr_rest, fr_stim)
    sam_lfp_i = sim_lfp(i_sig, i_ker, n_neurons, trial_length, fr_rest, fr_stim)

    return sam_lfp_y, sam_lfp_e, sam_lfp_i


######### SIMILARITY FUNCTIONS ##########

def compute_r(sig1, sig2):
    """compute pearson r between 2 signals"""
    r, _ = stats.pearsonr(sig1, sig2)
    return r

def compute_euc(sig1, sig2):
    """compute euclidian distance between 2 signals"""
    
    euc = np.sqrt(np.sum((sig1 - sig2) ** 2))
    
    return euc

def compute_rho(sig1, sig2):
    """compute spearman rho statistic between 2 signals"""

    rho, _ = stats.spearmanr(sig1, sig2)
    return rho

