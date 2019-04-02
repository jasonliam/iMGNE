import os
import numpy as np
from itertools import tee
from scipy.io import wavfile
import wave  # for phase vocoder
from scipy.signal import stft, istft
import torch
import torchvision
import torch.nn as nn
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
import librosa
import lws


class DataGenerator:

    def __init__(self, fpaths, chunk_size, window_size, window_overlap,
                 batch_size=1, downsample=1, sr=8000, #wav_format="PCM16",
                 vocoder=False, verbose=False):

        self.fpaths = fpaths
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.window_overlap = window_overlap  # sample rate at which to do STFT
        self.batch_size = batch_size
        self.sr = sr
        self.downsample = downsample  # average every n STFT samples
#         self.wav_format = wav_format

        self.use_phase_vocoder = vocoder
        self.verbose = verbose

        self.X_list = []  # list of FloatTensors representing each song in frequency space
        self.T_list = []

        # song indices, used for shuffling songs
        self.idx = range(len(self.fpaths))

        self._load()  # load and preprocess all songs in fpaths

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, i):
        return self.fpaths[self.idx[i]], self.X_list[self.idx[i]], self.T_list[self.idx[i]]

    def _load(self):
        if self.use_phase_vocoder:
            self._load_stft_vocoded()
        else:
            self._load_stft()

    def _load_stft_vocoded(self):

        lws_proc = lws.lws(self.window_size, self.window_overlap,
                           mode='music', perfectrec=True)

        for fpath in self.fpaths:

            x, fs = librosa.core.load(fpath, sr=None, mono=True)
            print("Loading file: {}, sample rate: {}".format(fpath, fs))
            print(x.shape, fs, x.shape[0] /
                  float(fs), x.dtype, x.min(), x.max())

            X = lws_proc.stft(x)

            # Extract magnitude and phase spectra
            X_mag = np.abs(X)
            X_phs = np.angle(X)

            # Unwrap phase
            X_phs_unwrapped = np.unwrap(X_phs, axis=0)

            # Find "instantaneous frequency" representation (delta phase)
            X_delta_phs = np.diff(X_phs_unwrapped, axis=0)
            X_delta_phs_padded = np.pad(
                X_delta_phs, [[1, 0], [0, 0]], 'constant')

            #stacked = np.stack([X_mag, X_delta_phs_padded], axis=2)
            #X = np.pad(zxx, [[0,1],[0,1],[0,0]], 'constant')
            #T = np.pad(zxx, [[1,0],[1,0],[0,0]], 'constant')

            zxx = np.vstack((X_mag.T, X_delta_phs_padded.T))

            # make raw train and teacher sets from STFT'd data
            xt_padding = np.zeros((zxx.shape[0], 1))
            X = np.hstack((xt_padding, zxx)).transpose()
            T = np.hstack((zxx, xt_padding)).transpose()
            assert (X[1:] == T[:-1]).all()

            # pad X and T at the end to the nearest multiple of batch_size*chunk_size (0 as silence)
            max_pad_len = self.batch_size * self.chunk_size
            X = np.vstack(
                (X, np.zeros((max_pad_len - X.shape[0] % max_pad_len, X.shape[1]))))
            T = np.vstack(
                (T, np.zeros((max_pad_len - T.shape[0] % max_pad_len, T.shape[1]))))

            # make minibatchs
            X = torch.FloatTensor(X).view(
                self.batch_size, -1, zxx.shape[0]).transpose(0, 1)
            T = torch.FloatTensor(T).view(
                self.batch_size, -1, zxx.shape[0]).transpose(0, 1)

            print("Data processing complete, X shape: {}".format(X.shape))
            print()

            self.X_list += [X]
            self.T_list += [T]

    def _load_stft(self):
        for i, fpath in enumerate(self.fpaths):

            # load wav file into 1-D ndarray
            data, sr = librosa.core.load(fpath, sr=None, mono=True)
            if sr != self.sr:
                data = librosa.core.resample(data, sr, self.sr)
                sr = self.sr

            # Do STFT (can't STFT on concat'd music: takes too much ram on long inputs)
            _, _, zxx_c = stft(
                data, fs=sr,  # window='blackmanharris',
                nperseg=self.window_size, noverlap=self.window_overlap)

            # do undersampling by averaging every n STFT samples
            zxx_c = np.average(
                zxx_c.reshape(zxx_c.shape[0], -1, self.downsample), axis=2)

            # break out and concat real and imaginary parts
            zxx = np.vstack((np.real(zxx_c), np.imag(zxx_c)))

            # make raw train and teacher sets from STFT'd data
            xt_padding = np.zeros((zxx.shape[0], 1))
            X = np.hstack((xt_padding, zxx)).transpose()
            T = np.hstack((zxx, xt_padding)).transpose()
            assert (X[1:] == T[:-1]).all()

            # pad X and T at the end to the nearest multiple of batch_size*chunk_size (0 as silence)
            max_pad_len = self.batch_size * self.chunk_size
            X = np.vstack(
                (X, np.zeros((max_pad_len - X.shape[0] % max_pad_len, X.shape[1]))))
            T = np.vstack(
                (T, np.zeros((max_pad_len - T.shape[0] % max_pad_len, T.shape[1]))))

            # make minibatchs
            X = torch.FloatTensor(X).view(
                self.batch_size, -1, zxx.shape[0]).transpose(0, 1)
            T = torch.FloatTensor(T).view(
                self.batch_size, -1, zxx.shape[0]).transpose(0, 1)

            # print("Data processing complete, X shape: {}".format(X.shape))
            stat_str = "File {}/{}, sr={}, zxx_c.shape={}, X.shape={}".format(
                i + 1, len(self.fpaths), sr, zxx_c.shape, X.shape)
            print(stat_str, end='\r')

            self.X_list += [X]
            self.T_list += [T]
        print()

    def randomize_idx(self):
        self.idx = np.random.permutation(self.idx)

    def reassemble_istft(self, X):
        # NOTE: expects X in shape (seq_length, 1, real+imag)

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        temp = X.reshape(-1, X.shape[2]).transpose()

        # reassemble real and imaginary parts of the output
        out = temp[:int(temp.shape[0]/2)] + 1j * temp[int(temp.shape[0]/2):]

        # upsample to original STFT resolution
        out = np.repeat(out, self.downsample, axis=1)

        # do iSTFT
        t, x = istft(out, fs=self.sr,  # window='blackmanharris',
                     nperseg=self.window_size, noverlap=self.window_overlap)

        # TODO: normalize output?

        return t, x
