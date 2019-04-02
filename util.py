import os
import numpy as np
from itertools import tee
from scipy.io import wavfile
import wave # for phase vocoder
from scipy.signal import stft, istft
import torch
import torchvision
import torch.nn as nn
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
import librosa

class DataGenerator:

    def __init__(self, fpaths, chunk_size, window_size, window_overlap,
                 batch_size=1, downsample=1, sr=8000, wav_format="PCM16",
                 vocoder=False, verbose=False):

        self.fpaths = fpaths
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.window_overlap = window_overlap  # sample rate at which to do STFT
        self.batch_size = batch_size
        self.sr = sr
        self.downsample=downsample  # average every n STFT samples
        self.wav_format = wav_format
        
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
        
        for fpath in self.fpaths:
            with WavReader(fpath) as reader:
                
                sr = reader._reader.getframerate()
                print("Loading file: {}, sample rate: {}".format(fpath, sr))
                                       
                # filename for audiotsm to write stft'd data to
                stft_fname = ''.join(fpath.split('.')[:-1]) + '_stft.npy'
                
                tsm = phasevocoder(reader.channels, 
                                   frame_length=self.window_size, 
                                   analysis_hop=self.window_overlap) 
                                   #synthesis_hop= self.window_size // 4)               
                
                tsm.run(reader, None, stft_fname=stft_fname, stft_only=True, verbose=self.verbose)        

                zxx = np.real(np.load(stft_fname))
                print("Loaded STFT data from {}, with shape: {}".format(stft_fname, zxx.shape))

                # break out and concat real and imaginary parts
                # zxx = np.vstack((np.real(zxx_c), np.imag(zxx_c)))

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
        for fpath in self.fpaths:

            # load wav file into 1-D ndarray
            # sr, data = wavfile.read(fpath)
            # if self.sr:
            #     assert self.sr == sr  # do not allow samples w/different sr
            # self.sr = sr
            data, sr = librosa.load(fpath, sr=self.sr)


            print("Loading file: {}, sample rate: {}".format(fpath, sr))

            # normalize inputs to (-1,1) -- using tanh for activation
            if self.wav_format == "PCM16":  # signed 16-bit integer encoding
                data = data / 2**15
            elif self.wav_format == "PCM32":  # signed 32-bit integer encoding
                data = data / 2**31
            elif self.wav_format == "FLOAT32":  # signed 32-bit float encoding
                pass  # Assume normalized input (if not, training is screwed)
            else:
                raise Exception('Unknown wav format')
            assert data.min() >= -1.0 and data.max() <= 1.0  # enforce input range

            # Do STFT (can't STFT on concat'd music: takes too much ram on long inputs)
            _, _, zxx_c = stft(
                data, fs=sr, #window='blackmanharris',
                nperseg=self.window_size, noverlap=self.window_overlap)

            print("Completed STFT, data shape: {}".format(zxx_c.shape))

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

            print("Data processing complete, X shape: {}".format(X.shape))
            print()

            self.X_list += [X]
            self.T_list += [T]
            
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
        t, x = istft(out, fs=self.sr, #window='blackmanharris',
                     nperseg=self.window_size, noverlap=self.window_overlap)

        # TODO: normalize output?

        return t, x
