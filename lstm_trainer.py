""" A class that trains LSTMs.. what did you expect

Important notes on semantics and behavior
    - LSTM state should persist through each training epoch
    - LSTM implicitly applies teacher forcing when given a sequence
    - In self.lstm_seq, model.curr_state is saved after each sequence fwd pass
    - When validation or evaluating, do NOT feed sequences to LSTM; Instead, 
        feed: single characters (idx) for model with embedding layer OR one hot 
        encoded indices for model with no embedding layer. Then, get an output, 
        flip a coin, then feed the input back
    - When training, it's important to detach hidden/cell states from the 
        dataflow DAG to limit BPTT. This prevents BPTT into previous sequences
        NOTE: Tensor.detach_() is different from Tensor.detach()!

TODOs:
    -  After each sequence in train and validation:
        Record output sequence (useful to see learning progress)
        Probably useful to check hidden/cell states            
"""

import numpy as np
import torch
import torchvision
import torch.nn as nn
import pickle
from datetime import datetime


class LSTMTrainer:

    def __init__(self, model, criterion, optimizer, device=None):

        # Set computing device, detect if not passed in
        if not device:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        print('Using {}'.format(self.device))

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs_trained = 0

        self.m = model.module if isinstance(model, nn.DataParallel) else model

    def train(self, train_gen, val_gen, n_epochs, shuffle=True,
              dump_model=False, dump_epochs=50, dump_loss=False):
        """  Train and validate self.model 
        """

        # 2-D arrays from every chunk of every epoch
        self.train_loss = []
        self.val_loss = []

        start_time = datetime.now()
        for epoch in range(n_epochs):

            # ==== train =====
            self.m.train()

            if shuffle:
                train_gen.randomize_idx()
                if val_gen:
                    val_gen.randomize_idx()

            epoch_train_loss = []
            epoch_val_loss = []
            for fpath, X, T in train_gen:

                # wipe model states before every song
                self.m.curr_state = self.m.init_hidden()

                num_chunks = X.shape[0] // train_gen.chunk_size
                chunk_size = train_gen.chunk_size

                for i in range(0, num_chunks):

                    self.optimizer.zero_grad()

                    chunk = X[i*chunk_size:(i+1)*chunk_size].to(self.device)
                    teacher = T[i*chunk_size:(i+1)*chunk_size].to(self.device)

                    out, _ = self.model(chunk, self.m.curr_state)

                    loss = self.criterion(out, teacher)
                    epoch_train_loss += [loss.item()]

                    loss.backward()
                    self.optimizer.step()

                    self.m.curr_state[0].detach_()
                    self.m.curr_state[1].detach_()

                    # print status bar
                    ic = int(i * 30 / num_chunks) + 1
                    time_elapsed = datetime.now() - start_time
                    stat_str = "[Train] Epoch {:04d} | {:<20.20s} | Chunk {:03d} [{}{}]{:05.1f}% | T+{}" .format(
                        self.epochs_trained+1, fpath.split('/')[-1], i+1, '#'*ic, '-'*(30-ic), 
                        i/num_chunks*100, str(time_elapsed))
                    print(stat_str, end='\r')

            self.train_loss += [epoch_train_loss]

            # ===== validation =====
            self.m.eval()
            with torch.no_grad():
                
                for fpath, X, T in val_gen:

                    # wipe model states before every song
                    self.m.curr_state = self.m.init_hidden()

                    num_chunks = X.shape[0] // val_gen.chunk_size
                    chunk_size = val_gen.chunk_size

                    for i in range(0, num_chunks):

                        chunk = X[i*chunk_size:(i+1)*chunk_size].to(self.device)
                        teacher = T[i*chunk_size:(i+1)*chunk_size].to(self.device)

                        out, _ = self.m(chunk, self.m.curr_state)

                        loss = self.criterion(out, teacher)
                        epoch_val_loss += [loss.item()]

                        # print status bar
                        ic = int(i * 30 / num_chunks) + 1
                        time_elapsed = datetime.now() - start_time
                        stat_str = "[Valdn] Epoch {:04d} | {:<20.20s} | Chunk {:03d} [{}{}]{:05.1f}% | T+{}" .format(
                            self.epochs_trained+1, fpath.split('/')[-1], i+1, '#'*ic, '-'*(30-ic), 
                            i/num_chunks*100, str(time_elapsed))
                        print(stat_str, end='\r')

            self.val_loss += [epoch_val_loss]

            # ===== post-epoch stuff =====
            self.epochs_trained += 1

            # dump model
            if dump_model and self.epochs_trained % dump_epochs == 0:
                model_file = "models/cs{}_h{}_e{}.ckpt".format(
                    train_gen.chunk_size, self.m.hidden_dim, self.epochs_trained)
                torch.save(self.m.state_dict(), model_file)

            # dump loss on every epoch
            if dump_loss:
                losses_file = "losses/cs{}_h{}_e{}.loss.pkl".format(
                    train_gen.chunk_size, self.m.hidden_dim, self.epochs_trained)
                pickle.dump((epoch_train_loss, epoch_val_loss),
                            open(losses_file, 'wb'))

        return self.train_loss, self.val_loss

    def load_model(self, model_path, epochs_trained):
        self.m.load_state_dict(torch.load(model_path))
        self.epochs_trained = epochs_trained

    def eval_model(self, primer_data_gen, silence_len=10, prime_len=10, gen_len=10):
        """ Generate some bangers
        """

        # get primer data; use only the first piece
        fpath, X, _ = primer_data_gen[0]

        self.m.eval()

        with torch.no_grad():

            # wipe model
            self.m.curr_state = self.m.init_hidden()

            eval_output = []
            hidden_states = []
            cell_states = []
            
            # hold the input silent for a short period
            # this will yield the network's natural oscillation patterns
            silence = torch.zeros(X.shape).to(self.device)
            for i in range(silence_len):
                out, states = self.m(silence[i:i+1], self.m.curr_state)
                eval_output += [out.cpu().numpy()]
                hidden_states += [states[0].cpu().numpy()]
                cell_states += [states[1].cpu().numpy()]

            # prime the model with some input
            primer = X[:prime_len].to(self.device)
            for i in range(prime_len):
                out, states = self.m(primer[i:i+1], self.m.curr_state)
                eval_output += [out.cpu().numpy()]
                hidden_states += [states[0].cpu().numpy()]
                cell_states += [states[1].cpu().numpy()]

            # start generation 
            for i in range(gen_len):
                print("{}/{}".format(i+1, gen_len), end='\r')
                out, states = self.m(out, self.m.curr_state)
                eval_output += [out.cpu().numpy()]
                hidden_states += [states[0].cpu().numpy()]
                cell_states += [states[1].cpu().numpy()]
                
        return eval_output, hidden_states, cell_states
