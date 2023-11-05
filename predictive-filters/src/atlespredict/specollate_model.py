import torch
import torch.nn as nn
import torch.nn.functional as F

from src.atlesconfig import config


# adding useless comment
class Net(nn.Module):
    def __init__(self, vocab_size, output_size=512, embedding_dim=512, hidden_lstm_dim=1024, lstm_layers=2):
        super(Net, self).__init__()
        self.spec_size = config.get_config(section="input", key="spec_size")
        self.spec_size = 80000
        self.seq_len = config.get_config(section="ml", key="pep_seq_len")
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.hidden_lstm_dim = hidden_lstm_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # dir_path = "/scratch/mtari008/job_" + os.environ['SLURM_JOB_ID'] + "/nist_massiv_80k_inch_aux-semi"
        # aux_means = np.load(join(dir_path, "aux_means.npy"))
        # aux_stds = np.load(join(dir_path, "aux_stds.npy"))
        # self.aux_means = torch.from_numpy(aux_means).float().to(0)
        # self.aux_stds = torch.from_numpy(aux_stds).float().to(0)

        ################### Spectra branch ###################
        # self.conv1_1    = nn.Conv1d(1, 32, 3, stride=1, padding=1)
        # self.maxpool1_1 = nn.MaxPool1d(50, stride=50)
        # self.linear1_1  = nn.Linear(int((self.spec_size*32)/50), 1024) # use with convo layers
        # self.linear1   = nn.Linear(35, 128)
        # self.linear2   = nn.Linear(128, 256)
        self.linear1_1 = nn.Linear(self.spec_size, 512)
        self.linear1_2 = nn.Linear(512, 256)
        # Spectra charge sub-branch
        # self.charge = config.get_config(section="input", key="charge")
        # self.linear_charge1 = nn.Linear(1024, self.charge)

        ################### Peptide branch ###################
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            self.hidden_lstm_dim,
            self.lstm_layers,
            # dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )

        # self.linear2 = nn.Linear(111, 64)
        self.linear2_1 = nn.Linear(self.hidden_lstm_dim * 2, 512)  # 2048, 1024
        self.linear2_2 = nn.Linear(512, 256)
        # NOTE: should the two layers below be just one layer.
        # self.linear_spec_mass1 = nn.Linear(1024, 256)
        # self.linear_spec_mass2 = nn.Linear(256, 1)
        # self.linear_spec_mass3 = nn.Linear(128, 1)
        # self.linear_pep_mass1 = nn.Linear(1024, 256)
        # self.linear_pep_mass2 = nn.Linear(256, 1)
        # self.linear_pep_mass3 = nn.Linear(128, 1)
        do = config.get_config(section="ml", key="dropout")
        self.dropout_conv1_1 = nn.Dropout(do)
        self.dropout1 = nn.Dropout(do)
        self.dropout2 = nn.Dropout(do)
        self.dropout1_1 = nn.Dropout(do)
        self.dropout1_2 = nn.Dropout(do)
        self.dropout1_3 = nn.Dropout(do)
        self.dropout1_4 = nn.Dropout(do)
        self.dropout1_5 = nn.Dropout(do)
        self.dropout2_1 = nn.Dropout(do)
        self.dropout2_2 = nn.Dropout(do)
        self.dropout2_3 = nn.Dropout(do)
        self.dropout2_4 = nn.Dropout(do)
        self.dropout2_5 = nn.Dropout(do)
        self.dropout2_6 = nn.Dropout(do)
        self.dropout_charge1 = nn.Dropout(do)
        self.dropout_spec_mass1 = nn.Dropout(do)
        self.dropout_spec_mass2 = nn.Dropout(do)
        self.dropout_spec_mass3 = nn.Dropout(do)
        self.dropout_pep_mass1 = nn.Dropout(do)
        self.dropout_pep_mass2 = nn.Dropout(do)
        self.dropout_pep_mass3 = nn.Dropout(do)
        print("dropout: {}".format(do))
        # self.dropout3 = nn.Dropout(0.3)

    def forward(self, data, data_type=None):
        assert not data_type or data_type == "specs" or data_type == "peps"
        res = []
        if not data_type or data_type == "specs":
            specs = data[0].squeeze()
            # mass_charge = specs[:, 0:35]
            # specs = specs[:, 35:]
            # out = F.relu(self.conv1_1(specs.view(-1, 1, self.spec_size)))
            # out = self.maxpool1_1(out)
            # out = self.dropout_conv1_1(out)

            # out = F.relu(self.linear1_1(out.view(out.size(0), -1))) # use this line with conv networks

            out = F.relu((self.linear1_1(specs.view(-1, self.spec_size))))
            out = self.dropout1_1(out)

            # mc_out = F.relu(self.linear1(mass_charge.view(-1, 35)))
            # mc_out = self.dropout1(mc_out)
            # mc_out = F.relu(self.linear2(mc_out))
            # mc_out = self.dropout2(mc_out)
            # out = torch.cat((mc_out, out), axis=1)

            out_spec = F.relu(self.linear1_2(out))
            # out_spec = out_spec * mc_out
            out_spec = F.normalize(out_spec)

            # out_ch = self.dropout_charge1(out)
            # out_ch = self.linear_charge1(out_ch)
            # out_ch = F.relu(out_ch)

            # out_mass = self.dropout_spec_mass1(out)
            # out_mass = self.linear_spec_mass1(out_mass)
            # out_mass = F.relu(out_mass)
            # out_mass = self.dropout_spec_mass2(out_mass)
            # out_mass = self.linear_spec_mass2(out_mass)
            # out_mass = F.relu(out_mass)
            # out_mass = self.dropout_spec_mass3(out_mass)
            # out_mass = self.linear_spec_mass3(out_mass)
            res.append(out_spec)
            # res.append(out_ch)
            # res.append(out_mass)
        if not data_type or data_type == "peps":
            for peps in data[1:3]:
                peps = peps.squeeze()
                # aux = peps[:, :111].float()
                # aux = (aux - self.aux_means) / self.aux_stds
                # peps = peps[:, 111:]
                embeds = self.embedding(peps)
                # embeds = self.one_hot_tensor(peps)
                hidden = self.init_hidden(len(peps))
                hidden = tuple([e.data for e in hidden])
                lstm_out, _ = self.lstm(embeds, hidden)
                lstm_out = lstm_out[:, -1, :]
                out = lstm_out.contiguous().view(-1, self.hidden_lstm_dim * 2)
                out = self.dropout2_1(out)

                out = F.relu((self.linear2_1(out)))
                out = self.dropout2_2(out)

                # m_out = F.relu(self.linear2(mass.view(-1, 27)))
                # m_out = self.dropout2(m_out)
                # out = torch.cat((m_out, out), axis=1)

                # aux_out = F.relu(self.linear2(aux))
                # aux_out = self.dropout2(aux_out)
                # out = torch.cat((aux_out, out), axis=1)

                out_pep = F.relu(self.linear2_2(out))
                out_pep = F.normalize(out_pep)

                # out_mass = self.dropout_pep_mass1(out)
                # out_mass = self.linear_pep_mass1(out_mass)
                # out_mass = F.relu(out_mass)
                # out_mass = self.dropout_pep_mass2(out_mass)
                # out_mass = self.linear_pep_mass2(out_mass)
                # out_mass = F.relu(out_mass)
                # out_mass = self.dropout_pep_mass3(out_mass)
                # out_mass = self.linear_pep_mass3(out_mass)
                res.append(out_pep)
                # res.append(out_mass)
        return res

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.lstm_layers * 2, batch_size, self.hidden_lstm_dim).zero_(),
            weight.new(self.lstm_layers * 2, batch_size, self.hidden_lstm_dim).zero_(),
        )
        return hidden

    def one_hot_tensor(self, peps):
        batch_size = len(peps)
        src = torch.zeros((batch_size, self.seq_len), dtype=torch.float16, device="cuda")
        src[peps > 0] = 1.0
        one_hots = torch.zeros((batch_size, self.seq_len, self.vocab_size), dtype=torch.float16, device="cuda")
        one_hots.scatter_(2, peps.view(batch_size, self.seq_len, 1), src.view(batch_size, self.seq_len, 1))
        one_hots.requires_grad = True
        return one_hots

    def name(self):
        return "Net"
