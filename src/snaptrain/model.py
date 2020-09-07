import torch.nn as nn
import torch.nn.functional as F

from src.snapconfig import config


class Net(nn.Module):
    def __init__(self, vocab_size, output_size=512, embedding_dim=512, hidden_lstm_dim=1024, lstm_layers=2):
        super(Net, self).__init__()

        self.spec_size = config.get_config(section='input', key='spec_size')
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.hidden_lstm_dim = hidden_lstm_dim
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_lstm_dim, self.lstm_layers,
                            # dropout=0.5, 
                            batch_first=True, bidirectional=True)
        # self.lstm = nn.DataParallel(self.lstm)
        
        self.linear1_1 = nn.Linear(self.spec_size, 512) # self.spec_size, 1024
        self.linear1_2 = nn.Linear(512, 256)            # 1024, 512
        #self.linear1_3 = nn.Linear(512, 256)

        self.linear2_1 = nn.Linear(self.hidden_lstm_dim * 2, 512) # 2048, 1024
        self.linear2_2 = nn.Linear(512, 256) # 1024, 512
        #self.linear2_3 = nn.Linear(256, 128)

        do = 0.3
        self.dropout1 = nn.Dropout(do)
        self.dropout2 = nn.Dropout(do)
        print("dropout: {}".format(do))
        #self.dropout3 = nn.Dropout(0.3)
        

    def forward(self, data, hidden, data_type=None):
        assert not data_type or data_type == "specs" or data_type == "peps"
        res = []
        if not data_type or data_type == "specs":
            specs = data[0]
            out = self.linear1_1(specs.view(-1, self.spec_size))
            out = F.relu(out)

            out = self.dropout2(out)
            out = self.linear1_2(out)
            out = F.relu(out)
            
            out_spec = F.normalize(out)
            res.append(out_spec)

        if not data_type or data_type == "peps":
            for peps in data[1:]:
                embeds = self.embedding(peps)
                lstm_out, _ = self.lstm(embeds, hidden)
                lstm_out = lstm_out[:, -1, :]
                out = lstm_out.contiguous().view(-1, self.hidden_lstm_dim * 2)

                out = self.dropout1(out)
                out = self.linear2_1(out)
                out = F.relu(out)

                out = self.dropout1(out)
                out = self.linear2_2(out)
                out = F.relu(out)

                out_pep = F.normalize(out)
                res.append(out_pep)
        res.append(hidden)

        return res

    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers * 2, batch_size, self.hidden_lstm_dim).zero_(),
                      weight.new(self.lstm_layers * 2, batch_size, self.hidden_lstm_dim).zero_())
        return hidden
    
    def name(self):
        return "Net"