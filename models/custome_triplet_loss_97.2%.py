class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_size = config.get_config(section='input', key='spec_size')
        self.searching = False
        
        self.linear1_1 = nn.Linear(self.spec_size, 1024)
        self.linear1_2 = nn.Linear(1024, 512)
        #self.linear1_3 = nn.Linear(512, 256)
        
        self.linear2_1 = nn.Linear(self.spec_size, 1024)
        self.linear2_2 = nn.Linear(1024, 512)
        #self.linear2_3 = nn.Linear(512, 256)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        #self.dropout3 = nn.Dropout(0.3)
        
    def forward(self, data):
        res = []
#         x = data[:, 0]
#         x = self.linear1_1(x.view(-1, self.spec_size))
#         x = F.relu(x)
#         x = self.dropout2(x)
        #x = self.linear1_2(x)
        #x = F.relu(x)
        #x = self.dropout2(x)
        #x = self.dropout2(x)
        #x = self.linear1_3(x)
        #x = F.relu(x)
        #if not self.searching:
        #    x = F.normalize(x)
        #res.append(x)
        for i in range(data.shape[1]):
            x = data[:, i]
            x = self.linear2_1(x.view(-1, self.spec_size))
            x = F.relu(x)
            #x = torch.tanh(x)
            x = self.dropout1(x)
            x = self.linear2_2(x)
            x = F.relu(x)
            #x = torch.tanh(x)
            #x = self.dropout2(x)
            #x = self.dropout2(x)
            #x = self.linear2_3(x)
            #x = F.relu(x)
            #if not self.searching:
            x = F.normalize(x)
            #x = self.linear1_3(x)
            #x = F.relu(x)
            res.append(x)
        
        return res
    
    def name(self):
        return "Net"