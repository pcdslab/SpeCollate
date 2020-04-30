class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.searching = False
        
        self.linear1_1 = nn.Linear(specsize, 512)
        self.linear1_2 = nn.Linear(1024, 512)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        
    def forward(self, data):
        res = []
        x = data[:, 0]
        x = self.linear1_1(x.view(-1, specsize))
        x = F.relu(x)
        x = self.dropout2(x)
        #x = self.linear1_2(x)
        #x = F.relu(x)
        #x = self.dropout2(x)
        #x = self.linear1_3(x)
        #x = F.relu(x)
        #if not self.searching:
        #    x = F.normalize(x)
        res.append(x)
        for i in range(data.shape[1]-1):
            x = data[:, i+1]
            x = self.linear2_1(x.view(-1, specsize))
            x = F.relu(x)
            x = self.dropout2(x)
            #x = self.linear2_2(x)
            #x = F.relu(x)
            #x = self.dropout2(x)
            #x = self.linear2_3(x)
            #x = F.relu(x)
            #if not self.searching:
            #    x = F.normalize(x)
            #x = self.linear1_3(x)
            #x = F.relu(x)
            res.append(x)
        
        return res
        
    def name(self):
        return "Net"