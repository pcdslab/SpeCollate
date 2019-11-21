class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        #self.conv1 = nn.Conv2d(1, 64, 7)
        self.linear1_1 = nn.Linear(specsize, 1024)
        self.linear1_2 = nn.Linear(1024, 512)
        self.linear1_3 = nn.Linear(1024, 512)
        #self.linear1_3 = nn.Linear(256, 128)
        self.linear1_4 = nn.Linear(512, 256)
        
        self.lstm2_1 = nn.Linear(specsize, 1024)
        self.linear2_1 = nn.Linear(1024, 512)
        self.linear2_2 = nn.Linear(512, 256)
        #self.pool1 = nn.MaxPool2d(2)
        #self.conv2 = nn.Conv2d(64, 128, 5)
        #self.conv3 = nn.Conv2d(128, 256, 5)
        
        self.linear1 = nn.Linear(128, 2)
        self.linear2 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(0.1)
        
        #self.linear2 = nn.Linear(512, 2)
        
    def forward(self, data):
        res = []
        for i in range(data.shape[1]):
            x = data[:, i]
            x = self.linear1_1(x.view(-1, specsize))
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.linear1_2(x)
            x = F.relu(x)
            #x = self.linear1_3(x)
            #x = F.relu(x)
            res.append(x)
        
#         x1 = data[:,0]
#         x1 = self.linear1_1(x1.view(-1, specsize))
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, training=self.training)
        
#         x1 = self.linear1_2(x1)
#         x1 = F.relu(x1)
        
#         x1 = self.linear1_3(x1)
#         x1 = F.relu(x1)
        
# #         x1 = self.linear1_3(x1)
# #         x1 = F.relu(x1)
        
# #         x1 = self.linear1_4(x1)
# #         x1 = F.relu(x1)
        
#         x2 = data[:,1]
        
#         x2 = self.linear1_1(x2.view(-1, specsize))
#         x2 = F.relu(x2)
#         x2 = F.dropout(x2, training=self.training)
        
#         x2 = self.linear1_2(x2)
#         x2 = F.relu(x2)
        
#         x2 = self.linear1_3(x2)
#         x2 = F.relu(x2)
        
#         x2 = self.linear1_3(x2)
#         x2 = F.relu(x2)
        
#         x2 = self.linear1_4(x2)
#         x2 = F.relu(x2)
        #res = F.pairwise_distance(x1, x2)
        #res = torch.abs(x1 - x2)
        #print(res.shape)
        #res = self.linear1(res)
        #res = self.linear2(res)
        
        #return F.log_softmax(res, 1)
        return res
        
#        for i in range(2): # Siamese nets; sharing weights
#            x = data[i]
#            x = self.conv1(x)
#            x = F.relu(x)
#            x = self.pool1(x)
#            x = self.conv2(x)
#            x = F.relu(x)
#            x = self.conv3(x)
#            x = F.relu(x)
#            
#            x = x.view(x.shape[0], -1)
#            x = self.linear1(x)
#            res.append(F.relu(x))
#        
#        res = torch.abs(res[1] - res[0])
#        res = self.linear2(res)
#        return res
    def name(self):
        return "Net"