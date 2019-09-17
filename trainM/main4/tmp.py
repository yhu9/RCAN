import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.weight = torch.nn.Parameter(data=torch.Tensor(5, 5), requires_grad=True)

        self.weight.data.uniform_(-1, 1)

    def forward(self, x):
        masked_wt = self.weight.mul(x)
        return masked_wt

    def maskloss(self, x):
        return sumweight,maxweight

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mask = Mask()

    #def forward(self,x):
    #    sumweight,maxweight = self.mask.maskloss(x)
    #    return sumweight,maxweight
    def forward(self,x):
        return self.mask(x)

model = Model()
indata = torch.rand(5,5)
ones = torch.ones(5,5)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

'''
while True:
    sumweight,maxweight = model(indata)
    print(maxweight)
    #print(x)
    print(model.mask.weight)
    #quit()
    optimizer.zero_grad()
    #loss = F.l1_loss(indata,x)
    print(maxweight)
    maxweight - 1
    loss = torch.mean(torch.abs(maxweight - 1))
    loss.backward()
    optimizer.step()

    print(model.mask.weight)
'''
import time
while True:
    x1 = model(indata)
    x2 = model(ones)

    loss2 = torch.mean(torch.abs(torch.sum(torch.abs(x2),dim=1) - 1))  #sum of each row should = 1
    value,indices = x2.max(1)
    loss3 = torch.mean(torch.abs(x2[:,indices] - 1))          #max value weight should = 1

    optimizer.zero_grad()
    #loss3 = torch.mean(x2.max(0) - 1)
    #print(torch.sum(x2,dim=1) - 1)
    #loss = F.l1_loss(x,indata)
    loss = loss2 + loss3
    loss.backward()
    optimizer.step()

    print(loss2,loss3)
    print(model.mask.weight)

    time.sleep(0.1)

for name, param in model.named_parameters():
    print(name,param)





