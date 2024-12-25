import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional
import torch.nn.functional
from func_pool import *
from math import ceil

class DAE_pretrain(nn.Module):
    def __init__(self, in_features, out_features, p_dropout=0.2):
        super(self.__class__, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_dropout = p_dropout
        self.vis_bias = torch.zeros(in_features, device="cuda:0")
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=True),
            nn.Sigmoid()
        )

    def forward(self, X):
        encoded = self.encoder(X)
        recon = 1/(1+torch.exp(-torch.mm(encoded, self.encoder[0].weight.data) - self.vis_bias))
        return encoded, recon
    
    def fit(self, train_X, test_X, lr=0.001, batch_size=256, epoches=100, momentum=0.1):
#        loss_fn = nn.MSELoss()
        loss_fn = nn.BCELoss()
        print('lr={:.3f}  bs={:d}  epoches={:d}  momentum={:.2f}  loss={:s}' .format(lr, batch_size, epoches, momentum, str(loss_fn)))
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum) # 

        # compute the initial test error
        test_num = test_X.size(dim=0)
        test_loss = 0.0
        test_batch_num = ceil(test_num/batch_size)
        for i in range(test_batch_num):
            lb = i*batch_size
            ub = min([(i+1)*batch_size, test_num])
            test_Xb = test_X[lb:ub,:]
            test_Xb = torch.autograd.Variable(test_Xb) # ???
            hidden, recon = self.forward(test_Xb)
            loss = loss_fn(recon, test_Xb)
            test_loss += loss
        print("#Epoch 0: Test Loss: %.6f" % (test_loss) )

        # update the paras via back-propagation
        self.train() # Sets the module in training mode. ???
        train_num, in_feas = train_X.size()
        train_batch_num = ceil(train_num/batch_size)
        # test_loss_p = torch.tensor(float('inf'))
        for epoch in range(epoches):
            place_to_0 = torch.rand((train_num, in_feas)) > self.p_dropout
            place_to_0 = place_to_0.to(device = train_X.device)
            train_Xnoi = train_X*place_to_0
            train_loss = 0.0
            for i in range(train_batch_num):
                lb = i*batch_size
                ub = min([(i+1)*batch_size, train_num])
                train_Xb = train_Xnoi[lb:ub, :]
                train_Xb = torch.autograd.Variable(train_Xb)
                hidden, recon = self.forward(train_Xb)
                loss = loss_fn(recon, train_Xb)
                train_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # compute the final test error
            test_loss = 0.0
            for i in range(test_batch_num):
                lb = i*batch_size
                ub = min([(i+1)*batch_size, test_num])
                test_Xb = test_X[lb:ub,:]
                test_Xb = torch.autograd.Variable(test_Xb) # ???
                hidden, recon = self.forward(test_Xb)
                loss = loss_fn(recon, test_Xb)
                test_loss += loss

            if epoch % 100 == 0:
                print("#Epoch {:4d}: Reconstruction Loss = {:e}  Test Loss = {:e}".format(epoch+1, train_loss, test_loss) )
        print("#Epoch {:4d}: Reconstruction Loss = {:e}  Test Loss = {:e}".format(epoch+1, train_loss, test_loss) )


class DAE(nn.Module):
    def __init__(self, in_features, out_features, layers, p_dropout=0.2):
        super(self.__class__, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_dropout = p_dropout
        self.nlayers = len(layers)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        for i in range(self.nlayers-1):
            self.encoder.add_module('{0}-{1}'.format(layers[i],layers[i+1]), nn.Linear(layers[i], layers[i+1]))
            self.encoder.add_module('Sigm{0}'.format(i), nn.Sigmoid()) # 名字不能一样，否则不执行 跳过
        for i in range(self.nlayers-1, 1, -1): 
            self.decoder.add_module('{0}-{1}'.format(layers[i],layers[i-1]), nn.Linear(layers[i], layers[i-1]))
            self.decoder.add_module('Sigm{0}'.format(i), nn.Sigmoid())
        self.decoder.add_module('{0}-{1}'.format(layers[1],layers[0]), nn.Linear(layers[1], layers[0]))
        self.decoder.add_module('Sigmoid', nn.Sigmoid())
        # self.encoder = nn.Sequential(
        #     # nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        #     nn.Linear(in_features=in_features, out_features=392,bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=392, out_features=196, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=196, out_features=out_features, bias=True),
        #     nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #     # nn.Linear(in_features=out_features, out_features=in_features, bias=True),
        #     nn.Linear(in_features=out_features, out_features=196), 
        #     nn.ReLU(),
        #     nn.Linear(in_features=196, out_features=392), 
        #     nn.ReLU(),
        #     nn.Linear(in_features=392, out_features=in_features),
        #     nn.Sigmoid()
        # )

    def forward(self, X):
        encoded = self.encoder(X)
        recon = self.decoder(encoded)
        return encoded, recon

    def fit(self, train_X, test_X, lr=0.001, batch_size=256, epoches=10, momentum=0.1):
#        loss_fn = nn.MSELoss()
        loss_fn = nn.BCELoss()
        print('lr={:.3f}  bs={:d}  epoches={:d}  momentum={:.2f}  loss={:s}' .format(lr, batch_size, epoches, momentum, str(loss_fn)))
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum) # 

        # compute the initial test error
        test_num = test_X.size(dim=0)
        test_loss = 0.0
        test_batch_num = ceil(test_num/batch_size)
        for i in range(test_batch_num):
            lb = i*batch_size
            ub = min([(i+1)*batch_size, test_num])
            test_Xb = test_X[lb:ub,:]
            test_Xb = torch.autograd.Variable(test_Xb) # ???
            hidden, recon = self.forward(test_Xb)
            loss = loss_fn(recon, test_Xb)
            test_loss += loss
        print("#Epoch 0: Test Loss: %.6f" % (test_loss) )

        # update the paras via back-propagation
        self.train() # Sets the module in training mode. ???
        train_num, in_feas = train_X.size()
        train_batch_num = ceil(train_num/batch_size)
        # test_loss_p = torch.tensor(float('inf'))
        for epoch in range(epoches):
            place_to_0 = torch.rand((train_num, in_feas)) > self.p_dropout
            place_to_0 = place_to_0.to(device = train_X.device)
            train_Xnoi = train_X*place_to_0
            train_loss = 0.0
            for i in range(train_batch_num):
                lb = i*batch_size
                ub = min([(i+1)*batch_size, train_num])
                train_Xb = train_Xnoi[lb:ub, :]
                train_Xb = torch.autograd.Variable(train_Xb)
                hidden, recon = self.forward(train_Xb)
                loss = loss_fn(recon, train_Xb)
                train_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # compute the final test error
            test_loss = 0.0
            for i in range(test_batch_num):
                lb = i*batch_size
                ub = min([(i+1)*batch_size, test_num])
                test_Xb = test_X[lb:ub,:]
                test_Xb = torch.autograd.Variable(test_Xb) # ???
                hidden, recon = self.forward(test_Xb)
                loss = loss_fn(recon, test_Xb)
                test_loss += loss

            if epoch % 100 == 0:
                print("#Epoch {:4d}: Reconstruction Loss = {:e}  Test Loss = {:e}".format(epoch+1, train_loss, test_loss) )
            # if test_loss < test_loss_p:
            #     test_loss_p = test_loss
            # else:break
        print("#Epoch {:4d}: Reconstruction Loss = {:e}  Test Loss = {:e}".format(epoch+1, train_loss, test_loss) )

traindata = scipy.io.loadmat("mnist_train.mat")
train_X = torch.from_numpy(traindata["train_X"]).type(torch.float32)
train_labels = torch.from_numpy(traindata["train_labels"])
testdata = scipy.io.loadmat("mnist_test.mat")
test_X = torch.from_numpy(testdata["test_X"]).type(torch.float32)
test_labels = torch.from_numpy(testdata["test_labels"])
del traindata, testdata

device = "cuda:0"
train_X = train_X.to(device)
test_X = test_X.to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)

# pre-train paras
#pre_epoches = 300
#pre_lr = 0.9
#pre_bs = 256
#pre_momentum = 0.8
#pre_pdrop = 0.2

# train paras
d_lb = 2
d_ub = 3
d_len = d_ub - d_lb
err_arr = torch.zeros(d_len)
lr = 0.9
bs = 256
epoches = 500
momentum = 0.8
p_dropout = 0.2

for target_dim in range(d_lb,d_ub):
    layers = [784, target_dim]
    dae = DAE(in_features=784, out_features=target_dim, layers=layers, p_dropout=p_dropout).to(device)

    # pre-train (DAEs)
#    print("Pre-training...")
#    pre_train_X = train_X.clone()
#    pre_test_X = test_X.clone()
#    L = len(layers) # 网络数
#    for i in range(L-1): # greedy layer-by-layer pre-training
#        dae_pre = DAE_pretrain(in_features=layers[i], out_features=layers[i+1], p_dropout=pre_pdrop).to(device)
#        dae_pre.fit(pre_train_X, pre_test_X, lr=pre_lr, batch_size=pre_bs, epoches=pre_epoches, momentum=pre_momentum)
#        pre_train_X, _ = dae_pre.forward(pre_train_X)
#        pre_test_X, _ = dae_pre.forward(pre_test_X)
#        dae.encoder[i*2].weight.data.copy_(dae_pre.encoder[0].weight.data)
#        dae.encoder[i*2].bias.data.copy_(dae_pre.encoder[0].bias.data)
#        del dae_pre
#    layers.reverse() # 反向
#    for i in range(L-1):
#        dae_pre = DAE_pretrain(in_features=layers[i], out_features=layers[i+1]).to(device)
#        dae_pre.fit(pre_train_X, pre_test_X, lr=pre_lr, batch_size=pre_bs, epoches=pre_epoches, momentum=pre_momentum)
#        pre_train_X, _ = dae_pre.forward(pre_train_X)
#        pre_test_X, _ = dae_pre.forward(pre_test_X)
#        dae.decoder[i*2].weight.data.copy_(dae_pre.encoder[0].weight.data)
#        dae.decoder[i*2].bias.data.copy_(dae_pre.encoder[0].bias.data)
#        del dae_pre
    # pre-train (随机初始化, 两种 pre-train 二选一)
    for i in range(dae.nlayers-1):
        dae.encoder[2*i].weight.data.uniform_(-1/layers[i], 1/layers[i]).to(device)
        dae.encoder[2*i].bias.data.uniform_(-0.0, 0.0).to(device)
        dae.decoder[2*i].weight.data.uniform_(-1/layers[dae.nlayers-1-i], 1/layers[dae.nlayers-1-i]).to(device)
        dae.decoder[2*i].bias.data.uniform_(-0.0, 0.0).to(device)

    # fine tune (back-propagation)
    print("...Training...")
    dae.fit(train_X, test_X, lr=lr, batch_size=bs, epoches=epoches, momentum=momentum)
    mapped_train,_ = dae.forward(train_X)
    mapped_test, _ = dae.forward(test_X)
    err_rate = onenn_error(mapped_train, train_labels, mapped_test, test_labels)
    print("target:{:d}D  1NN err = {:.6f}".format(target_dim, err_rate))
    err_arr[target_dim-d_lb] = err_rate

for i in range(d_len):
    print("{:.6f}".format(err_arr[i]))

