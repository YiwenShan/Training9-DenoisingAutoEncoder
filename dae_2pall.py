import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional
import torch.nn.functional
from func_pool import *
from math import ceil
import time

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
            self.encoder.add_module('ReLU{0}'.format(i), nn.ReLU()) # 名字不能一样，否则不执行 跳过
        for i in range(self.nlayers-1, 1, -1): 
            self.decoder.add_module('{0}-{1}'.format(layers[i],layers[i-1]), nn.Linear(layers[i], layers[i-1]))
            self.decoder.add_module('ReLU{0}'.format(i), nn.ReLU())
        self.decoder.add_module('{0}-{1}'.format(layers[1],layers[0]), nn.Linear(layers[1], layers[0]))
        self.decoder.add_module('Sigmoid', nn.Sigmoid())


    def forward(self, X):
        encoded = self.encoder(X)
        recon = self.decoder(encoded)
        return encoded, recon

    def fit(self, train_X, test_X, lr=0.001, batch_size=256, epoches=10, momentum=0.1):
        loss_fn = nn.MSELoss()
        print('target_dim={:d} noi_rate={:.1f}  lr={:.3f}  bs={:d}  epoches={:d}  momentum={:.2f}  loss={:s}' .format(self.out_features, self.p_dropout, lr, batch_size, epoches, momentum, str(loss_fn)))
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
                loss = loss_fn(recon, train_X[lb:ub, :]) # 不是train_Xb啊!!!
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

import  os
print(os.getcwd()) #获取当前工作目录路径
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

d_lb = 13
d_ub = 14
d_len = d_ub - d_lb
err_arr = torch.zeros(d_len)
# p_dropout = 0.2
p_drop_arr = [0.0, 0.1, 0.2, 0.3, 0.4]
epoches = 1000
for p_idx in range(len(p_drop_arr)):
    p_dropout = p_drop_arr[p_idx]
    for target_dim in range(d_lb,d_ub):
        time_start = time.perf_counter()
        layers = [784, 392, 196, target_dim]
        dae = DAE(in_features=784, out_features=target_dim, layers=layers, p_dropout=p_dropout).to(device)
        # pre-train (linear RBM)
        # machine = train_lin_rbm(train_X, h=target_dim)
        # dae.encoder[0].weight.data.copy_(machine['W'].T)
        # dae.encoder[0].bias.data.copy_(machine['bias_upW'].squeeze())
        # pre-train (随机初始化, 两种 pre-train 二选一)
        for i in range(len(layers)-1):
            dae.encoder[2*i].weight.data.uniform_(-1/layers[i], 1/layers[i]).to(device)
            dae.encoder[2*i].bias.data.uniform_(-0.0, 0.0).to(device)
            dae.decoder[2*i].weight.data.uniform_(-1/layers[dae.nlayers-1-i], 1/layers[dae.nlayers-1-i]).to(device)
            dae.decoder[2*i].bias.data.uniform_(-0.0, 0.0).to(device)

        # fine tune (back-propagation)
        dae.fit(train_X, test_X, lr=0.5, batch_size=256, epoches=epoches, momentum=0.8)
        mapped_train,_ = dae.forward(train_X)
        mapped_test, _ = dae.forward(test_X)
        # mapped_train = mapped_train.detach().cpu() # 放gpu会超显存，只能cpu
        # mapped_test = mapped_test.detach().cpu()
        err_rate = onenn_error(mapped_train, train_labels, mapped_test, test_labels)
        time_end = time.perf_counter()
        print("target:{:d}D  1NN err = {:.6f}  time:{:.8f}s".format(target_dim, err_rate, time_end-time_start))
        err_arr[target_dim-d_lb] = err_rate
        torch.save(dae.state_dict(), "model/dae_drop"+str(p_dropout)+"_d"+str(target_dim)+"_eph"+str(epoches)+"_lay"+str(len(dae.encoder)/2)+".pth")

    for i in range(d_len):
        print("{:.6f}".format(err_arr[i]))
# 2D visualization
# mapped_test = mapped_test.detach().cpu().numpy() # detach()去梯度
# plt.figure()
# for c in range(1,11):
#     this_class_idx = (test_labels==c).squeeze(-1,) # torch..Size([nc,1]) -> Size([nc])
#     plt.scatter(mapped_test[this_class_idx,0], mapped_test[this_class_idx,1], s=1)
# plt.axis('equal')
# plt.show()
# plt.pause()

