import torch

def onenn_error(train_X, train_labels, test_X, test_labels):
    # Compute pairwise distance matrix
    n = train_X.size(dim=0) # 训练样本数
    ntest = test_X.size(dim=0) # 测试样本数
    
    pred_cls = torch.zeros((ntest,1), device="cuda:0")
    for j in range(ntest):
        dist = torch.sum((test_X[j,:].repeat(n,1)-train_X)**2, dim=1) # 维度: [n]
        idx = torch.argmin(dist)
        pred_cls[j,0] = train_labels[idx]
    err = torch.sum(test_labels != pred_cls)/ntest
    return err


def train_lin_rbm(X, h, eta=1e-3, max_iter=50, weight_cost=2e-4):
    device = "cuda"

    n,v = X.shape # n:样本数 v:特征数
    W = torch.normal(mean=0.0, std=0.1, size=(v,h)).to(device) # W ~ N(0, 0.01)
    bias_upW = torch.zeros((1,h)).to(device)
    bias_downW = torch.zeros((1,v)).to(device)

    dW = torch.zeros((v,h)).to(device)
    dBias_upW = torch.zeros((1,h)).to(device)
    dBias_downW = torch.zeros((1,v)).to(device)

    bs = 256; # batch size
    for iter in range(0,max_iter):
        err = 0
        ind = torch.randperm(n).to(device) # [0,n-1] dtype=torch.int64

        if iter<5: momentum = 0.5
        else: momentum = 0.9
        for batch in range(0,n-bs,bs):
            ub = min(batch+bs-1, n)
            vis1 = X[ind[batch:ub+1],:] # bs*v
            hid1 = torch.mm(vis1,W) + bias_upW.repeat((bs,1)) # bs*h

            hid_states = hid1 + torch.normal(mean=0.0, std=1.0, size=(bs,h)).to(device) # bs*h {True, False}
            vis2 = 1/(1 + torch.exp( -torch.mm(hid_states,W.T) - bias_downW.repeat((bs,1)) ) ) # bs*v
            hid2 = torch.mm(vis2, W) + bias_upW.repeat((bs,1)) # bs*h

            posprods = torch.mm(vis1.T, hid1)/bs # (v*bs)*(bs*h) = v*h
            negprods = torch.mm(vis2.T, hid2)/bs # (v*bs)*(bs*h) = v*h
            dW = momentum*dW + eta*(posprods - negprods - weight_cost*W) # 忘加 -wieght_cost*W
            dBias_upW = momentum*dBias_upW + eta/bs * (sum(hid1,0) - sum(hid2,0)) # 1*h
            dBias_downW = momentum*dBias_downW + eta/bs * (sum(vis1,0) - sum(vis2,0)) # 1*v

            W += dW
            bias_upW += dBias_upW
            bias_downW += dBias_downW
            err += torch.sum( (vis1-vis2)**2 )/n
        if iter%10==0:
            print("Iter {:d}: rec. err = {:e}".format(iter, err))
    machine = {'W':W, 'bias_upW':bias_upW, 'bias_downW':bias_downW}
    return machine
