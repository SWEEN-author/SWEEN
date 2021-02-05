import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Ensemble(nn.Module):
    def __init__(self, model_list, device, w=None, exp=True):
        super(Ensemble, self).__init__()
        self.exp = exp
        self.n = len(model_list)
        self.device = device
        if w is not None:
            self.w = nn.Parameter(w.to(self.device))
        elif self.exp:
            self.w = nn.Parameter(torch.zeros(self.n).to(self.device))
        else:
            self.w = nn.Parameter((torch.ones(self.n)/self.n).to(self.device))
        for i in range(self.n):
            exec('self.model_'+str(i)+'=model_list[i].to(self.device)')
    
    def set_training_paras(self, weight =True, train_id = []):
        if weight:
            self.w.requires_grad = True
        else:
            self.w.requires_grad = False
        for i in range(self.n):
            if i in train_id:
                exec('for p in self.model_'+str(i)+'.parameters(): p.requires_grad = True')
            else:
                exec('for p in self.model_'+str(i)+'.parameters(): p.requires_grad = False')
    
    def add_submodel(self, model):
        '''add a new submodel and re-initialize weights. Used in gradient boosting.'''
        exec('self.model_'+str(self.n)+'=model.to(self.device)')
        self.n = self.n + 1
        self.w = nn.Parameter(torch.zeros(self.n).to(self.device))
    
    def init_weights(self):
        if self.exp:
            self.w = nn.Parameter(torch.zeros(self.n).to(self.device))
        else:
            self.w = nn.Parameter((torch.ones(self.n)/self.n).to(self.device))
    
    def rand_init(self):
        self.w = nn.Parameter(torch.rand(self.n).to(self.device))

    def set_weights(self, w):
        self.w = nn.Parameter(w.to(self.device))
        
    def forward(self, x):
        out = 0
        if self.exp:
            for i in range(self.n):
                loc = locals()
                exec('out += torch.exp(self.w[i])*self.model_'+str(i)+'(x)')
                out = loc['out']
            out = out/self.w.exp().sum()
        else:
            for i in range(self.n):
                loc = locals()
                exec('out += self.w[i]*self.model_'+str(i)+'(x)')
                out = loc['out']
            #out = out/self.w.sum()
        return out


class AEnsemble(Ensemble):
    '''Adaptive Prediction Ensemble Model.
       Only used at inference time.'''
    def __init__(self, model_list, device, w=None, exp=True, threshold=0.999, cl=0.95,
                 num_classes=10,rt_num=True):
        super(AEnsemble, self).__init__(model_list, device, w, exp)
        if not self.exp:
            raise NotImplementedError("Not Implemented")
        self.threshold = threshold
        self.cl = cl
        self.m = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        self.z = self.m.icdf(torch.tensor((1+self.cl)/2).to(device))
        self.rt_num = rt_num
        self.num_classes = num_classes
                
    def forward(self, x):
        with torch.no_grad():
            alpha = self.w.exp()/self.w.exp().sum()
            w_order = alpha.argsort(descending=True)
            num_pt = x.size()[0]
            out = torch.zeros(num_pt, self.num_classes)
            pred = 0
            ele_left = torch.tensor([i for i in range(x.size()[0])])
    
            if self.rt_num:
                total_cal = 0
                
            for i in range(self.n):
                c = len(ele_left)
                pse_idx = torch.tensor([i for i in range(c)])
                idx = w_order[i].item()
                loc = locals()
                exec('prob = self.model_'+str(idx)+'(x)')
                total_cal += len(ele_left)
                prob = loc['prob']
                pred = prob.unsqueeze(0) if i==0 else torch.cat((pred, prob.unsqueeze(0)),0)
                cw = alpha[w_order[:i+1]]
                cur_pred = (cw.view(-1,1,1)*pred).sum(dim=0)/cw.sum()
                p_L, L = torch.max(cur_pred,dim=1)
                
                if i == (self.n-1):
                    for t in pse_idx:
                        out[ele_left[t],:] = cur_pred[t,:]
                    if self.rt_num:
                        return out, total_cal
                    return out
                        
                if i == 0:
                    indices = p_L > self.threshold
                    for t in pse_idx[indices]:
                        out[ele_left[t],:] = cur_pred[t,:]
                        
                    ele_left = ele_left[~indices]
                    if len(ele_left) ==0:
                        if self.rt_num:
                            return out, total_cal
                        return out
                    pred = pred[:,~indices,:]
                    x = x[~indices,:,:,:]
                    continue
                
                pi = 0
                for j in range(c):
                    temp = pred[:,j,L[j]].unsqueeze(1)
                    pi = temp if j==0 else torch.cat((pi, temp),1)
                    
       
                l = self.z*(torch.sum(cw**2).sqrt()/cw.sum())*\
                torch.sqrt(torch.mul(cw.unsqueeze(1),(pi-p_L)**2).sum(dim=0)/cw.sum())
                    
                
                indices = p_L>0.5+l
                for t in pse_idx[indices]:
                    out[ele_left[t],:] = cur_pred[t,:]
                
                ele_left = ele_left[~indices]
                if len(ele_left) ==0:
                    if self.rt_num:
                        return out, total_cal
                    return out
                pred = pred[:,~indices,:]
                x = x[~indices,:,:,:]