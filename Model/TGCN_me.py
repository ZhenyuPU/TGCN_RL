import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

"""
    GCN: 静态图
"""

class GCN(torch.nn.Module):
    def __init__(self, state_dim, leak_rate=0.2, dropout_rate=0.5):
        self.unit = state_dim - 1    # 除去时间t
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.weight_1 = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))
        nn.init.xavier_normal_(self.weight_1)
        self.weight_2 = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))
        nn.init.xavier_normal_(self.weight_2)
        self.biases_1 = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.biases_2 = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.leakyrelu = nn.LeakyReLU(leak_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        laplacian = self.correlation_layer(x)                   # 11 * 11
        x = laplacian @ x                                       # 11 * 1
        x = F.relu(x @ self.weight_1 + self.biases_1)           # 11 * 1
        x = laplacian @ x                                       # 11 * 1
        x = torch.sigmoid(x @ self.weight_2 + self.biases_2)    # 11 * 1
        return x

    def correlation_layer(self, x):
        """
            attention + 近似归一化的laplacian L
            L * X * W
        """
        attention = self.self_attention(x)        # bat * 11 * 11
        attention = torch.mean(attention, dim=0)  # 11 * 11
        degree = torch.sum(attention, dim=1)      
        # Laplacian
        attention = 0.5 * (attention + attention.T)
        degree_diag = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat, torch.matmul(degree_diag - attention, diagonal_degree_hat))
        return laplacian

    def self_attention(self, x):
        # x: bat * 11 * 1
        bat, N, feat = x.size()   # bat, N = 11, feat: 1
        key = torch.matmul(x, self.weight_key)     # bat * 11 * 1    self.weight_key(11 * 1)
        query = torch.matmul(x, self.weight_query) # bat * 11 * 1    self.weight_query(11 * 1)
        data  = np.matmul(query, key.T)            # bat * 11 * 11
        data  = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention
    


"""
    TGCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



class TGCNGraphConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(TGCNGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim))
        nn.init.xavier_normal_(self.weights)
        self.bias = bias
        # if bias:
        #     self.biases = nn.Parameter(torch.FloatTensor(self.output_dim))
        #     nn.init.xavier_normal_(self.biases)
        # else:
        #     self.register_parameter('biases', None)


    def forward(self, laplacian, x):
        # bat_size, num_nodes, feat = x.shape
        # laplacian: N, N
        bat_size, _, _ = x.shape
        laplacian = laplacian.unsqueeze(0).expand(bat_size, -1, -1)
        x = torch.matmul(laplacian, x) 
        output = x @ self.weights
        # if self.bias:
        #     output += self.biases
        return output

   
    
class GRUPredictor(torch.nn.Module):
    def __init__(self, time_step, hidden_size, num_layers, output_size, device):
        super(GRUPredictor, self).__init__()
        self.input_size = time_step
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.gru = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)
        self.device = device
 
    def forward(self, input):
        # input(batch_size, node num, time_step)
        output, _ = self.gru(input)   # bat, node, hidden dim
        # last_out = output.permute(0, 2, 1).contiguous()  # 最后一个时间步输出bat, node num, hidden size
        pred = self.fc(output)      # bat, node num, prediction horizon
        return pred

class TGCN(torch.nn.Module):
    def __init__(self, units, stack_cnt, time_step, horizon=1, dropout_rate=0.5, leaky_rate=0.2, device='cpu'):
        super(TGCN, self).__init__()
        self.unit   =   units                   # node num
        self.stack_cnt  =   stack_cnt
        self.alpha  =   leaky_rate
        self.time_step  =   time_step
        self.horizon    =   horizon             # prediction horizon

        # attention parameters
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.graph_conv = TGCNGraphConv(input_dim=self.time_step, output_dim=self.horizon, bias=1.0)
        self.GRUPredictor = GRUPredictor(self.time_step, hidden_size=64, num_layers=1, output_size=self.horizon, device='cpu')
        self.gru = nn.GRU(self.time_step, self.unit)
        self.to(device)

    def forward(self, x):
        # x: bat, N, time step
        # x = x.permute(0, 2, 1).contiguous()
        bat, _, _ = x.size()
        laplacian   =   self.correlation_layer(x.permute(0, 2, 1).contiguous())
        for i in range(self.stack_cnt):
            x = self.graph_conv(laplacian, x)
        # bat, N, time step 
        forecast = self.GRUPredictor(x).permute(0, 2, 1).contiguous()    # origin: bat, prediction, node num
        # output = forecast.view(bat, -1)
        return forecast  # bat, predictiom * node num

    def correlation_layer(self, x):
        """
            attention + 近似归一化的laplacian L
            input: bat * time step * node num
        """
        input, _ = self.gru(x.permute(2, 0, 1).contiguous())    # 将x的时间信息提取出来，并构造出每个结点对每个结点的特征？input: node num, bat size, gru feat=node num
        input = input.permute(1, 0, 2).contiguous()         # bat, node, gru feat
        attention = self.self_attention(input)        # bat * 11 * 11
        attention = torch.mean(attention, dim=0)  # 11 * 11
        degree = torch.sum(attention, dim=1)      
        # Laplacian
        attention = 0.5 * (attention + attention.T)
        degree_diag = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat, torch.matmul(degree_diag - attention, diagonal_degree_hat))    # laplacian: bat, 
        return laplacian

    def self_attention(self, x):
        # x: bat, node, gru feat
        x = x.permute(0, 2, 1).contiguous()     # bat, gru feat, node
        bat, feat, N = x.size()   # bat, N = 11, feat: 1
        key = torch.matmul(x, self.weight_key)     # bat * feat * 1   
        query = torch.matmul(x, self.weight_query) # bat * feat * 1   
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data  = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention
