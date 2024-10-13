"""
    New version
    T-GCN
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class TGCNGraphConvulution(nn.Module):
    def __init__(self, num_gru_units, output_dim, bias = 0.0):
        super(TGCNGraphConvulution, self).__init__()
        self.num_gru_units = num_gru_units
        self.input_dim = self.num_gru_units + 1
        self.output_dim = output_dim
        self.bias_init_value = bias
        self.weights = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim))
        nn.init.xavier_normal_(self.weights)
        self.biases = nn.Parameter(
            torch.Tensor(self.output_dim))
        nn.init.constant_(self.biases, self.bias_init_value)
    
    def forward(self, inputs, hidden_state, laplacian):
        """
            只提取时刻t的X_t
        """
        self.laplacian = laplacian
        batch_size, num_nodes, _ = inputs.shape
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self.num_gru_units))  # bat, num, hidden_dim
        # bat, num_nodes, num_gru_units+1
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # num_nodes, num_gru_units+1, bat
        concatenation = concatenation.permute(1, 2, 0).contiguous()
        concatenation = concatenation.reshape((num_nodes, self.input_dim * batch_size))
        a_times_concat = self.laplacian @ concatenation
        a_times_concat = a_times_concat.reshape((num_nodes, self.input_dim, batch_size))
        a_times_concat = a_times_concat.permute(2, 0, 1)
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self.input_dim))
        outputs = a_times_concat @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self.output_dim)).reshape((batch_size, num_nodes * self.output_dim))
        return outputs




class TGCNCell(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, dropout_rate, leaky_rate, device):
        super(TGCNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.alpha  =   leaky_rate
        self.time_step = time_step

        # attention parameters
        self.weight_key = nn.Parameter(torch.zeros(size=(self.input_dim, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.input_dim, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.graph_conv1 = TGCNGraphConvulution(self.hidden_dim, self.hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvulution(self.hidden_dim, self.hidden_dim)

        self.gru = nn.GRU(1, self.input_dim)

        self.to(device)
        
    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        self.laplacian = self.correlation_layer(inputs.permute(0, 2, 1).contiguous())
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state, self.laplacian))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state, self.laplacian))
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state
    
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


class TGCN(nn.Module):
    def __init__(self, unit, stack_cnt, time_step, multi_layer, horizon, dropout_rate=0.5, leaky_rate=0.2, device='cpu'):
        super(TGCN, self).__init__()
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.time_step = time_step
        self.dropout_rate = dropout_rate
        self.leaky_rate = leaky_rate
        self.device = device
        self.hidden_dim = horizon     # prediction horizon
        self.tgcn_cell = TGCNCell(self.unit, self.time_step, self.hidden_dim, self.dropout_rate, self.leaky_rate, self.device)
    
    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self.unit == num_nodes and self.time_step == seq_len
        hidden_state = torch.zeros(batch_size, num_nodes * self.hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self.hidden_dim))
        return output.reshape((batch_size, num_nodes * self.hidden_dim))
