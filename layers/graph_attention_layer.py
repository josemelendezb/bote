import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.ReLU()


    def forward(self, input, adj,flage=True):
        h = torch.matmul(input, self.W)
        batch_size = h.size()[0]
        token_lenth = h.size()[1]

        a_input = torch.cat([h.repeat_interleave(repeats=token_lenth,dim=2).view(batch_size,token_lenth * token_lenth, -1), h.repeat_interleave(token_lenth, dim=0).view(batch_size,token_lenth * token_lenth, -1)], dim=2).view(batch_size,token_lenth,-1, 2 * self.out_features)  # 这里让每两个节点的向量都连接在一起遍历一次得到 bacth* N * N * (2 * out_features)大小的矩阵

        e = self.leakyrelu(torch.matmul(a_input, self .a).squeeze(3))

        e = self.relu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)


        attention = torch.softmax(attention, dim=2)  # Here is a non-linear transformation, the weight becomes closer to 1, and the weight without weight becomes 0

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'