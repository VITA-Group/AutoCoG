
# code copy from https://github.com/ap229997/Conditional-Batch-Norm/blob/6e237ed5794246e1bbbe95bbda9acf81d0cdeace/model/cbn.py#L9 

import torch
import torch.nn as nn
import random

'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''
class CBN(nn.Module):

    def __init__(self , lstm_size ,
                 emb_size ,
                 c_out ,
                 batch_size ,
                 c_in ,
                 height=1 ,
                 width=1 ,
                 use_betas=True ,
                 use_gammas=True ,
                 eps=1.0e-5):
        super(CBN, self).__init__()

        self.lstm_size = lstm_size # size of the lstm emb which is input to MLP
        self.emb_size = emb_size # size of hidden layer of MLP
        self.c_out = c_out # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = batch_size
        self.channels = c_in
        self.height = height
        self.width = width

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size , self.c_out),
            ).cuda()

        self.fc_beta = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size , self.c_out),
            ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel
    Arguments:
        lstm_emb : lstm embedding of the question
    Returns:
        delta_betas, delta_gammas : for each layer
    '''
    def create_cbn_input(self, lstm_emb):

        if self.use_betas:
            delta_betas = self.fc_beta(lstm_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()

        if self.use_gammas:
            delta_gammas = self.fc_gamma(lstm_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values
    Arguments:
        feature : feature map from the previous layer
        lstm_emb : lstm embedding of the question
    Returns:
        out : beta and gamma normalized feature map
        lstm_emb : lstm embedding of the question (unchanged)
    Note : lstm_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require lstm question embeddings
    '''
    def forward(self, feature, lstm_emb):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape


        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, lstm_emb

class GnnCBN(nn.Module):
    def __init__(self , cfg , embeddings , c_in , c_out):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.DATA.NODES
        self.classes = cfg.DATA.CLASSES
        self.hidden_dim =cfg.MODEL.BATCHNORM.HIDDEN_DIM
        self.class_embeddings = embeddings
        emb_size = embeddings.size(1)

        self.cbn = CBN(emb_size,
                       self.hidden_dim,
                       c_out,
                       cfg.DATA.NODES,
                       c_in)
        self.linear = nn.Linear(c_out, self.classes)

    def forward(self, x:torch.Tensor, random_factor=True):
        pseudo_classification = self.linear(x)
        if random_factor and self.training:
            pseudo_labels = pseudo_classification.argmax(dim=1)
            random_select = random.sample([i for i in range(pseudo_labels.size(0))], int(0.2 * pseudo_labels.size(0)))
            pseudo_labels[random_select] = torch.tensor([random.randint(0, self.classes-1) for _ in random_select]).cuda()
        else:
            pseudo_labels = pseudo_classification.argmax(dim=1)
        input_emb = self.class_embeddings[pseudo_labels]
        x = x.unsqueeze(-1).unsqueeze(-1)

        output, emb = self.cbn(x, input_emb)

        if self.training:
            return output.squeeze(), pseudo_classification
        else:
            return output.squeeze()











