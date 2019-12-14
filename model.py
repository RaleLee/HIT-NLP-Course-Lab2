import torch
import torch.nn as nn

class MLPLayer(nn.Module):
    '''
    Use MLP to update performance
    '''
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()

        self.__linear_layer = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=True
        )
        self.__dropout_layer = nn.Dropout(dropout_rate)
    
    def forward(self, input_emb):
        drop = self.__dropout_layer(input_emb)

        linear_out = self.__linear_layer(drop)
        return linear_out

class LinearDecoder(nn.Module):
    '''
    Use linear layer to decode result

    '''

    def __init__(self, hidden_dim, dropout_rate, n_class):
        super().__init__()

        self.__linear_layer = nn.Linear(
            hidden_dim, n_class, bias=True
        )
        self.__dropout_layer = nn.Dropout(dropout_rate)
    
    def forward(self, input_emb):

        drop = self.__dropout_layer(input_emb)

        linear_out = self.__linear_layer(drop)
        return linear_out



class TaggingLayer(nn.Module):
    '''
    Do Tagging to find the result

    use BERT as an encoder and MLP for update performance
    '''

    def __init__(self, bert_base, hidden_dim, dropout_rate, n_class, finetuning=False):
        super().__init__()
        # bert base
        self.__bert_base = bert_base
        # mlp to update performance
        self.__mlplayer = MLPLayer(hidden_dim, dropout_rate)
        # decoder to get the result
        self.__decoder = LinearDecoder(hidden_dim, dropout_rate, n_class)
        # fintuning for bert
        self.__finetuning = finetuning
    
    def forward(self, input_emb, mask, n_layers=1):
        if self.training and self.__finetuning:
            self.__bert_base.train()

            bert_out, _  = self.__bert_base(input_emb, attention_mask=mask)
            emb = bert_out[-1]
        else:
            self.__bert_base.eval()
            with torch.no_grad():
                bert_out, _ = self.__bert_base(input_emb, attention_mask=mask)
                emb = bert_out[-1]
        
        # layer numbers, to update performace
        if n_layers > 1:
            for i in range(n_layers):
                mlp_out = self.__mlplayer(emb)

            decoder_out = self.__decoder(emb)
        else:
            decoder_out = self.__decoder(emb)
        return decoder_out

class JointAsOpCaPoNet(nn.Module):
    '''
    Use tagging method to get the As_Op result and Ca_Po result
    First Layer to get the As_Op tag
    Second Layer to get the Ca_Po tag
    '''

    def __init__(self, bert_base, hidden_dim, dropout_rate, tag_class, n_class, finetuning=False):
        super().__init__()
        # Joint As_Op Net
        self.__joint_As_Op = TaggingLayer(
            bert_base, 
            hidden_dim, 
            dropout_rate, 
            tag_class, 
            finetuning=True
        )

        # Joing Ca_Po Net
        self.__joint_Ca_Po = TaggingLayer(
            bert_base, 
            hidden_dim, 
            dropout_rate, 
            n_class, 
            finetuning=True
        )

    def forward(self, input_emb, mask):
        tag_out_ao = self.__joint_As_Op(input_emb, mask)
        tag_out_cp = self.__joint_Ca_Po(input_emb, mask)

        return tag_out_ao, tag_out_cp