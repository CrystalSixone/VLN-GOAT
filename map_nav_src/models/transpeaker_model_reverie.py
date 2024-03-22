import os

import math
import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from reverie.parser import parse_args
args = parse_args()

# Transformer Parameters
d_model = args.h_dim 
d_ff = args.proj_hidden 
d_k = d_v = args.aemb  # dimension of K(=Q), V
n_layers_encoder = args.speaker_layer_num  
n_layers_decoder = args.speaker_layer_num
n_heads = args.speaker_head_num   
image_n_heads = args.speaker_head_num 

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) 
    return pad_attn_mask.expand(batch_size, len_q, len_k)  

def get_dec_enc_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.unsqueeze(1) 
    return pad_attn_mask.expand(batch_size, len_q, len_k)  


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(args.speaker_dropout)

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) 
        
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn) 
        context = torch.matmul(attn, V) 
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,Q_hidden_size,K_hidden_size,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = Q_hidden_size
        self.n_heads = n_heads
        self.W_Q = nn.Linear(Q_hidden_size, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(K_hidden_size, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(K_hidden_size, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, Q_hidden_size, bias=False)
        self.dropout = nn.Dropout(args.speaker_dropout)
        self.dot_product = ScaledDotProductAttention()
    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, d_k).transpose(1,2) 
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, d_k).transpose(1,2)  
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, d_v).transpose(1,2) 
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        if args.use_drop:
            Q = self.dropout(Q)
            K = self.dropout(K)
            V = self.dropout(V)
        context, attn = self.dot_product(Q,K,V,attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * d_v)
        output = self.fc(context) 
        output = nn.LayerNorm(self.hidden_size).cuda()(output + residual)
        output = self.dropout(output)
        return output, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_size):
        super(PoswiseFeedForwardNet, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
                nn.Linear(hidden_size, d_ff, bias=False),
                nn.ReLU(), 
                nn.Dropout(args.speaker_dropout),
                nn.Linear(d_ff, hidden_size, bias=False)
            )
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.hidden_size).cuda()(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,hidden_size,n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(hidden_size,hidden_size,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size)

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) 
        enc_outputs = self.pos_ffn(enc_outputs)
        
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self,Q_size,K_size,n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(Q_size,Q_size,n_heads)
        self.dec_enc_attn = MultiHeadAttention(Q_size,K_size,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(Q_size)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask=None):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) 
        return dec_outputs, dec_self_attn, dec_enc_attn

class TranspeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size,dropout_ratio=0.1,feat_drop_ratio=args.featdropout,obj_size=768):
        super(TranspeakerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.pos_emb = PositionalEncoding(hidden_size)

        self.drop_feat = nn.Dropout(p=feat_drop_ratio) 
        self.drop = nn.Dropout(p=args.speaker_dropout) 

        self.down_size = nn.Linear(feature_size,hidden_size)
        self.layers = nn.ModuleList([EncoderLayer(hidden_size,n_heads) for _ in range(n_layers_encoder)])

        self.image_self_attn = MultiHeadAttention(hidden_size,feature_size,image_n_heads)

        # For object
        obj_feat_size = obj_size
        self.obj_feat_linear = nn.Linear(obj_feat_size,hidden_size)
        self.obj_ang_linear = nn.Linear(4,hidden_size)
        self.obj_pos_linear = nn.Linear(3,hidden_size)
        self.ob_obj_linear = nn.Linear(hidden_size*2,hidden_size)
        self.obj_id_linear = nn.Embedding(50, hidden_size)

        if args.obj_type == 'attn':
            self.ob_obj_attn = MultiHeadAttention(hidden_size,hidden_size,image_n_heads)

    def forward(self, action_inputs, feature_inputs, obj_feat,obj_ang,obj_pos, already_dropfeat=False,ctx_mask=None,obj_ids=None,length=None,target_obj_info=None):
        batch_size, max_length, _ = action_inputs.size()
        if ctx_mask is not None:
            ctx_mask = get_attn_pad_mask(ctx_mask,ctx_mask) 
        
        if args.last_pos:
            # For SOON, only use the last viewpoint as inputs
            batch_size, _ = action_inputs.size()
            max_length = 1
        else:
            batch_size, max_length, _ = action_inputs.size()
        
        # object
        if obj_feat is not None:
            obj_feat_emb = self.obj_feat_linear(obj_feat)
            obj_ang_emb = self.obj_ang_linear(obj_ang)
            obj_pos_emb = self.obj_pos_linear(obj_pos)
            if obj_ids is not None:
                obj_ids = obj_ids.int().squeeze()
                obj_ids_emb = self.obj_id_linear(obj_ids) 
                obj_emb = obj_feat_emb + obj_ang_emb + obj_pos_emb + obj_ids_emb
            else:
                obj_emb = obj_feat_emb + obj_ang_emb + obj_pos_emb
            obj_emb = self.drop_feat(obj_emb)
        
        if args.pred_objids != "":
            if target_obj_info is not None:
                target_obj_feat_emb = self.obj_feat_linear(target_obj_info[...,4:4+args.obj_feat_size])
                target_obj_ang_emb = self.obj_ang_linear(target_obj_info[...,:4])
                target_obj_pos_emb = self.obj_pos_linear(target_obj_info[...,4+args.obj_feat_size:4+args.obj_feat_size+3])
                target_obj_ids_emb = self.obj_id_linear(target_obj_info[...,-1].int())
                target_obj_emb = target_obj_feat_emb + target_obj_ang_emb + target_obj_pos_emb + target_obj_ids_emb
                target_obj_emb = self.drop_feat(target_obj_emb)

        if not already_dropfeat: 
            action_inputs[...,:args.obj_feat_size] = self.drop_feat(action_inputs[...,:args.obj_feat_size])
        ctx = self.down_size(action_inputs) 

        if not already_dropfeat:
            feature_inputs[...,:args.obj_feat_size] = self.drop_feat(feature_inputs[...,:args.obj_feat_size])
        
        ctx = ctx.reshape(batch_size*max_length, -1, self.hidden_size)

        feature_inputs = feature_inputs.reshape(batch_size*max_length,36,-1) 
        enc_inputs, attns = self.image_self_attn(
            ctx, 
            feature_inputs,
            feature_inputs 
        ) 
        
        enc_inputs = enc_inputs.view(batch_size, max_length, -1) 
        attns = attns.view(batch_size,max_length,-1) 
        
        if obj_feat is not None:
            if args.obj_type == 'mean':
                obj_emb = obj_emb.mean(dim=2) 
                enc_inputs = torch.cat((enc_inputs,obj_emb),dim=-1) 
                enc_inputs = self.ob_obj_linear(enc_inputs)
            elif args.obj_type == 'attn':
                enc_inputs = enc_inputs.reshape(batch_size*max_length,1,-1)
                obj_inputs = obj_emb.reshape(batch_size*max_length,-1,self.hidden_size) 
                enc_inputs, ob_obj_attn = self.ob_obj_attn(enc_inputs,obj_inputs,obj_inputs) 
                enc_inputs = enc_inputs.reshape(batch_size,max_length,self.hidden_size)
        if target_obj_info is not None:
            target_obj_emb = target_obj_emb.unsqueeze(1)
            enc_inputs  = torch.cat((enc_inputs,target_obj_emb),dim=1) 
        enc_outputs = self.pos_emb(enc_inputs.transpose(0,1)).transpose(0,1) 
        enc_self_attns = []
        if not args.last_pos:
            for layer in self.layers:
                enc_outputs, enc_self_attn = layer(enc_outputs,ctx_mask) 
                enc_self_attns.append(enc_self_attn)
    
        return enc_inputs, enc_outputs, enc_self_attn 

        
class TranspeakerDecoder(nn.Module):
    def __init__(self,tgt_vocab_size, word_size, hidden_size,padding_idx):
        super(TranspeakerDecoder, self).__init__()

        self.tgt_vocab_size = tgt_vocab_size
        self.word_size = word_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(tgt_vocab_size, word_size,padding_idx)
        
        self.pos_emb = PositionalEncoding(word_size)
        self.layers = nn.ModuleList([DecoderLayer(word_size,hidden_size,n_heads) for _ in range(n_layers_decoder)])
        self.drop = nn.Dropout(p=args.speaker_dropout)
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs,ctx_mask=None):
        dec_outputs = self.embedding(dec_inputs.to(torch.int64))
        dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1) 
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs,dec_inputs).cuda()
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),0).cuda()
        if ctx_mask is not None:
            dec_enc_attn_mask = get_dec_enc_mask(dec_inputs,ctx_mask) 
        else:
            dec_enc_attn_mask = None

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask) 

            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transpeaker(nn.Module):
    def __init__(self, feature_size, hidden_size, word_size, tgt_vocab_size, padding_idx=0, obj_size=768):
        super(Transpeaker, self).__init__()
        self.encoder = TranspeakerEncoder(feature_size, hidden_size, obj_size=obj_size).cuda()
        self.decoder = TranspeakerDecoder(tgt_vocab_size, word_size, hidden_size, padding_idx=padding_idx).cuda()
        self.projection = nn.Linear(word_size, tgt_vocab_size, bias=False).cuda()
        self.dropout = nn.Dropout(args.speaker_dropout)

    def forward(self, action_embeddings,world_state_embeddings, dec_inputs,obj_feat,obj_ang,obj_pos,ctx_mask=None,already_dropfeat=False,obj_ids=None,length=None,target_obj_info=None):
        if ctx_mask is not None:
            ctx_mask = ctx_mask.cuda()
        enc_inputs, enc_outputs, enc_self_attns = self.encoder(action_embeddings,world_state_embeddings,obj_feat,obj_ang,obj_pos,ctx_mask=ctx_mask,obj_ids=obj_ids,length=length,target_obj_info=target_obj_info)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs,ctx_mask=ctx_mask) 
        if args.use_drop:
            dec_outputs = self.dropout(dec_outputs) 
        dec_logits = self.projection(dec_outputs)
        
        return dec_logits,enc_self_attns,dec_self_attns,dec_enc_attns 