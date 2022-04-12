import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import numpy as np

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
# from ... import knowgraph_conceptnet 
from knowgraph_conceptnet import KnowledgeGraph

class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)


    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        """
        input : position emb + already gen caption (probably)
        enc_output : tensor of output of all the encoders
        mask_pad :  (b_s, seq_len, 1)
        mask_self_att : (1, 1, seq_len, seq_len)
        mask_enc_att :enc mask for input partial cap, size: (b_s, 1, 1, seq_len)
        """
        
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        # cross attention between each word and the context features
        enc_att3 = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad
        enc_att = enc_att3 * mask_pad


        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        ff = ff.clamp(min=1e-4)
        return ff


class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None, transform_tok =None, device = None, on_lisa=True, edge_select="random", spec = None, seg_token=False,KG = None):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        self.KG = KG
        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())
        self.max_pref = 0
        self.seg_token = seg_token
        
    
    def load_partial_pt_tok(self, vocab_size, d_model, padd):
        clip_pt = 
        num_new_toks = vocab_size - len(clip_pt)
        newtok_emb = Parameter(torch.rand(num_new_toks, d_model))
        # TODO: how to use padding index
        word_emb = torch.cat((clip_pt, newtok_emb), 0)

        return word_emb

    def forward(self, input, encoder_output, mask_encoder, contextfeat):
        """
        input : caption , probably the part that is already generated, so increases each time
        encoder_output : tensor of output of all the encoders
        masked_encoder : enc mask for input partial cap, size: (b_s, 1, 1, seq_len)
        """

        # get the KG matrices
        max_pref = 0
        self.stateful_1 = self.running_seq[0][0] if self.running_seq.size(0) >1 else 0
        if self.stateful_1 == 0:
            with torch.no_grad():
                know_sent_batch, position_batch, visible_matrix_batch, seg_batch = self.KG.get_vm_from_imgid(contextfeat)
            visible_matrix_batch = visible_matrix_batch == 0
            seg_batch = seg_batch == 1

            # store the sizes of the sentence and the KG part
            max_pref = know_sent_batch.size(1)           
            self.max_pref = max_pref

        bs , seq_len = input.size()
        tot_seq = max_pref + seq_len

        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
         
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()   # (b_s, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        # print("attmask4", mask_self_attention.size())

        mask_self_attention_copy = mask_self_attention.clone()

        seq = torch.arange(max_pref, seq_len + max_pref).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:
            # print("I AM SO FULL")
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
            self.running_seq.add_(1)
            seq = self.running_seq + self.max_pref
            if self.stateful_1 == 1:
                # self.running_seq.add_(max_pref)
                # print("segb, masatt:", seg_batch.unsqueeze(1).unsqueeze(1).size(), mask_self_attention_copy.size())
                self.running_mask_self_attention = torch.cat([seg_batch.unsqueeze(1).unsqueeze(1), mask_self_attention_copy], -1)
            # print("test self masks:", self.running_mask_self_attention, mask_self_attention)
        # print("stateful1 is:", self.stateful_1)

        # if statefull and not first number skip this
        if  self.stateful_1 < 2:
            seq_len1 , seq_len2 =  mask_self_attention.size(-2), mask_self_attention.size(-1)
            tot_seq1 = max_pref + seq_len1
            tot_seq2 = max_pref + seq_len2

            # compute the combi mask
            combi_mask = torch.ones((bs, tot_seq1, tot_seq2))
            combi_mask[:,:max_pref,:max_pref] = visible_matrix_batch
            combi_mask[:, -seq_len1:, -seq_len2:] = mask_self_attention.squeeze(1)

            # Need to invert seg_batch to easily find keywords
            seg_batch = seg_batch == 0
            for i, seg_item in enumerate(seg_batch):
                combi_mask[i,max_pref:, :max_pref][:,seg_item] = 0

            combi_mask = combi_mask.unsqueeze(1).gt(0).to(input.device)
            input = torch.cat((know_sent_batch, input) ,-1)
            seq = torch.cat((position_batch, seq),-1)
        else:
            combi_mask = mask_self_attention
        # print("input, seq:", input, seq)
        wordemb = self.word_emb(input)
        posemb = self.pos_emb(seq)
        # print("word emb, pos emb siz:", wordemb.size(), posemb.size())
        out =  wordemb + posemb 
        if self.seg_token == True and self.stateful_1 < 2:
            out[:,:-seq_len,:] += 1
        mask_pad = seq.clone().detach()
        mask_pad = mask_pad != 0
        mask_pad[:,-seq_len:] = mask_queries.squeeze(-1)

        mask_pad = mask_pad.unsqueeze(-1).float().to(input.device)
        for i, l in enumerate(self.layers):                    
            out = l(out, encoder_output, mask_pad, combi_mask, mask_encoder)


        
        if self.stateful_1 <2:
            out = out[:,-seq_len:,:]

        out = self.fc(out)

        return F.log_softmax(out, dim=-1)



class VanillaDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(VanillaDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        # self.KG = KnowledgeGraph(transform_tok = transform_tok, device = device, on_lisa = on_lisa, edge_select=edge_select)
        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

        self.max_pref = 0




    def forward(self, input, encoder_output, mask_encoder, contextfeat):
        """
        input : caption , probably the part that is already generated, so increases each time
        encoder_output : tensor of output of all the encoders
        masked_encoder : enc mask for input partial cap, size: (b_s, 1, 1, seq_len)
        """
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()   # (b_s, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        mask_self_attention = mask_self_attention.to(torch.bool)
        mask_encoder = mask_encoder.to(torch.bool)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)