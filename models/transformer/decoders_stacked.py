import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import numpy as np

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
# from ... import knowgraph_conceptnet 
import clip


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
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

class StackedPromptDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None,  spec = None, seg_token=False, KG = None, enc_model="ViT", pt_tokemb = False):
        super(StackedPromptDecoder, self).__init__()
        self.d_model = d_model

        if pt_tokemb:
            print("using pretrained token embeddings")
            self.word_emb = embedding_table(vocab_size, d_model, padding_idx, enc_model, spec["device"])
        else:
            self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

        self.layers_prompt = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
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

        
        kw_tokens = 15
        self.pos_start_sent =  kw_tokens + KG.first_pos_idx
        
    

    def forward(self, input, encoder_output, mask_encoder, contextfeat):
        """
        input : caption , probably the part that is already generated, so increases each time
        encoder_output : tensor of output of all the encoders
        masked_encoder : enc mask for input partial cap, size: (b_s, 1, 1, seq_len)
        """

        # get the KG matrices
        max_pref = 0
        self.stateful_1 = self.running_seq[0][0] if self.running_seq.size(0) >1 else 0
        b_s, seq_len = input.shape[:2]

        if self.stateful_1 == 0:
            with torch.no_grad():
                know_sent_batch, position_batch, visible_matrix_batch, seg_batch = self.KG.get_vm_from_imgid(contextfeat)
            visible_matrix_batch = visible_matrix_batch == 0
            seg_batch = seg_batch == 1

            ksb_pad_mask = position_batch != 0
            ksb_pad_mask = ksb_pad_mask.unsqueeze(-1).float()

            # store the sizes of the sentence and the KG part
            max_pref = know_sent_batch.size(1)           
            self.max_pref = max_pref

            # because keywords prompt can have variable size length due to tokenization, create mask so that kw tensor is of same size.
            seg_batch_neg = ~seg_batch
            kw_lengths = np.array([sum(map(int, segbatch_i)) for segbatch_i in seg_batch_neg])
            max_kw = max(kw_lengths)
            mask_kw = torch.zeros((b_s, max_kw),device=input.device)
            kw_len_dif = kw_lengths - max_kw 

            # for each batch, set the right most to mask 1, if it is padded.
            for j, kw_len in enumerate(kw_len_dif):
                for i in range(kw_len, 0):
                    mask_kw[j][i] = 1
            sent_kw_mask = mask_kw.unsqueeze(1).repeat(1,seq_len+max_kw,1).unsqueeze(1) # from shape (bs,kw) -> (bs, 1, seqlen+kw, kw) , this is for the sentence MSCA


        
        # so only 0, for pad in queries
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
         
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()   # (b_s, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        mask_self_attention_copy = mask_self_attention.clone()

        seq = torch.arange(self.pos_start_sent , seq_len + self.pos_start_sent, device = input.device).view(1, -1).expand(b_s, -1) # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:            
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_att_sent = self.running_mask_self_attention
            seq = self.running_seq + self.pos_start_sent
            self.running_seq.add_(1)

            # if self.stateful_1 == 1:
            #     self.running_mask_self_attention = torch.cat((sent_kw_mask, mask_self_att_sent), -1).gt(0)

        wordemb = self.word_emb(input)
        posemb = self.pos_emb(seq)
        sent_out =  wordemb + posemb 

        # if statefull and not first number skip this
        if  self.stateful_1 < 2:
            mask_queries =torch.cat((torch.ones((b_s, max_kw, 1), device=input.device), mask_queries), 1)
            top_row = torch.ones((b_s, 1, max_kw, seq_len), device=input.device)
            mask_self_att_sent = torch.cat((top_row, mask_self_attention_copy),-2)
            mask_self_att_sent = torch.cat((sent_kw_mask, mask_self_att_sent), -1).gt(0)
            seg_batch = ~ seg_batch    # Invert seg_batch to find keywords

            if self._is_stateful:
                self.running_mask_self_attention = torch.cat((mask_kw.unsqueeze(1).unsqueeze(1), mask_self_attention_copy), -1).gt(0)
        
            # input for the KG Decoder
            wordemb_ksb = self.word_emb(know_sent_batch)        
            posemb_ksb = self.pos_emb(position_batch)
            ksb_out = wordemb_ksb + posemb_ksb
            if self.seg_token == True:
                ksb_out += 1

            # compute the kw tensor
            pad_emb = self.word_emb(torch.tensor([self.padding_idx], device=input.device))
            pad_kw_tensor = pad_emb.clone().detach().repeat(b_s, max_kw, 1)
            visible_matrix_batch = visible_matrix_batch.unsqueeze(1)

        # if self.stateful_1 < 2:
            for i, l in enumerate(self.layers_prompt):                    
                ksb_out = l(ksb_out, encoder_output, ksb_pad_mask, visible_matrix_batch, mask_encoder)

            for i in range(len(seg_batch)):
                kws = ksb_out[i][seg_batch[i]] 
                pad_kw_tensor[i,:kw_lengths[i]] = kws
            keywords_output = pad_kw_tensor
            sent_out = torch.cat((keywords_output, sent_out), 1)
            
        for i, l in enumerate(self.layers):                    
            sent_out = l(sent_out, encoder_output, mask_queries, mask_self_att_sent, mask_encoder)

        if self.stateful_1 < 2:
            sent_out = sent_out[:,max_kw:]
            
        out = self.fc(sent_out)

        return F.log_softmax(out, dim=-1)


    