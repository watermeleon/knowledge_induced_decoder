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

class PromptDecoderLayer(Module):
    def __init__(self, *args, **kwargs):
        super(PromptDecoderLayer, self).__init__()

        self.sent_msca = MSCA(*args, **kwargs)
        self.ksb_msca = MSCA(*args, **kwargs)

    def forward(self, know_sent_batch, input_sent, enc_output, mask_pad, ksb_pad_mask,  mask_self_att_sent, mask_self_att_ksb , mask_enc_att, seg_batch, pad_kw_tensor, kw_lengths):
        """
        input : position emb + already gen caption (probably)
        enc_output : tensor of output of all the encoders
        mask_pad :  (b_s, seq_len, 1) -> masks all the padding tokens
        mask_self_att : (1, 1, seq_len, seq_len)
        mask_enc_att :enc mask for input partial cap, size: (b_s, 1, 1, seq_len)

        mask_self_att_sent = the original triangle mask + plut 1's on the place of kw's 
        mask_self_att_prompt = the visibility matrix
        """
        # If is statefull, don't have: seg_batch, mask_self_att_ksb
        if self._is_stateful:
            kw_sent = input_sent
        else:
            # keywords= know_sent_batch[seg_batch]   # check if seqbatch slicing works like this
            max_kw = max([sum(map(int, segbatch_i)) for segbatch_i in seg_batch])
            # print("max kw len is:", max_kw)


            newlist = []
            for i in range(len(seg_batch)):
                kws = know_sent_batch[i][seg_batch[i]] 
                newlist.append(kws)
                # intlist = map(int, seg_batch[i])
                # print("total true:", sum(intlist))
                pad_kw_tensor[i,:kw_lengths[i]] = kws

            keywords = pad_kw_tensor
            # keywords = torch.cat([know_sent_batch[i][seg_batch[i]] for i in range(len(seg_batch))], -1)


            kw_sent = torch.cat((keywords.clone(), input_sent) ,-2)
        sent_outp = self.sent_msca(input_sent, kw_sent, enc_output, mask_pad, mask_self_att_sent, mask_enc_att)
        # if is statefull: get these from memory
        ksb_outp = self.ksb_msca(know_sent_batch, know_sent_batch, enc_output, ksb_pad_mask, mask_self_att_ksb, mask_enc_att)

        return ksb_outp, sent_outp

class MSCA(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MSCA, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)


    def forward(self, input, self_att1,  enc_output,  mask_pad, mask_self_att, mask_enc_att):
        """
        input : position emb + already gen caption (probably)
        enc_output : tensor of output of all the encoders
        mask_pad :  (b_s, seq_len, 1)
        mask_self_att : (1, 1, seq_len, seq_len)
        mask_enc_att :enc mask for input partial cap, size: (b_s, 1, 1, seq_len)
        """
        
        self_att = self.self_att(input, self_att1, self_att1, mask_self_att)
        self_att = self_att * mask_pad

        # cross attention between each word and the context features
        enc_att3 = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad
        enc_att = enc_att3 * mask_pad


        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        ff = ff.clamp(min=1e-4)
        return ff

class PromptDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None,  spec = None, seg_token=False, KG = None, enc_model="ViT", pt_tokemb = False):
        super(PromptDecoder, self).__init__()
        self.d_model = d_model
        # self.pad_tokenid = spec["pad_tokenid"]
        self.num_kw= 6

        if pt_tokemb:
            print("using pretrained token embeddings")
            self.word_emb = embedding_table(vocab_size, d_model, padding_idx, enc_model, spec["device"])
        else:
            self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [PromptDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
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

            ksb_pad_mask = position_batch == 0
            ksb_pad_mask = ksb_pad_mask.unsqueeze(-1).float()
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
        mask_self_attention_copy = mask_self_attention.clone()

        seq = torch.arange(self.num_kw +1 , seq_len + self.num_kw+1, device = input.device).view(1, -1).expand(b_s, -1) # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
            self.running_seq.add_(1)
            seq = self.running_seq + self.max_pref
            if self.stateful_1 == 1:
                self.running_mask_self_attention = torch.cat([seg_batch.unsqueeze(1).unsqueeze(1), mask_self_attention_copy], -1)

        # if statefull and not first number skip this
        if  self.stateful_1 < 2:
            # Need to invert seg_batch to easily find keywords
            seg_batch = seg_batch == 0

        wordemb = self.word_emb(input)
        posemb = self.pos_emb(seq)
        sent_out =  wordemb + posemb 

        ksb_out = self.word_emb(know_sent_batch) + self.pos_emb(position_batch)

        if self.seg_token == True and self.stateful_1 < 2:
            ksb_out += 1


        # because keywords prompt can have variable size length due to tokenization, create mask so that kw tensor is of same size.
        kw_lengths = np.array([sum(map(int, segbatch_i)) for segbatch_i in seg_batch])
        max_kw = max(kw_lengths)
        mask_kw = torch.zeros((bs, max_kw),device=input.device)
        kw_len_dif = kw_lengths - max_kw 
        for j, kw_len in enumerate(kw_len_dif):
            for i in range(kw_len, 0):
                mask_kw[j][i] = 1

        sent_kw_mask = mask_kw.unsqueeze(1).repeat(1,seq_len,1).unsqueeze(1) # from shape (bs,kw) -> (bs, 1, seqlen, kw)
        mask_self_att_sent = torch.cat((sent_kw_mask, mask_self_attention), -1).gt(0)

        # compute the kw tensor
        pad_emb = self.word_emb(torch.tensor([self.padding_idx], device=input.device))
        pad_kw_tensor = pad_emb.clone().detach().repeat(bs, max_kw, 1)

        visible_matrix_batch = visible_matrix_batch.unsqueeze(1)
        
        for i, l in enumerate(self.layers):                    
            ksb_out, sent_out = l(ksb_out, sent_out, encoder_output, mask_queries, ksb_pad_mask, mask_self_att_sent, visible_matrix_batch, mask_encoder, seg_batch, pad_kw_tensor, kw_lengths)

        out = self.fc(sent_out)

        return F.log_softmax(out, dim=-1)


    