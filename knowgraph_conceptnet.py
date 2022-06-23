# coding: utf-8
"""
KnowledgeGraph
"""
from imp import release_lock
import os
from posixpath import split
import os.path
from os import path
import resource
from PIL import Image

from zmq import device
import config as config
# import pkuseg
import numpy as np
from collections import defaultdict
import csv
import h5py 
import pickle

import torch 
from transformers import  CLIPTextModel, CLIPTokenizer,CLIPProcessor, CLIPModel, AutoTokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from data.utils import *
from treelib import Node, Tree

import faiss
class empty_fais_knn(object):
    def get_nn(self,q_emb):
        return []

#  FAISS cosine distance
class get_fais_knn(object):
    def __init__(self, words, embeddings, k = 4):
        self.dimension = 512    # dimensions of each vector                         
        self.k = k       # return 3 nearest neighbours        
        self.embeddings = embeddings.astype('float32')
        self.words = words
        self.index = faiss.index_factory(512, "Flat", faiss.METRIC_INNER_PRODUCT)

        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)   # add the vectors and update the index
    
    def get_nn(self, q_emb):
        faiss.normalize_L2(q_emb)
        _, indices = self.index.search(q_emb, self.k)
        nn_words = np.squeeze(self.words[indices]).tolist()
        if self.k == 1:
            nn_words = [nn_words]
        return nn_words

#   FAISS but L2 distance
# class get_fais_knn(object):
#     def __init__(self, words, embeddings, k = 4):
#         self.dimension = 512    # dimensions of each vector                         
#         self.k = k       # return 3 nearest neighbours        
#         self.embeddings = embeddings.astype('float32')
#         self.words = words
#         self.index = faiss.IndexFlatL2(self.dimension) 

#         # faiss.normalize_L2(self.embeddings)
#         self.index.add(self.embeddings)   # add the vectors and update the index
    
#     def get_nn(self, q_emb):
#         # faiss.normalize_L2(q_emb)
#         _, indices = self.index.search(q_emb, self.k)
#         nn_words = np.squeeze(self.words[indices]).tolist()
#         if self.k == 1:
#             nn_words = [nn_words]
#         return nn_words

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, predicate=False, tokenizer = None, transform_tok = None, device= None,  edge_select="random", spec=None, kw_size = 4, rw_size = 3, enc_model = "ViT-B_32", only_kw = False, norel= False, only_l2r = False, use_faiss = False, rc_posidx2 = False):
        self.only_kw = only_kw
        self.predicate = predicate
        self.kw_size = kw_size
        self.rw_size = rw_size
        self.device = device
        self.norel = norel
        self.only_l2r = only_l2r
        self.tokenizer_dec = tokenizer
        self.rc_posidx2 = rc_posidx2
        self.use_imagegraph = False

        # max num related words is 5 + relationship label  = 6, but make 8 to binary reasons?
        self.first_pos_idx = 5*rw_size
        print("using edge select type:", edge_select)
        pretok = ""
        if edge_select == "clipemb_pretok":
            pretok = "_pretok"
            edge_select = "clipemb"
        graph_path = '../data_files/CN_feats/concNet_nested_emb_'+ str(enc_model)+ pretok +'_maxtok.pkl'

        with open(graph_path, 'rb') as f:
                        self.lookupdict = pickle.load(f)

        self.edge_select = edge_select
        self.tokenizer = get_tokenizer("spacy")

        self.ps = PorterStemmer()
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.cossim = torch.nn.CosineSimilarity()
        print("loading clip in KG")
        self.model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.transformer_tokenizer = transform_tok
        self.spec = spec
        if spec is not None:
            self.remlist = [spec["bos_tokenid"], spec["eos_tokenid"]]
    
        # pth_clipemb = "../data_files/keyword_embedding_"+str(enc_model)+".pkl"
        pth_clipemb = "../data_files/keyword_embedding_ViT-B_32.pkl"
        
        with open(pth_clipemb, 'rb') as f:
                    all_wordemb = pickle.load(f)
        if device == "cpu":
            self.all_keywords = [word.encode('latin-1').decode('utf-8') for word in all_wordemb["captions"]]
        else:
            self.all_keywords = [word for word in all_wordemb["captions"]]

        self.all_keywordembed = torch.stack([word for word in all_wordemb["clip_embedding"]]).to(self.device)

        if use_faiss == True:
            self.edge_select = "clipemb_faiss"        
            self.newlookupdict = self.create_faiss_nested(self.lookupdict)
            self.kw_sim = get_fais_knn(np.array(self.all_keywords), np.copy(self.all_keywordembed.detach().cpu().numpy()), k= self.kw_size)


    def create_faiss_nested(self, lookupdict):
        newlookupdict = {}
        for unigram, all_edges in lookupdict.items():
            all_edges = np.array(all_edges)
            if len(all_edges) == 0:
                unigram_nn_obj = empty_fais_knn()
            else:
                edges_str_list = np.array(all_edges[:,0])
                edges_emb = torch.tensor(np.vstack(all_edges[:, 1]).astype(np.float32)).to(self.device)
                unigram_nn_obj = get_fais_knn(edges_str_list, np.copy(edges_emb.detach().cpu().numpy()), k= self.rw_size)
            newlookupdict[unigram] = unigram_nn_obj
        return newlookupdict


    def best_clip_score(self, rel_wordembed, max_edges, image_emb):
        res = self.cossim(rel_wordembed, image_emb)
        topNind = torch.topk(res.flatten(), max_edges).indices
        return topNind.detach().cpu().numpy()


    def get_ranked_edges(self, unigram, max_edges,  image_emb= None):
        if not self.edge_select == "clipemb_faiss":
            all_edges = np.array(self.lookupdict[unigram])

            if len(all_edges)<=max_edges:
                if self.edge_select == "clipemb" and len(all_edges) > 0:
                    return all_edges[:,0]
                else:
                    return all_edges
        
        if self.edge_select == "random":
            rand_ind = np.random.choice(len(all_edges), max_edges , replace=False)
            randitems =  all_edges[rand_ind]
            return randitems
        elif self.edge_select == "clipemb":
            edges_str_list = all_edges[:,0]
            edges_emb = torch.tensor(np.vstack(all_edges[:, 1]).astype(np.float32)).to(self.device)
            bestitems = self.best_clip_score(edges_emb, max_edges,image_emb= image_emb)
            bestwords = edges_str_list[bestitems]
            return bestwords
        elif self.edge_select == "clipemb_faiss":
            uni_nn_obj = self.newlookupdict[unigram]
            bestwords = uni_nn_obj.get_nn(image_emb.clone().detach().cpu().numpy())
            return bestwords


    def tokenize_wordid(self, sent_batch):
        token_batch = []
        for inp_sents in sent_batch:
            berttokens = self.transformer_tokenizer(inp_sents, padding=False)
            newberttokens = [word[1:-1] for word in berttokens.input_ids]
            token_batch.append(newberttokens)
        return token_batch


    def entities_tokenized_pretok(self, entities):
        order_rel = []
        combitoklist = []
        for ent in entities:
            if self.norel:
                relatedword_idx = int(not(ent[2]))
                combitoks = ent[relatedword_idx]
            else:
                combitoks = ent[0] + ent[1]
            combitoklist.append(combitoks)
            # print("entities:", self.tokenizer_dec.decode(combitoks))
            order_rel.append(ent[2])

        return combitoklist , order_rel


    def get_vm_from_imgid(self, contextfeat):
        # retrieve the Keywords for each contextfeat and call the knowledgewithvm
        all_img_embs = contextfeat
        sent_batch = []
        newcontfeat = contextfeat.float().squeeze(1).clone().detach().cpu().numpy()
        sent_batch = self.kw_sim.get_nn(newcontfeat) 

        return self.add_knowledge_with_vm(sent_batch, image_emb=all_img_embs, max_edges=self.rw_size, add_pad=True, max_length=64, prefix_size = None)

    def gen_imagegraph(self,tree,  unigram, entities, order_rel):
        tree.create_node(unigram, unigram, parent="root") 
        count = 0
        for ent in entities:
            ent_string = self.transformer_tokenizer.decode(ent)
            tree.create_node(ent_string, unigram + str(count) , parent=unigram)
            count +=1

    def add_knowledge_with_vm(self, sent_batch, image_emb=None, max_edges=5, add_pad=True, max_length=128, prefix_size = None):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        if prefix_size is not None:
            imgpref = ["[IMG]"]*prefix_size
            imgpref = " ".join(imgpref) + " "
            new_sentt_batch = [imgpref + sent for sent in sent_batch]
            sent_batch = new_sentt_batch
        
        split_sent_batch = self.tokenize_wordid(sent_batch)
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        sent_sizes = []
        first_kw_tok = []
        for sent_it, split_sent in enumerate(split_sent_batch):
            # create tree
            sent_tree = []
            pos_idx_tree = []   # for the relative idx of related word
            abs_idx_tree = []
            
            pos_idx = self.first_pos_idx        # the position indx for the transformer
            abs_idx = -1        # the idx of a token in the list
            abs_idx_src = []    # stores the idx of the keywords    
            num_toks = 0
            pos_idx_kw_start = [abs_idx + 1]

            split_sent_vanilla = sent_batch[sent_it]
            if self.use_imagegraph:
                img_tree = Tree()
                img_tree.create_node("root", "root") 


            for token_it, token in enumerate(split_sent):  
                unigram = split_sent_vanilla[token_it]
                # print("keyword:", unigram )  
                entities_words, entities,  order_rel = [], [], []

                if not self.only_kw:
                    if str(unigram) not in self.special_tags:
                        entities_words = self.get_ranked_edges(unigram, max_edges = max_edges, image_emb = image_emb[sent_it])
                    entities,  order_rel = [], []
                    # print("entities words:", entities_words)
                    if len(entities_words) != 0:
                        entities , order_rel = self.entities_tokenized_pretok(entities_words)
                if self.use_imagegraph:
                    self.gen_imagegraph(img_tree, unigram, entities, order_rel)
                sent_tree.append((token, entities))
                if str(token) in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                kw_pos = token_pos_idx[-1]
                kw_left_pos = kw_right_pos = kw_pos
                for j, ent in enumerate(entities):
                    """ 
                    ent_pos_idx : from token, +1 for each part of entity
                    ent_abs_idx : seems to be same actually
                    """

                    if self.only_l2r or order_rel[j]==0:
                        ent_pos_idx = [kw_right_pos + i for i in range(1, len(ent)+1)]
                        kw_right_pos = ent_pos_idx[-1]
                    else:
                        ent_pos_idx = [kw_left_pos - i for i in range(1, len(ent)+1)]
                        kw_left_pos = ent_pos_idx[-1]
                        ent_pos_idx.reverse()

                    if self.rc_posidx2 == False:
                        kw_left_pos = kw_right_pos = kw_pos
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

                pos_idx_kw_start.append( abs_idx +1)

            if self.use_imagegraph:
                img_tree.show()

            first_kw_tok.append(pos_idx_kw_start[:-1])  # drop the last because it is where a (nonexisting) following kw would start
            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if str(word) in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word 
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                #abs index of the keyword
                src_ids = item[0]
                # In my case one iteration cuz unigram so token is 1 thing.
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                # for all RW; related words
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1
            
            src_length = len(know_sent)
            sent_sizes.append(src_length)
            if len(know_sent) < max_length:
                PAD_TOKEN = self.spec["pad_tokenid"]
                pad_num = max_length - src_length
                know_sent += [PAD_TOKEN] * pad_num
                seg += [1] * pad_num
                pos += [0] * pad_num

                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
                for n, row in enumerate(visible_matrix[-pad_num:]):
                    row[src_length+ n] = 1
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
        
        maxlen = max(sent_sizes)

        return torch.tensor(np.array(know_sent_batch)).to(self.device)[:,:maxlen], torch.tensor(np.array(position_batch)).to(self.device)[:,:maxlen], torch.tensor(np.array(visible_matrix_batch)).to(self.device)[:,:maxlen,:maxlen], torch.tensor(np.array(seg_batch)).to(self.device)[:,:maxlen], np.array(first_kw_tok)

