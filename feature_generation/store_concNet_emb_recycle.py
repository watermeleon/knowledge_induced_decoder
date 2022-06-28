"""
From the nested concnet dict: ass_onlyenglish.csv
Store the embeddings of each word in the English only ConceptNet.
outfile: dict[concept_word] = concept_embedding
"""
import numpy as np
from transformers import CLIPModel, CLIPTokenizerFast
import pickle
import torch
from tqdm import tqdm

import csv
from collections import defaultdict
import clip
import argparse


def store_conceptnet_embeddings(clipmodel_name, args):
    clipmodel = clipmodel_name.replace('/', '_')
    print("hey")
    all_conceptnetwords = set()

    file = open('../../data_files/ass_onlyenglish.csv')
    csvreader = csv.reader(file)
    for i, row in enumerate(csvreader):
        conc_one = str(row[2].split("/")[3]).replace("_", " ")
        conc_two = str(row[3].split("/")[3]).replace("_", " ")
        all_conceptnetwords.add(conc_one)
        all_conceptnetwords.add(conc_two)

    all_conceptnetwords = list(all_conceptnetwords)    
    print(len(all_conceptnetwords))

    # device = torch.device('cpu')
    device = torch.device(args.device)
    model, _ = clip.load(clipmodel_name, device=device)

    pick_dict = {}
    out_path = "../../data_files/conceptNet_embedding_"+clipmodel+args.cn_version+".pkl"
    file_to_store = open(out_path, "wb")
    if args.use_stored:
        # load the precomputed conceptnet clip word embeddings
        cn_wordfeats_path = "../../data_files/CN_feats/conceptNet_embedding_"+clipmodel+".pkl"
        with open(cn_wordfeats_path, 'rb') as f:
                cn_wordfeats_stored = pickle.load(f)
        cn_keyset = set(cn_wordfeats.keys())

    word_prefix = ""
    # for word_list in all_conceptnetwords:
    for j in tqdm(range(len(all_conceptnetwords))):
        word_list = all_conceptnetwords[j]
        word = word_prefix + word_list
        word2 = str(word).encode('latin-1').decode('utf-8')
        if args.use_stored:
            if word2 in cn_keyset:
                prefix = cn_wordfeats_stored[word2]
                pick_dict[word2] = prefix

        # word = "golden gate bridge"
        with torch.no_grad():
            word_tok = clip.tokenize(word).to(device)
            prefix  = model.encode_text(word_tok).squeeze().detach().cpu().numpy()

        # word2 = str(word).encode('latin-1').decode('utf-8')
        pick_dict[word2] = prefix

    pickle.dump(pick_dict, file_to_store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--clipmodel', default="clip_github", choices=('clip_github','huggingface'))
    # parser.add_argument('--tokenizer', default="clip_github", choices=('clip_github','huggingface'))
    parser.add_argument('--clip_model', default="ViT-B/32", choices=('RN50x4', 'ViT-B/32'))
    parser.add_argument('--device', default="cuda", choices=('cuda','cpu'))
    parser.add_argument('--cn_version', type=str, default="")
    parser.add_argument('--use_stored', action='store_true')




    args = parser.parse_args()

    store_conceptnet_embeddings(args.clip_model, args)
