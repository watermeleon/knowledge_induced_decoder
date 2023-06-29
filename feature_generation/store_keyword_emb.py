
"""
From the file of keywords: openwebtext_10th.csv
Store the embeddings of each keyword as a dict of two entries: 
{"captions": keywords, "clip_embedding": embedding_list}
Use the already computed embeddings from the concNet if available: conceptNet_embedding_rn50x4.pkl"
"""

import pickle
import csv
import argparse
from tqdm import tqdm
import numpy as np
import torch
import clip



def main(clipmodel_name, kw_file, cn_version):
    clipmodel = clipmodel_name.replace('/', '_')

    print("starting now ...")
    # load the keywords
    file = open(kw_file, 'r', encoding="UTF-8")
    with file:
        reader = csv.reader(file)
        keywords = [word[0] for word in reader]
    print("Number of keywords:", len(keywords))

    # load the precomputed conceptnet clip word embeddings
    embedding_file = "conceptNet_embedding_"+clipmodel+".pkl"
    embedding_file = "concNet_nested_emb_ViT-B_32_pretok_maxtok.pkl"
    cn_wordfeats_path = "../data_files/CN_feats/" + embedding_file
    with open(cn_wordfeats_path, 'rb') as file:
        cn_wordfeats = pickle.load(file)
    cn_keyset = set(cn_wordfeats.keys())

    device = torch.device('cuda')
    model, _ = clip.load(clipmodel_name, device=device)
    out_path = "../data_files/keyword_embedding_" + \
        clipmodel + cn_version + ".pkl"
    file_to_store = open(out_path, "wb")

    embedding_list = []
    for j in tqdm(range(len(keywords))):
        word = keywords[j]
        if word in cn_keyset:
            # make sure the stored embeddings are not tensors
            assert not torch.is_tensor(
                cn_wordfeats[word]), "keyword not in dict"
            prefix = cn_wordfeats[word]
        else:
            with torch.no_grad():
                word_tok = clip.tokenize(word).to(device)
                prefix = model.encode_text(
                    word_tok).squeeze().detach().cpu().numpy()

        embedding_list.append(prefix)
    pickle.dump({"captions": keywords, "clip_embedding": torch.tensor(
        np.array(embedding_list))}, file_to_store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', default="ViT-B/32",
                        choices=('RN50x4', 'ViT-B/32'))
    parser.add_argument('--kw_file', type=str,
                        default='../data_files/openwebtext_10th.csv')
    parser.add_argument('--cn_version', type=str, default="")

    args = parser.parse_args()
    main(args.clip_model, args.kw_file, args.cn_version)
