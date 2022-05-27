
"""
From the file of keywords: openwebtext_10th.csv
Store the embeddings of each keyword as a dict of two entries: {"captions": keywords, "clip_embedding": embedding_list}
Use the already computed embeddings from the concNet if available: conceptNet_embedding_rn50x4.pkl"
"""
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
import pickle
import torch
import csv
import clip
import argparse
from tqdm import tqdm



def main(clipmodel_name: str):
    clipmodel = clipmodel_name.replace('/', '_')

    print("starting now ...")
    # load the keywords
    f = open('../../data_files/openwebtext_10th.csv', 'r')
    with f:
        reader = csv.reader(f)
        keywords = [word[0] for word in reader]
    print("len kw:", len(keywords))

    # load the precomputed conceptnet clip word embeddings
    cn_wordfeats_path = "../../data_files/conceptNet_embedding_"+clipmodel+".pkl"
    with open(cn_wordfeats_path, 'rb') as f:
                cn_wordfeats = pickle.load(f)
    cn_keyset = set(cn_wordfeats.keys())


    device = torch.device('cuda')
    model, _ = clip.load(clipmodel_name, device=device)
    out_path = "../../data_files/keyword_embedding_"+clipmodel+".pkl"
    file_to_store = open(out_path, "wb")


    embedding_list = []
    # for i, word in enumerate(keywords):
    for j in tqdm(range(len(keywords))):
        word = keywords[j]
        if word in cn_keyset:
            # make sure the stored embeddings are not tensors
            assert not torch.is_tensor(cn_wordfeats[word]), "keyword not in dict"
            prefix = cn_wordfeats[word]
        else:
            with torch.no_grad():
                word_tok = clip.tokenize(word).to(device)
                prefix  = model.encode_text(word_tok).squeeze().detach().cpu().numpy()

        embedding_list.append(prefix)
    pickle.dump({"captions": keywords, "clip_embedding":torch.tensor(np.array(embedding_list))}, file_to_store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', default="ViT-B/32", choices=('RN50x4', 'ViT-B/32'))

    args = parser.parse_args()
    exit(main(args.clip_model))
