"""
From the nested concnet dict: ass_onlyenglish.csv
Store the embeddings of each word in the English only ConceptNet.
outfile: dict[concept_word] = concept_embedding
"""

import pickle
import csv
import argparse
from tqdm import tqdm
import torch
import clip


def store_conceptnet_embeddings(clipmodel_name, device):
    clipmodel = clipmodel_name.replace('/', '_')
    all_conceptnetwords = set()

    print("Start reading conceptNet")
    file = open('../../data_files/ass_onlyenglish.csv')
    csvreader = csv.reader(file)
    for _, row in enumerate(csvreader):
        conc_one = str(row[2].split("/")[3]).replace("_", " ")
        conc_two = str(row[3].split("/")[3]).replace("_", " ")
        all_conceptnetwords.add(conc_one)
        all_conceptnetwords.add(conc_two)

    all_conceptnetwords = list(all_conceptnetwords)
    print("Number of concepts found:", len(all_conceptnetwords))

    device = torch.device(device)
    model, _ = clip.load(clipmodel_name, device=device)

    pick_dict = {}
    out_path = "../../data_files/conceptNet_embedding_"+clipmodel+".pkl"
    file_to_store = open(out_path, "wb")

    word_prefix = ""
    # for word_list in all_conceptnetwords:
    for j in tqdm(range(len(all_conceptnetwords))):
        word_list = all_conceptnetwords[j]
        word = word_prefix + word_list
        with torch.no_grad():
            word_tok = clip.tokenize(word).to(device)
            prefix = model.encode_text(
                word_tok).squeeze().detach().cpu().numpy()

        word2 = str(word).encode('latin-1').decode('utf-8')
        pick_dict[word2] = prefix

    pickle.dump(pick_dict, file_to_store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', default="ViT-B/32",
                        choices=('RN50x4', 'ViT-B/32'))
    parser.add_argument('--device', default="cuda", choices=('cuda', 'cpu'))

    args = parser.parse_args()

    store_conceptnet_embeddings(args.clip_model, args.device)
