"""
Using the stored embeddings for each conceptnet_concept
Use the nested filterd CN dict, and create a similar dict which also contains embeddings
Input: conceptnet_filt_nest_labels.pkl
outfile: dict[keyword] = listof : [ [tokword1, tokword2, relword_item[-1]], related_concept_embedding]
"""
import pickle
import argparse
import numpy as np
from transformers import CLIPTokenizerFast


def create_nested(clipmodel, pretok, tok_thresh, nested_cn_path, cn_version):
    clipmodel = clipmodel.replace("/", "_")

    print("started creating nested dict...")
    # load the nested dict
    pth_clipemb = nested_cn_path
    with open(pth_clipemb, "rb") as f:
        cn_dict = pickle.load(f)

    pretok_label = "_pretok" if pretok else ""
    cn_wordfeats_path = (
        "../../data_files/CN_feats/conceptNet_embedding_" + clipmodel + ".pkl"
    )
    out_path = (
        "../../data_files/CN_feats/concNet_nested_emb_"
        + clipmodel
        + pretok_label
        + cn_version
        + "_maxtok.pkl"
    )

    # use the same tokenizer for both github and HF clip
    tokenizer = CLIPTokenizerFast.from_pretrained(
        "../models/tokenizers_stored/CLIPTokenizerFast"
    )

    # load the stored embeddings:
    with open(cn_wordfeats_path, "rb") as f:
        cn_wordfeats = pickle.load(f)

    file_to_store = open(out_path, "wb")

    cn_wordsNemb_dict = {}
    totemb = 0
    for i, (keyword, relword_list) in enumerate(cn_dict.items()):

        rw_emb_list = []
        for relword_item in relword_list:
            # l2r edge is 0, so need to invert the binary
            rw_idx = int(not (relword_item[-1]))
            related_word = relword_item[rw_idx]
            if related_word not in cn_wordfeats:
                print(keyword, "relword:", relword_item, related_word)
            assert related_word in cn_wordfeats, "related word not in wordfeats"
            rw_emb = cn_wordfeats[related_word]

            # store al the words pretokenized
            if pretok:
                word1, word2 = relword_item[0], relword_item[1]
                tokword1 = (
                    tokenizer(word1, padding=True, return_tensors="pt")
                    .input_ids.squeeze()
                    .tolist()[1:-1]
                )
                tokword2 = (
                    tokenizer(word2, padding=True, return_tensors="pt")
                    .input_ids.squeeze()
                    .tolist()[1:-1]
                )
                if len(tokword1) > tok_thresh or len(tokword2) > tok_thresh:
                    continue
                relword_item = [tokword1, tokword2, relword_item[-1]]
            rw_emb_list.append([relword_item, rw_emb])
            totemb += 1

        cn_wordsNemb_dict[keyword] = np.array(rw_emb_list)
        if i % 100 == 0:
            print(i, totemb)

    print(totemb)
    pickle.dump(cn_wordsNemb_dict, file_to_store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model", default="ViT-B/32", choices=("RN50x4", "ViT-B/32")
    )
    parser.add_argument(
        "--nested_cn_path",
        type=str,
        default="../../data_files/CN_feats/conceptnet_filt_nest_labels.pkl",
    )
    parser.add_argument("--pretok", action="store_true")
    parser.add_argument("--cn_version", type=str, default="")

    parser.add_argument("--tok_thresh", type=int, default=4)

    args = parser.parse_args()

    create_nested(
        args.clip_model,
        args.pretok,
        args.tok_thresh,
        args.nested_cn_path,
        args.cn_version,
    )
