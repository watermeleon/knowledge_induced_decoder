
from knowgraph_conceptnet import KnowledgeGraph
import numpy as np

from transformers import  CLIPTokenizer
import pickle
import torch
import argparse



def create_nested(clipmodel):
    # load the nested dict
    pth_clipemb = "../data_files/concNet_filtBanana_save.pkl"
    with open(pth_clipemb, 'rb') as f:
                CN_dict = pickle.load(f)

    if clipmodel == "huggingface":
        tokenizerBW =  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        cn_wordfeats_path = "../Datasets/conceptNet_embedding_hf_lisa_2.pkl"
        out_path = "../data_files/concNetFilt_emb_Banana_lisa2.pkl"

    else: 
        cn_wordfeats_path = "../data_files/conceptNet_embedding_rn50x4.pkl"
        out_path = "../data_files/concNet_nested_emb_rn50x4.pkl"

    # load the stored embeddings:
    with open(cn_wordfeats_path, 'rb') as f:
                cn_wordfeats = pickle.load(f)
    if clipmodel == "huggingface":
        cn_wordfeats =  {str(k).encode('latin-1').decode('utf-8') : v for k,v in cn_wordfeats.items()}

    file_to_store = open(out_path, "wb")

    # device = torch.device('cpu')
    # device = torch.device('cuda')

    CN_wordsNemb_dict = {}
    totemb = 0
    for i, (keyword, relword_list) in enumerate(CN_dict.items()):

        rw_emb_list = []
        for relword_item in relword_list:
            # l2r edge is 0, so need to invert the binary
            rw_idx = int(not(relword_item[-1]))
            related_word = relword_item[rw_idx]

            # assert related_word in cn_wordfeats, "related word not in wordfeats"
            try:
                rw_emb = cn_wordfeats[related_word]
            except:
                print("kw, relword:", keyword, relword_item)
                quit()

            rw_emb_list.append([relword_item, rw_emb])
            totemb += 1

        CN_wordsNemb_dict[keyword] = np.array(rw_emb_list)
        if i%100 == 0:
            print(i, totemb)
        # break
    print(totemb)
    pickle.dump(CN_wordsNemb_dict, file_to_store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clipmodel', default="clip_github", choices=('clip_github','huggingface'))
    parser.add_argument('--createfile', default="nested_with_emb", choices=('nested_with_emb','nested_pretok'))

    args = parser.parse_args()
    if args.createfile == "nested_with_emb":
        create_nested(args.clipmodel)
    elif args.createfile == "nested_pretok":
        create_pretok(args.clipmodel)
    # exit(main(args.clipmodel))
