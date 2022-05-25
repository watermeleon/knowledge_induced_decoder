"""
Using the stored embeddings for each conceptnet_concept
Use the nested filterd CN dict, and create a similar dict which also contains embeddings
Input: conceptnet_filt_nest_labels.pkl
outfile: dict[keyword] = listof : [ [tokword1, tokword2, relword_item[-1]], related_concept_embedding]
"""

from knowgraph_conceptnet import KnowledgeGraph
import numpy as np

from transformers import  CLIPTokenizer, CLIPTokenizerFast
import pickle
import torch
import argparse
import clip
import ftfy

allrel = ['<|Antonym|>', '<|AtLocation|>', '<|CapableOf|>', '<|Causes|>', '<|CausesDesire|>', '<|CreatedBy|>', '<|DefinedAs|>', '<|DerivedFrom|>', '<|Desires|>', '<|DistinctFrom|>', '<|Entails|>', '<|EtymologicallyDerivedFrom|>', '<|EtymologicallyRelatedTo|>', '<|FormOf|>', '<|HasA|>', '<|HasContext|>', '<|HasFirstSubevent|>', '<|HasLastSubevent|>', '<|HasPrerequisite|>', '<|HasProperty|>', '<|HasSubevent|>', '<|InstanceOf|>', '<|IsA|>', '<|LocatedNear|>', '<|MadeOf|>', '<|MannerOf|>', '<|MotivatedByGoal|>', '<|NotCapableOf|>', '<|NotDesires|>', '<|NotHasProperty|>', '<|PartOf|>', '<|ReceivesAction|>', '<|RelatedTo|>', '<|SimilarTo|>', '<|SymbolOf|>', '<|Synonym|>', '<|UsedFor|>', '<|capital|>', '<|field|>', '<|genre|>', '<|genus|>', '<|influencedBy|>', '<|knownFor|>', '<|language|>', '<|leader|>', '<|occupation|>', '<|product|>']
allrel = [ftfy.fix_text(rel) for rel in allrel]
print(allrel)

def create_nested(clipmodel, pretok, tok_thresh):
    print("started creating nested dict...")
    # load the nested dict
    pth_clipemb = "../data_files/CN_feats/conceptnet_filt_nest_labels.pkl"
    with open(pth_clipemb, 'rb') as f:
                CN_dict = pickle.load(f)
    print("hi")

    pretok_label = "_pretok" if pretok else ""
    if clipmodel == "huggingface":
        cn_wordfeats_path = "../data_files/CN_feats/conceptNet_embedding_ViT.pkl"
        # out_path = "../data_files/concNetFilt_emb_Banana_lisa2" + pretok_label + ".pkl"
        out_path = "../data_files/CN_feats/concNet_nested_emb_ViT" + pretok_label + "_maxtok.pkl"

    else: 
        cn_wordfeats_path = "../data_files/CN_feats/conceptNet_embedding_rn50x4.pkl"
        out_path = "../data_files/CN_feats/concNet_nested_emb_rn50x4" + pretok_label + ".pkl"

    # use the same tokenizer for both github and HF clip
    tokenizerBW =  CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    tokenizerBW.add_tokens(allrel, special_tokens=True)
    
    # load the stored embeddings:
    with open(cn_wordfeats_path, 'rb') as f:
                cn_wordfeats = pickle.load(f)
    if clipmodel == "huggingface":
        cn_wordfeats =  {str(k).encode('latin-1').decode('utf-8') : v for k,v in cn_wordfeats.items()}

    file_to_store = open(out_path, "wb")

    CN_wordsNemb_dict = {}
    totemb = 0
    for i, (keyword, relword_list) in enumerate(CN_dict.items()):

        rw_emb_list = []
        for relword_item in relword_list:
            # l2r edge is 0, so need to invert the binary
            rw_idx = int(not(relword_item[-1]))
            related_word = relword_item[rw_idx]
            if related_word not in cn_wordfeats:
                print(keyword,"relword:", relword_item, related_word)
            assert related_word in cn_wordfeats, "related word not in wordfeats"
            rw_emb = cn_wordfeats[related_word]
            if pretok:
                word1, word2 = relword_item[0], relword_item[1]
                tokword1 = tokenizerBW(word1, padding=True, return_tensors="pt").input_ids.squeeze().tolist()[1:-1]
                tokword2 = tokenizerBW(word2, padding=True, return_tensors="pt").input_ids.squeeze().tolist()[1:-1]
                if len(tokword1) > tok_thresh or len(tokword2) > tok_thresh:
                    continue
                relword_item = [tokword1, tokword2, relword_item[-1]]
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
    parser.add_argument('--pretok', action='store_true')
    parser.add_argument('--tok_thresh', type=int, default=5)

    args = parser.parse_args()

    create_nested(args.clipmodel, args.pretok, args.tok_thresh)

