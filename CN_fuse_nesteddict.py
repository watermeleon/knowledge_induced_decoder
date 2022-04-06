
from knowgraph_conceptnet import KnowledgeGraph
import numpy as np

from transformers import  CLIPTokenizer
import pickle
import torch

tokenizerBW =  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

pth_clipemb = "../data_files/concNet_filtBanana_save.pkl"
# pth_clipemb = "/media/leonardo/Floppy/concNet_filtBanana_save.pkl"
with open(pth_clipemb, 'rb') as f:
            CN_dict = pickle.load(f)


# load the stored embeddings:
# cn_wordfeats_path = "/media/leonardo/Floppy/conceptNet_embedding_hf_lisa_2.pkl"
cn_wordfeats_path = "../Datasets/conceptNet_embedding_hf_lisa_2.pkl"

with open(cn_wordfeats_path, 'rb') as f:
            cn_wordfeats = pickle.load(f)
cn_wordfeats =  {str(k).encode('latin-1').decode('utf-8') : v for k,v in cn_wordfeats.items()}


# device = torch.device('cpu')
device = torch.device('cuda')


CN_wordsNemb_dict = {}
i = 0
out_path = "concNetFilt_emb_Banana_lisa2.pkl"

file_to_store = open(out_path, "wb")

# word_prefix = ""
totemb = 0
for keyword, relword_list in CN_dict.items():

    rw_emb_list = []
    for relword_item in relword_list:
        # l2r edge is 0, so need to invert the binary
        rw_idx = int(not(relword_item[-1]))
        related_word = relword_item[rw_idx]
                
        if related_word not in cn_wordfeats:
            print("not in wordfeats")
        try:
            # related_word = str(related_word).encode('latin-1').decode('utf-8')
            rw_emb = cn_wordfeats[related_word]
        except:
            print("kw, relword:", keyword, relword_item)
            quit()
        if len(rw_emb) != 512:
            print("len is not 512, emb is:", rw_emb)


        rw_rel = relword_item

        rw_emb_list.append([rw_rel, rw_emb])
        totemb += 1

    CN_wordsNemb_dict[keyword] = np.array(rw_emb_list)
    if i%100 == 0:
        print(i, totemb)
    i+=1
    # break
print(totemb)
pickle.dump(CN_wordsNemb_dict, file_to_store)