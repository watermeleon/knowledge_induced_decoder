
print("hey")

from knowgraph_conceptnet import KnowledgeGraph
import numpy as np

from transformers import  CLIPTokenizer
import pickle
import torch


tokenizerBW =  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


# pth_clipemb = "../data_files/concNetFilt_emb_Banana_lisa2_pretok.pkl"
pth_clipemb = "/media/leonardo/Floppy/concNetFilt_emb_Banana_lisa2.pkl"
with open(pth_clipemb, 'rb') as f:
            CN_dict = pickle.load(f)


# load the stored embeddings:
# cn_wordfeats_path = "/media/leonardo/Floppy/conceptNet_embedding_hf_lisa_2.pkl"
# cn_wordfeats_path = "../Datasets/conceptNet_embedding_hf_lisa_2.pkl"

# with open(cn_wordfeats_path, 'rb') as f:
#             cn_wordfeats = pickle.load(f)
# cn_wordfeats =  {str(k).encode('latin-1').decode('utf-8') : v for k,v in cn_wordfeats.items()}


# device = torch.device('cpu')
device = torch.device('cuda')


CN_wordsNemb_dict = {}
i = 0

out_path = "../data_files/concNetFilt_emb_Banana_lisa2_pretok.pkl"

file_to_store = open(out_path, "wb")

# word_prefix = ""
totemb = 0
for keyword, relword_list in CN_dict.items():

    rw_emb_list = []
    for relword_item in relword_list:
        # l2r edge is 0, so need to invert the binary
        rw_idx = int(not(relword_item[0][-1]))
        if rw_idx not in [0,1]:
            print("this is not a binary digit:", rw_idx)
            quit()

        # related_word = relword_item[rw_idx]
        # rw_tok = tokenizerBW(related_word, padding=True, return_tensors="pt").to(device)
        
        # word1 or 2 could either be relation or RW, doesn't mather for here
        word1 = relword_item[0][0].tolist()
        word2 = relword_item[0][1].tolist()
        tokword1 = tokenizerBW(word1, padding=True, return_tensors="pt").input_ids.squeeze()
        tokword2 = tokenizerBW(word2, padding=True, return_tensors="pt").input_ids.squeeze()
        relword_item[0][0] = tokword1
        relword_item[0][1] = tokword2

        rw_emb_list.append(relword_item)
        totemb += 1

    CN_wordsNemb_dict[keyword] = np.array(rw_emb_list)
    if i%100 == 0:
        print(i, totemb)
    i+=1
    # break
print(totemb)
pickle.dump(CN_wordsNemb_dict, file_to_store)