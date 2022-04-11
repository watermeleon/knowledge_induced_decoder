
# from knowgraph_conceptnet import KnowledgeGraph
# from data.utils import *
from sys import prefix
import numpy as np
# from scipy import spatial

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
import pickle
import torch
# from transformers import AutoTokenizer
import csv
from collections import defaultdict
import clip

print("hey")
# all_conceptnetwords = set()

# file = open('../data_files/ass_onlyenglish.csv')
# csvreader = csv.reader(file)
# lookupdict_ow = defaultdict(list)

# for i, row in enumerate(csvreader):
#     conc_one = str(row[2].split("/")[3]).replace("_", " ")
#     conc_two = str(row[3].split("/")[3]).replace("_", " ")
#     all_conceptnetwords.add(conc_one)
#     all_conceptnetwords.add(conc_two)
    # load the precomputed conceptnet clip word embeddings
cn_wordfeats_path = "../data_files/conceptNet_embedding_rn50x4.pkl"
with open(cn_wordfeats_path, 'rb') as f:
            cn_wordfeats = pickle.load(f)


# print(len(all_conceptnetwords))

# device = torch.device('cpu')
device = torch.device('cuda')

# clipmodel = "clip_github"

# if clipmodel == "huggingface":
#     model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# elif clipmodel == "clip_github":
#     model, preprocess = clip.load("RN50x4", device=device)
# else:
#     print("unkown clipmodel: use huggingface or github model")
#     quit()


pick_dict = {}
i = 0
out_path = "conceptNet_embedding_rn50x4_2.pkl"
file_to_store = open(out_path, "wb")

word_prefix = ""

for word_list in cn_wordfeats.keys():
    word = word_prefix + word_list
    # word = "golden gate bridge"
    # with torch.no_grad():
        # if clipmodel == "huggingface":
        #     word_tok =   tokenizer(word, padding=True, return_tensors="pt").to(device)
        #     prefix = model_clip.get_text_features(**word_tok)
        #     pefix =  prefix.squeeze().detach().cpu().numpy()
        # else:
        #     word_tok = clip.tokenize(word).to(device)
        #     prefix  = model.encode_text(word_tok)
    prefix = cn_wordfeats[word].squeeze().detach().cpu().numpy()
    word2 = str(word).encode('latin-1').decode('utf-8')
    pick_dict[word2] = prefix

    if i%1000 == 0:
        print(i)
    i+=1
    # break
pickle.dump(pick_dict, file_to_store)