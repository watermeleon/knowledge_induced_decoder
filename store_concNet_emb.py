
from knowgraph_conceptnet import KnowledgeGraph
# from data.utils import *
import numpy as np
# from scipy import spatial

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
import pickle
import torch
# from transformers import AutoTokenizer
import csv
from collections import defaultdict


print("hey")
all_conceptnetwords = set()

file = open('../data_files/ass_onlyenglish.csv')
csvreader = csv.reader(file)
lookupdict_ow = defaultdict(list)

for i, row in enumerate(csvreader):
    conc_one = str(row[2].split("/")[3]).replace("_", " ")
    conc_two = str(row[3].split("/")[3]).replace("_", " ")
    all_conceptnetwords.add(conc_one)
    all_conceptnetwords.add(conc_two)



print(len(all_conceptnetwords))

# device = torch.device('cpu')
device = torch.device('cuda')


# model_txt_hf = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

pick_dict = {}
i = 0
out_path = "conceptNet_embedding_hf_lisa_2.pkl"
file_to_store = open(out_path, "wb")

word_prefix = ""

for word_list in all_conceptnetwords:
    word = word_prefix + word_list
    # word = "golden gate bridge"
    word_tok =   tokenizer(word, padding=True, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     prefix = model_clip(**word_tok)
    # prefix = prefix.pooler_output.cpu().numpy()[0]
    prefix = model_clip.get_text_features(**word_tok)


    pick_dict[word] = prefix.squeeze().detach().cpu().numpy()

    if i%1000 == 0:
        print(i)
    i+=1
    # break
pickle.dump(pick_dict, file_to_store)