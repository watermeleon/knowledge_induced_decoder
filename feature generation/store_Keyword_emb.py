import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
import pickle
import torch
import csv
import clip
import argparse



def main(clipmodel: str):
    print("starting now ...")

    # load the keywords
    f = open('../data_files/openwebtext_10th.csv', 'r')
    with f:
        reader = csv.reader(f)
        keywords = [word[0] for word in reader]
    print("len kw:", len(keywords))

    # load the precomputed conceptnet clip word embeddings
    cn_wordfeats_path = "../data_files/conceptNet_embedding_rn50x4.pkl"
    with open(cn_wordfeats_path, 'rb') as f:
                cn_wordfeats = pickle.load(f)
    cn_keyset = set(cn_wordfeats.keys())

    # device = torch.device('cpu')
    device = torch.device('cuda')
    if clipmodel == "huggingface":
        model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    elif clipmodel == "clip_github":
        model, _ = clip.load("RN50x4", device=device)



    out_path = "../data_files/keyword_embedding_rn50x4_cpu.pkl"
    file_to_store = open(out_path, "wb")


    embedding_list = []
    for i, word in enumerate(keywords):
        if word in cn_keyset:
            # make sure the stored embeddings are not tensors
            assert not torch.is_tensor(cn_wordfeats[word]), "keyword not in dict"
            word_emb = cn_wordfeats[word]
        else:
            with torch.no_grad():
                if clipmodel == "huggingface":
                    word_tok =   tokenizer(word, padding=True, return_tensors="pt").to(device)
                    word_emb = model_clip.get_text_features(**word_tok)
                    word_emb =  word_emb.squeeze().detach().cpu().numpy()
                else:
                    word_tok = clip.tokenize(word).to(device)
                    word_emb  = model.encode_text(word_tok)
                    word_emb = word_emb.detach().squeeze().cpu().numpy()

        embedding_list.append(word_emb)

        if i%100 == 0:
            print(i)
    pickle.dump({"captions": keywords, "clip_embedding":torch.tensor(np.array(embedding_list))}, file_to_store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clipmodel', default="clip_github", choices=('clip_github','huggingface'))
    args = parser.parse_args()
    exit(main(args.clipmodel))
