from sys import prefix
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import h5py
from os import listdir
from os.path import isfile, join
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_last_emb_vit(model, image):
    with torch.no_grad():
        x = image.type(model.dtype)
        visual_enc = model.visual
        
        # copied below
        x = visual_enc.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([visual_enc.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual_enc.positional_embedding.to(x.dtype)
        x = visual_enc.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual_enc.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[0][1:]  #remove the CLS feature   
        return x


def get_last_emb_RN(model, image):
    with torch.no_grad():
        x = image.type(model.dtype)
        visual_enc = model.visual

        
        def stem(x):
            for conv, bn in [(visual_enc.conv1, visual_enc.bn1), (visual_enc.conv2, visual_enc.bn2), (visual_enc.conv3, visual_enc.bn3)]:
                x = visual_enc.relu(bn(conv(x)))
            x = visual_enc.avgpool(x)
            return x
        x = x.type(visual_enc.conv1.weight.dtype)
        x = stem(x)
        x = visual_enc.layer1(x)
        x = visual_enc.layer2(x)       
        x = visual_enc.layer3(x)
        x = visual_enc.layer4(x)

        x = x[0].T.reshape([81, 2560])
        return x


def main(clip_model_type: str, emb_layer:str, split:str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"../../Datasets/NoCaps/clipemb_nocaps_{clip_model_name}_{emb_layer}_{split}_TEMP.h5"
    hf = h5py.File(out_path, 'w')

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # get the right function to extract the prepool features, is different for the Transformer and ResNet models
    if clip_model_type == "ViT-B/32":
        get_last_emb = get_last_emb_vit
    else:
        get_last_emb = get_last_emb_RN

    images_path = "../../Datasets/NoCaps/NoCaps_" +str(split)+"/"
    all_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    print("%0d captions loaded from json " % len(all_files))
    saved_ids = []

    for i in tqdm(range(len(all_files))):
        filename = all_files[i]
        img_id = filename.split("_")[-1][:-4]
        print(img_id)
        saved_ids.append(img_id)

        file_path = images_path + filename

        image = io.imread(file_path)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            if emb_layer == "prepoolfeats":
                embedding = get_last_emb(clip_model, image).cpu()
            else:
                embedding = clip_model.encode_image(image).cpu().float()

        file_name = '%d_features' % int(img_id)
        hf.create_dataset(file_name, data=embedding)
    
    print('Done')
    hf.close()
    print("%0d embeddings saved " % len(saved_ids))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--emb_layer', default="contextfeat", choices=('contextfeat', 'prepoolfeats'))
    parser.add_argument('--data_split', default="val", choices=('val', 'test'))

    args = parser.parse_args()
    exit(main(args.clip_model_type, args.emb_layer, args.data_split))
