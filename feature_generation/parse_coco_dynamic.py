import argparse
import json
from tqdm import tqdm
from skimage import io
from PIL import Image
import h5py

import torch
import clip








def get_last_emb_vit(model, image):
    """Return the last embedding of an input image of a ViT image encoder"""
    with torch.no_grad():
        x = image.type(model.dtype)
        visual_enc = model.visual

        # copied below
        x = visual_enc.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                visual_enc.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + visual_enc.positional_embedding.to(x.dtype)
        x = visual_enc.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual_enc.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[0][1:]  # remove the CLS feature
        return x


def get_last_emb_RN(model, image):
    """Return the last embedding of an input image of a ResNet image encoder"""
    with torch.no_grad():
        x = image.type(model.dtype)
        visual_enc = model.visual

        def stem(x):
            for conv, bn in [
                (visual_enc.conv1, visual_enc.bn1),
                (visual_enc.conv2, visual_enc.bn2),
                (visual_enc.conv3, visual_enc.bn3),
            ]:
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


def main(clip_model_type: str, emb_layer: str):
    device = torch.device("cuda:0")
    clip_model_name = clip_model_type.replace("/", "_")
    out_path = f"./data/coco/clipemb_{clip_model_name}_{emb_layer}_train7.h5"
    output_file = h5py.File(out_path, "w")

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # get the right function to extract the prepool features, is different for the Transformer and ResNet models
    if clip_model_type == "ViT-B/32":
        get_last_emb = get_last_emb_vit
    else:
        get_last_emb = get_last_emb_RN

    path_to_data = "../data_files/"
    dataset_name = "dataset_coco.json"
    f = open(path_to_data + dataset_name)
    data = json.load(f)
    data = data["images"]
    f.close()

    print("%0d captions loaded from json " % len(data))
    saved_ids = []

    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["cocoid"]
        if img_id in saved_ids:
            continue
        saved_ids.append(img_id)

        filename = d["filename"]
        file_path = ".././Datasets/MsCoco/" + d["filepath"] + "/" + filename

        image = io.imread(file_path)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            if emb_layer == "prepoolfeats":
                embedding = get_last_emb(clip_model, image).cpu()
            else:
                embedding = clip_model.encode_image(image).cpu()

        file_name = "%d_features" % int(img_id)
        output_file.create_dataset(file_name, data=embedding)

    print("Done")
    output_file.close()
    print("%0d embeddings saved " % len(saved_ids))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model_type",
        default="RN50x4",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"),
    )
    parser.add_argument(
        "--emb_layer", default="contextfeat", choices=("contextfeat", "prepoolfeats")
    )
    args = parser.parse_args()
    main(args.clip_model_type, args.emb_layer)
