# Generate the necessary features and files

## Image Features

For Coco: run  `parse_coco_dynamic.py`
For Nocaps: run  `parse_nocaps_dynamic.py`
Both files contain the following arguments:
| Argument | Possible values |
|------|------|
| `--clip_model_type` | The clip model (choices: RN50, RN101, RN50x4, ViT-B/32) |
| `--emb_layer` |  The embedding layer (choices: contextfeat, prepoolfeats)  |

To run our model, run either `parse_coco_dynamic.py` or `parse_nocaps_dynamic.py` depending on the dataset. Run the appropriate python file twice, once using each `--emb_layer` option, and select the appropriate Vision Encoder backbone using `--clip_model_type` .


## ConceptNet and Keywords Files

1. To gather the top10th keywords follow the notebook:  
    - `keywords_openwebtext.ipynb`
2. We already provide the conceptnet file which only kept the edges where both concepts are english.
    - How ConceptNet is created: **TODO**
3. Create the filetered nested dict: 
    - `python gen_nested_CN.py`
4. For each item in the English conceptnet, store its embeddings in :
    - `python store_concNet_emb.py` 
5. Merge the nested CN dict with the embedding using:
    - `python CN_fuse_nesteddict.py --pretok --tok_thresh 4`
6. Store all the keyword embeddings, uses the overlap of already stored CN embeddings: 
    - `python store_Keyword_emb.py`


| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
