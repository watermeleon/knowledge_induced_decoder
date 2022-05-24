import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from data import ImageDetectionsField, TextField, RawField, ClipEmbDetectionsField
from data import COCO, DataLoader
import evaluation
# from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from models.transformer import Transformer, MemoryAugmentedEncoder, PromptDecoder, ScaledDotProductAttentionMemory, MultiLevelEncoder, ScaledDotProductAttention, VanillaDecoder, ParallelPromptDecoder , StackedPromptDecoder
from knowgraph_conceptnet import KnowledgeGraph
from transformers import CLIPTokenizer, CLIPTokenizerFast, AutoTokenizer

import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import ftfy
import json

from pathlib import Path

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def gen_captions(model, dataloader, spec, transform_tok, out_file):
    import itertools
    model.eval()

    all_captions = []
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            # images = images.to(device)
            images, img_ids = images
            context_feats =  torch.stack(caps_gt[1])
            context_feats = context_feats[:,0,:,:]
            images, context_feats = images.to(device), context_feats.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, context_feats, 40, spec['eos_tokenid'], 5, out_size=1)

            caps_gen = [transform_tok.decode(sent) for sent in out] 
            caps_gen = [sent.split("<|endoftext|>")[0].strip() for sent in caps_gen]

            for (img_id, gen_i) in zip(img_ids, caps_gen):
                entry = {"caption": gen_i, "image_id":img_id.item() }
                all_captions.append(entry)
            # break
            pbar.update()

    with open(out_file, 'w') as f:
        json.dump(all_captions, f)
    # return scores



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PromptDecoder - KG- Transformer')
   # training basics
    parser.add_argument('--exp_name', type=str, default='kg_prompt_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)

    parser.add_argument('--resume', type=str, default="best", choices=['best', 'last'])

    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])
    parser.add_argument('--feat_size', type=int, default=2048)

    # paths
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--contextfeat_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')

    # encoder and decoder
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--enc_model', type=str, default="ViT", choices=['ViT', 'rn50x4'])
    
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--seg_token', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--decoder', type=str, default="kg_infused", choices=['vanilla', 'kg_infused', 'prompt_decoder', 'stacked'])
    parser.add_argument('--one_kw_token', action='store_true') # for the stackeddecoder

    # training specifics
    parser.add_argument('--start_rl', action='store_true')
    parser.add_argument('--no_rl', action='store_true')
    parser.add_argument('--tokenizer', type=str, default="bert", choices=['bert', 'clip'])
    parser.add_argument('--pt_token_emb', action='store_true') # for the KG part of the decoder
    parser.add_argument('--optimizer', type=str, default="adam", choices=['adam', 'adamW'])
    parser.add_argument('--sampling_method', type=str, default="beam", choices=['topk', 'beam', 'nucleus'])
    parser.add_argument('--sampling_temp', type=float, default=1)

    # knowledge graph related
    parser.add_argument('--only_kw', action='store_true')
    parser.add_argument('--no_rel_label', action='store_true')
    parser.add_argument('--rel_only_l2r', action='store_true')
    parser.add_argument('--num_keywords', type=int, default=4)
    parser.add_argument('--num_relatedwords', type=int, default=4)
    parser.add_argument('--edge_select', type=str, default="random", choices=['random', 'clipemb','clipemb_pretok'])


    args = parser.parse_args()
    print(args)
    print('Meshed-Memory Transformer Evaluation')
    print('path', args.features_path)
    device = torch.device(args.device)

    # load transformer numericalizer/tokenizer
    if args.tokenizer == "bert":
        tokenizerBW = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.tokenizer == "clip":
        tokenizerBW =  CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", pad_token = "[PAD]")
        tokenizerBW_dec =  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", pad_token = "[PAD]")
    else:
        print("ERROR: unrecogniezed transformer tokenizer:", args.tokenizer)


    print("size tokenizer:", len(tokenizerBW))
    allrel = ['<|Antonym|>', '<|AtLocation|>', '<|CapableOf|>', '<|Causes|>', '<|CausesDesire|>', '<|CreatedBy|>', '<|DefinedAs|>', '<|DerivedFrom|>', '<|Desires|>', '<|DistinctFrom|>', '<|Entails|>', '<|EtymologicallyDerivedFrom|>', '<|EtymologicallyRelatedTo|>', '<|FormOf|>', '<|HasA|>', '<|HasContext|>', '<|HasFirstSubevent|>', '<|HasLastSubevent|>', '<|HasPrerequisite|>', '<|HasProperty|>', '<|HasSubevent|>', '<|InstanceOf|>', '<|IsA|>', '<|LocatedNear|>', '<|MadeOf|>', '<|MannerOf|>', '<|MotivatedByGoal|>', '<|NotCapableOf|>', '<|NotDesires|>', '<|NotHasProperty|>', '<|PartOf|>', '<|ReceivesAction|>', '<|RelatedTo|>', '<|SimilarTo|>', '<|SymbolOf|>', '<|Synonym|>', '<|UsedFor|>', '<|capital|>', '<|field|>', '<|genre|>', '<|genus|>', '<|influencedBy|>', '<|knownFor|>', '<|language|>', '<|leader|>', '<|occupation|>', '<|product|>']
    allrel = [ftfy.fix_text(rel) for rel in allrel]

    tokenizerBW.add_tokens(allrel, special_tokens=True)
    if args.tokenizer == "clip":
        tokenizerBW_dec.add_tokens(allrel, special_tokens=True)

    # initialize training specifications
    cls_tok = tokenizerBW.cls_token
    spec = {}
    # do this because bert tokenizer doesn't sue bos, but cls, and sep i.s.o. eos..
    sample_txt = tokenizerBW("[PAD]").input_ids
    spec['eos_tokenid'] =  tokenizerBW.sep_token_id if cls_tok is not None else sample_txt[-1]
    spec['bos_tokenid'] =  tokenizerBW.cls_token_id if cls_tok is not None else sample_txt[0]
    spec['pad_tokenid'] = sample_txt[1]
    spec['tdqm_disable'] = False
    spec["device"] = device
    print("Selected specifications:", spec)


    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False,print_img_name=True)
    # get the 1d clip emb features
    clipemb_field = ClipEmbDetectionsField(detections_path=args.contextfeat_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(pad_token='[PAD]', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False, transform_tok = tokenizerBW, use_vocab= False)
    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder,cocoid_field= clipemb_field)
    train_dataset, val_dataset, test_dataset = dataset.splits
    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    inp_feat_size = args.feat_size
    print("size is of feats set :", args.feat_size)
    encoder = MemoryAugmentedEncoder(3, 0, d_in=inp_feat_size,  attention_module=ScaledDotProductAttention,
                                     attention_module_kwargs={'m': args.m}, dropout=args.dropout, d_model = args.d_model, h=args.head)

    seg_token = args.seg_token == "True"
    knowledge_graph = KnowledgeGraph(transform_tok = tokenizerBW, device = device, edge_select=args.edge_select, spec = spec, kw_size = args.num_keywords, rw_size = args.num_relatedwords , enc_model = args.enc_model, only_kw=args.only_kw, norel= args.no_rel_label, only_l2r = args.rel_only_l2r)

    if args.decoder == "kg_infused":
        print("using normal dec")
        decoder = PromptDecoder(len(tokenizerBW), 128, args.N_dec, spec['pad_tokenid'],h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, d_model = args.d_model)
    elif args.decoder == "parallel":
        print("using parallel dec")
        decoder = ParallelPromptDecoder(len(tokenizerBW), 128, args.N_dec, spec['pad_tokenid'], h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, d_model = args.d_model)
    elif args.decoder == "stacked":
        print("using stacked decoder")
        decoder = StackedPromptDecoder(len(tokenizerBW), 128, args.N_dec, spec['pad_tokenid'], h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, one_kw_token=args.one_kw_token, d_model = args.d_model)
    elif args.decoder == "vanilla":
       print("using vanilla decoder")
       decoder = VanillaDecoder(len(tokenizerBW), 128, args.N_dec, spec['pad_tokenid'], h=args.head, enc_model = args.enc_model, dropout=args.dropout, d_model = args.d_model)
    model = Transformer(spec['bos_tokenid'], encoder, decoder).to(device)
    model.sampling_temp = args.sampling_temp
    model.sampling_method = args.sampling_method
    # data = torch.load('meshed_memory_transformer.pth')
    # model.load_state_dict(data['state_dict'])

if args.resume == "last":
    fname = 'saved_models/%s_last.pth' % args.exp_name
else:
    fname = 'saved_models/%s_best.pth' % args.exp_name

if os.path.exists(fname):
    if device == torch.device('cpu'):
        print("on cpu")
        data = torch.load(fname,map_location=device)
    else:
        data = torch.load(fname)

    torch.set_rng_state(data['torch_rng_state'])
    torch.cuda.set_rng_state(data['cuda_rng_state'])
    np.random.set_state(data['numpy_rng_state'])
    random.setstate(data['random_rng_state'])
    model.load_state_dict(data['state_dict'], strict=False)
    # optim.load_state_dict(data['optimizer'])
    # scheduler.load_state_dict(data['scheduler'])
    start_epoch = data['epoch'] + 1
    best_cider = data['best_cider']
    patience = data['patience']
    use_rl = data['use_rl']
    print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
        data['epoch'], data['val_loss'], data['best_cider']))

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), "img_id": clipemb_field})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers,shuffle=True)

    # Sampling options: ['topk', 'beam', 'nucleus'])
    variationlist = [("nucleus", 1),("topk", 1), ("beam", 1)]
    #  ("topk", 0.2),("topk", 2)]
    for sampling_method , sampling_temp in variationlist:
        model.sampling_temp = sampling_temp
        model.sampling_method = sampling_method
        output_path = "./generated_sentences/"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        out_file = args.exp_name +"_coco_" +str(sampling_method) +"_"+str(sampling_temp)+".json"
        out_file = output_path + out_file
        print("Output path:", out_file)

        gen_captions(model, dict_dataloader_test, spec, tokenizerBW_dec, out_file)

else:
    print("path doesn't exist:", fname)

