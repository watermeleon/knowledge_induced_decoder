import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from data import ImageDetectionsField, TextField, RawField, ClipEmbDetectionsField
from data import COCO, DataLoader, NoCaps
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

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, spec, transform_tok):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images, img_ids = images
            # print("image ids :")
            caps_gt, context_feats = caps_gt[0], torch.stack(caps_gt[1])
            context_feats = context_feats[:,0,:,:]
            images, context_feats = images.to(device), context_feats.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, context_feats, 20, spec['eos_tokenid'], 5, out_size=1)

            caps_gen = [transform_tok.decode(sent) for sent in out] 
            caps_gen = [sent.split("<|endoftext|>")[0] for sent in caps_gen]
            # print("capsgen:", caps_gen)
            caps_gt = [tuple([ftfy.fix_text(sent) for sent in img_batch]) for img_batch in caps_gt]


            # caps_gt = [tuple([sent.encode('latin-1').decode('utf-8') for sent in img_batch]) for img_batch in caps_gt]
            # print("caps GT:", caps_gt, "\n")
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


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
    parser.add_argument('--enc_model', type=str, default="ViT-B/32", choices=['ViT-B/32', 'rn50x4'])    
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--seg_token', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--seg_token_kw', action='store_true')

    parser.add_argument('--decoder', type=str, default="kg_infused", choices=['vanilla', 'kg_infused', 'parallel', 'stacked'])
    parser.add_argument('--one_kw_token', action='store_true') # for the stackeddecoder
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--tf_model_conf', type=str, default="alt", choices=['alt', 'base', 'tiny']) # if not alt, overwrites other head and dmodel param
    parser.add_argument('--pll_dec', type=int, default=1)

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
    parser.add_argument('--use_faiss', action='store_true')
    parser.add_argument('--rc_posidx2', action='store_true')
    parser.add_argument('--nocaps', action='store_true')

    args = parser.parse_args()
    print(args)
    print('Meshed-Memory Transformer Evaluation')
    print('path', args.features_path)
    device = torch.device(args.device)
    new_enc_model = args.enc_model.replace('/', '_')
    args.enc_model = new_enc_model

    if args.tf_model_conf != "alt":
        if args.tf_model_conf == "base":
            args.d_model = 512
            args.head = 8
        if args.tf_model_conf == "tiny":
            args.d_model = 384
            args.head = 6


  # load transformer numericalizer/tokenizer
    if args.tokenizer == "bert":
        tokenizerBW = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.tokenizer == "clip":
        tokenizerBW =  CLIPTokenizerFast.from_pretrained("./models/tokenizers_stored/CLIPTokenizerFast")
        tokenizerBW_dec =  CLIPTokenizer.from_pretrained("./models/tokenizers_stored/CLIPTokenizer")
    else:
        print("ERROR: unrecogniezed transformer tokenizer:", args.tokenizer)

    print("size tokenizer:", len(tokenizerBW))
    

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

    pad_token_id = 0
    if args.tokenizer == "clip":
        pad_token_id = tokenizerBW.encode(tokenizerBW.pad_token)[1]

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False,print_img_name=True)
    # get the 1d clip emb features
    clipemb_field = ClipEmbDetectionsField(detections_path=args.contextfeat_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(pad_token='[PAD]', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False, transform_tok = tokenizerBW, use_vocab= False, pad_token_id=pad_token_id)
    
     # Create the dataset
    if args.nocaps:
        dataset = NoCaps(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder,cocoid_field= clipemb_field)
        train_dataset, val_dataset, test_dataset = dataset.splits
        #big time cheating:
        test_dataset = val_dataset
    else:
        dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder,cocoid_field= clipemb_field)
        train_dataset, val_dataset, test_dataset = dataset.splits    
    
    baseline_vocab = "vocab_coco_baseline_vocab.pkl"
    if not os.path.isfile(baseline_vocab):
        print("Building vocabulary: ERROR this shouldn't be happening")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open(baseline_vocab, 'rb'))

    # Model and dataloaders
    inp_feat_size = args.feat_size
    print("size is of feats set :", args.feat_size)
    encoder = MemoryAugmentedEncoder(args.N_enc, 0, d_in=inp_feat_size,  attention_module=ScaledDotProductAttention,
                                     attention_module_kwargs={'m': args.m}, dropout=args.dropout, d_model = args.d_model, h=args.head)

    seg_token = args.seg_token == "True"
    knowledge_graph = KnowledgeGraph(transform_tok = tokenizerBW, device = device, edge_select=args.edge_select, spec = spec, kw_size = args.num_keywords, rw_size = args.num_relatedwords , enc_model = args.enc_model, only_kw=args.only_kw, norel= args.no_rel_label, only_l2r = args.rel_only_l2r, use_faiss = args.use_faiss, rc_posidx2 =args.rc_posidx2)

    if args.decoder == "kg_infused":
        print("using normal dec")
        decoder = PromptDecoder(len(tokenizerBW), 128, args.N_dec, spec['pad_tokenid'],h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, d_model = args.d_model)
    elif args.decoder == "parallel":
        print("using parallel dec")
        decoder = ParallelPromptDecoder(len(tokenizerBW), 128, args.N_dec, spec['pad_tokenid'], h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, d_model = args.d_model, pll_dec_type = args.pll_dec, seg_token_kw = args.seg_token_kw)
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

    scores = predict_captions(model, dict_dataloader_test, spec, tokenizerBW_dec)
    print(scores)
