import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from data import ImageDetectionsField, TextField, RawField, ClipEmbDetectionsField
from data import COCO, DataLoader
import evaluation
# from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, MultiLevelEncoder, ScaledDotProductAttention, VanillaDecoder, PromptDecoder
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
            # images = images.to(device)
            caps_gt, context_feats = caps_gt[0], torch.stack(caps_gt[1])
            context_feats = context_feats[:,0,:,:]
            images, context_feats = images.to(device), context_feats.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, context_feats, 20, spec['eos_tokenid'], 5, out_size=1)

            caps_gen = [transform_tok.decode(sent) for sent in out] 
            caps_gen = [sent.split("<|endoftext|>")[0] for sent in caps_gen]
            print("capsgen:", caps_gen)
            print("caps GT:", caps_gt, "\n")
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')

    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)

    parser.add_argument('--contextfeat_path', type=str)
    parser.add_argument('--onlisa', type=str, default="True", choices=['True', 'False'])
    parser.add_argument('--seg_token', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--edge_select', type=str, default="random", choices=['random', 'clipemb','clipemb_pretok'])
    parser.add_argument('--decoder', type=str, default="kg_infused", choices=['vanilla', 'kg_infused', 'prompt_decoder'])
    parser.add_argument('--tokenizer', type=str, default="bert", choices=['bert', 'clip'])
    parser.add_argument('--enc_model', type=str, default="ViT", choices=['ViT', 'rn50x4'])
    parser.add_argument('--num_keywords', type=int, default=5)
    parser.add_argument('--num_relatedwords', type=int, default=5)
    parser.add_argument('--feat_size', type=int, default=2048)
    parser.add_argument('--resume', type=str, default="best", choices=['best', 'last'])
    parser.add_argument('--d_att', type=int, default=64)
    parser.add_argument('--pt_token_emb', action='store_true')

    parser.add_argument('--sampling_method', type=str, default="nucleus", choices=['topk', 'beam', 'nucleus'])
    parser.add_argument('--sampling_temp', type=float, default=1)


    args = parser.parse_args()
    print(args)
    print('Meshed-Memory Transformer Evaluation')
    print('path', args.features_path)


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
                                     attention_module_kwargs={'m': args.m})

    onlisa = args.onlisa == "True"
    seg_token = args.seg_token == "True"
    knowledge_graph = KnowledgeGraph(transform_tok = tokenizerBW, device = device, on_lisa = onlisa, edge_select=args.edge_select, spec = spec, kw_size = args.num_keywords, rw_size = args.num_relatedwords , enc_model = args.enc_model)

    if args.decoder == "kg_infused":
        decoder = MeshedDecoder(len(tokenizerBW), 128, 3, spec['pad_tokenid'], d_k=args.d_att, d_v=args.d_att, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb)
    elif args.decoder == "prompt_decoder":
        decoder = PromptDecoder(len(tokenizerBW), 128, 3, spec['pad_tokenid'], d_k=args.d_att, d_v=args.d_att, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb)
    elif args.decoder == "vanilla":
        decoder = VanillaDecoder(len(tokenizerBW), 128, 3, spec['pad_tokenid'], d_k=args.d_att, d_v=args.d_att, enc_model = args.enc_model)

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
