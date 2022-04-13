
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ast import arg
import random
from data import ImageDetectionsField, TextField, RawField, ClipEmbDetectionsField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, MultiLevelEncoder, ScaledDotProductAttention, VanillaDecoder
from knowgraph_conceptnet import KnowledgeGraph

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from torch import autograd
from transformers import AutoTokenizer, CLIPTokenizer, CLIPTokenizerFast
# from torch.utils.data import IterableDataset  
import cProfile
import pstats 
import ftfy

# import training_functions 
seed_num = 1234
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


exec(open("training_functions.py").read())


if __name__ == '__main__':
    device = torch.device('cuda')

    # device = torch.device('cpu')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--contextfeat_path', type=str)
    parser.add_argument('--onlisa', type=str, default="True", choices=['True', 'False'])
    parser.add_argument('--seg_token', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--edge_select', type=str, default="random", choices=['random', 'clipemb','clipemb_pretok'])
    parser.add_argument('--decoder', type=str, default="kg_infused", choices=['vanilla', 'kg_infused'])
    parser.add_argument('--tokenizer', type=str, default="bert", choices=['bert', 'clip'])
    parser.add_argument('--enc_model', type=str, default="ViT", choices=['ViT', 'rn50x4'])
    parser.add_argument('--pt_token_emb', action='store_true')

    parser.add_argument('--d_att', type=int, default=64)

    parser.add_argument('--num_keywords', type=int, default=4)
    parser.add_argument('--num_relatedwords', type=int, default=4)


    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--feat_size', type=int, default=2048)

    args = parser.parse_args()
    print(args)

    print('KG context Transformer Training')



    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

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
    # spec['pad_tokenid'] = tokenizerBW.pad_token_id
    spec['pad_tokenid'] = sample_txt[1]

    spec['tdqm_disable'] = False
    spec["device"] = device
    print("Selected specifications:", spec)

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
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
    elif args.decoder == "vanilla":
        decoder = VanillaDecoder(len(tokenizerBW), 128, 3, spec['pad_tokenid'], d_k=args.d_att, d_v=args.d_att, enc_model = args.enc_model)

    model = Transformer(spec['bos_tokenid'], encoder, decoder).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField(), "img_id": clipemb_field})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField(), "img_id": clipemb_field})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), "img_id": clipemb_field})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=spec['pad_tokenid'])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
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
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))


    ########################################################################################################################################################################
    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=False,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        # scores = evaluate_metrics(model, dict_dataloader_val, spec, transform_tok = tokenizerBW_dec)
        # print("these scores be all like:", scores)
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, spec, len(tokenizerBW))
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, spec)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, spec, len(tokenizerBW))
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, spec, transform_tok = tokenizerBW_dec)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, spec, transform_tok = tokenizerBW_dec)
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break
