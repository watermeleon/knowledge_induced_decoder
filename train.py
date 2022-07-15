
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ast import arg
import random
from data import ImageDetectionsField, TextField, RawField, ClipEmbDetectionsField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, PromptDecoder, ScaledDotProductAttentionMemory, MultiLevelEncoder, ScaledDotProductAttention, VanillaDecoder, ParallelPromptDecoder , StackedPromptDecoder
from knowgraph_conceptnet import KnowledgeGraph

import torch
from torch.optim import Adam , AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
# import multiprocessing
from multiprocessing import set_start_method, Pool
from shutil import copyfile
from torch import autograd
from transformers import AutoTokenizer, CLIPTokenizer, CLIPTokenizerFast, GPT2TokenizerFast, GPT2Tokenizer
# from torch.utils.data import IterableDataset  

from models.beam_search.gpt2_generation import generate_beam , generate2

import cProfile
import pstats 
import ftfy
import wandb
# import training_functions 
seed_num = 1234
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

set_start_method('forkserver')
exec(open("training_functions.py").read())



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PromptDecoder - KG- Transformer')

    # training basics
    parser.add_argument('--exp_name', type=str, default='kg_prompt_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
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
    parser.add_argument('--d_model', type=int, default=512)


    parser.add_argument('--seg_token', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--seg_token_kw', action='store_true')
    parser.add_argument('--seg_param', action='store_true')
    parser.add_argument('--decoder', type=str, default="kg_infused", choices=['vanilla', 'kg_infused', 'parallel', 'stacked'])
    parser.add_argument('--pll_dec', type=int, default=1)
    parser.add_argument('--stck_gpt2', action='store_true')

    parser.add_argument('--one_kw_token', action='store_true') # for the stackeddecoder
    parser.add_argument('--tf_model_conf', type=str, default="alt", choices=['alt', 'base', 'tiny']) # if not alt, overwrites other head and dmodel param

    # training specifics
    parser.add_argument('--start_rl', action='store_true')
    parser.add_argument('--no_rl', action='store_true')
    parser.add_argument('--tokenizer', type=str, default="bert", choices=['bert', 'clip', 'gpt2'])
    parser.add_argument('--pt_token_emb', action='store_true') # for the KG part of the decoder
    parser.add_argument('--optimizer', type=str, default="adam", choices=['adam', 'adamW'])

    # knowledge graph related
    parser.add_argument('--only_kw', action='store_true')
    parser.add_argument('--no_rel_label', action='store_true')
    parser.add_argument('--rel_only_l2r', action='store_true')
    parser.add_argument('--num_keywords', type=int, default=4)
    parser.add_argument('--num_relatedwords', type=int, default=4)
    parser.add_argument('--edge_select', type=str, default="random", choices=['random', 'clipemb','clipemb_pretok'])
    parser.add_argument('--use_faiss', action='store_true')
    parser.add_argument('--rc_posidx2', action='store_true')
    parser.add_argument('--cn_version', type=str, default="",  choices=["", "noname",'imgnet1k'])


    args = parser.parse_args()
    print(args)

    print('KG context Transformer Training')
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

    wandb.init(project="hyptuning3" ,name=args.exp_name, entity="watermelontology")
    wandb.config.update(args)
    print(wandb.config)

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    tokenizerBW_clip =  CLIPTokenizerFast.from_pretrained("./models/tokenizers_stored/CLIPTokenizerFast")

    # load transformer numericalizer/tokenizer
    if args.tokenizer == "bert":
        tokenizerBW = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.tokenizer == "clip":
        tokenizerBW =  CLIPTokenizerFast.from_pretrained("./models/tokenizers_stored/CLIPTokenizerFast")
        tokenizerBW_dec =  CLIPTokenizer.from_pretrained("./models/tokenizers_stored/CLIPTokenizer")
    elif args.tokenizer == "gpt2":
        tokenizerBW =  GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizerBW_dec =  GPT2Tokenizer.from_pretrained("gpt2")
    else:
        print("ERROR: unrecogniezed transformer tokenizer:", args.tokenizer)

    print("size tokenizer:", len(tokenizerBW))

    # initialize training specifications
    cls_tok = tokenizerBW.cls_token
    spec = {}
    # do this because bert tokenizer doesn't sue bos, but cls, and sep i.s.o. eos..
    if args.tokenizer == "clip":
        spec['pad_token'] = '[PAD]'
        sample_txt = tokenizerBW(spec['pad_token'] ).input_ids
        spec['eos_tokenid'] =  tokenizerBW.sep_token_id if cls_tok is not None else sample_txt[-1]
        spec['bos_tokenid'] =  tokenizerBW.cls_token_id if cls_tok is not None else sample_txt[0]
        spec['pad_tokenid'] = sample_txt[1]
    else:
        stop_token = "."
        spec['eos_tokenid'] = tokenizerBW.encode(stop_token)[0]
        spec['bos_tokenid'] = 0
        spec['pad_tokenid'] = 0
        spec['pad_token'] = tokenizerBW.decode(spec['pad_tokenid'])

    spec['tdqm_disable'] = False
    spec["device"] = device
    print("Selected specifications:", spec)

    pad_token_id = 0
    if args.tokenizer == "clip":
        pad_token_id = tokenizerBW.encode(tokenizerBW.pad_token)[1]
    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
    # get the 1d clip emb features
    clipemb_field = ClipEmbDetectionsField(detections_path=args.contextfeat_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(pad_token=spec['pad_token'], lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False, transform_tok = tokenizerBW, use_vocab= False, pad_token_id=pad_token_id)
    # Create the dataset
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

    knowledge_graph = KnowledgeGraph(transform_tok = tokenizerBW_clip, device = device, edge_select=args.edge_select, spec = spec, kw_size = args.num_keywords, rw_size = args.num_relatedwords , enc_model = args.enc_model, only_kw=args.only_kw, norel= args.no_rel_label, only_l2r = args.rel_only_l2r, use_faiss = args.use_faiss, rc_posidx2 =args.rc_posidx2, cn_version=args.cn_version)

    max_inp_seq= 128
    if args.decoder == "kg_infused":
        print("using normal dec")
        decoder = PromptDecoder(len(tokenizerBW), max_inp_seq, args.N_dec, spec['pad_tokenid'],h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, d_model = args.d_model, seg_param=args.seg_param)
    elif args.decoder == "parallel":
        print("using parallel dec")
        decoder = ParallelPromptDecoder(len(tokenizerBW), max_inp_seq, args.N_dec, spec['pad_tokenid'], h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, d_model = args.d_model, pll_dec_type = args.pll_dec, seg_token_kw = args.seg_token_kw, seg_param=args.seg_param)
    elif args.decoder == "stacked":
        print("using stacked decoder")
        decoder = StackedPromptDecoder(len(tokenizerBW), max_inp_seq, args.N_dec, spec['pad_tokenid'], h=args.head, seg_token= seg_token, KG = knowledge_graph , enc_model= args.enc_model, spec=spec, pt_tokemb=args.pt_token_emb, dropout=args.dropout, one_kw_token=args.one_kw_token, d_model = args.d_model, seg_token_kw = args.seg_token_kw, use_gpt=args.stck_gpt2, seg_param=args.seg_param)
    elif args.decoder == "vanilla":
       print("using vanilla decoder")
       decoder = VanillaDecoder(len(tokenizerBW), max_inp_seq, args.N_dec, spec['pad_tokenid'], h=args.head, enc_model = args.enc_model, dropout=args.dropout, d_model = args.d_model, seg_token_kw = args.seg_token_kw, seg_param=args.seg_param)
 
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
    if args.optimizer == "adamW":
        optim = AdamW(model.parameters(), lr=1, betas=(0.9, 0.98))
    else:
        optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))

    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=spec['pad_tokenid'])
    # changed by leon
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
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    if args.start_rl:
        use_rl = True
        patience = 0
        optim = Adam(model.parameters(), lr=5e-6)
        print("Switching to RL")


    if args.decoder == "stacked" and args.stck_gpt2:
        evaluate_metrics = evaluate_metrics_gpt2
    else:
        evaluate_metrics = evaluate_metrics_standard

    ########################################################################################################################################################################
    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=max(2, args.batch_size // 6), shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        # scores = evaluate_metrics(model, dict_dataloader_val, spec, transform_tok = tokenizerBW_dec)
        # print("these scores be all like:", scores)
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, spec, len(tokenizerBW))
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, spec, tokenizerBW_dec)
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
        
        val_scores = scores.copy()
        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, spec, transform_tok = tokenizerBW_dec)
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        test_scores = scores.copy()

        # log weights and biases results:    
        ep_metrics = {"epoch": e ,"train_loss": train_loss,"val_loss":val_loss, "val_cider": val_scores['CIDEr'],"val_bleu1":val_scores['BLEU'][0],
                         "val_bleu4":val_scores['BLEU'][3], "val_meteor": val_scores['METEOR'], "val_rouge": val_scores['ROUGE'], 
                         "test_cider": test_scores['CIDEr'],"test_bleu1":test_scores['BLEU'][0],
                         "test_bleu4":test_scores['BLEU'][3], "test_meteor": test_scores['METEOR'], "test_rouge": test_scores['ROUGE']
                         
                         }
        if use_rl:
            ep_metrics["reward"] = reward
            ep_metrics["reward_baseline"] = reward_baseline
        wandb.log(ep_metrics)
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

        if (switch_to_rl and args.no_rl )or exit_train:
            writer.close()
            break
