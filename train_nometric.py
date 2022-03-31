
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ast import arg
import random
from data import ImageDetectionsField, TextField, RawField, ClipEmbDetectionsField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, MultiLevelEncoder, ScaledDotProductAttention
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
from tokenizers import BertWordPieceTokenizer
from torch import autograd
from transformers import AutoTokenizer
# from torch.utils.data import IterableDataset  
import cProfile
import pstats 

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)



def evaluate_loss(model, dataloader, loss_fn, text_field, vocab_size):
    # Validation loss
    model.eval()
    running_loss = .0
    print("now doing eval loss")
    i = 0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader),  disable=False) as pbar:
        with torch.no_grad():
            for it, (detections, captions, context_feats) in enumerate(dataloader):
                detections, captions, context_feats = detections.to(device), captions.to(device), context_feats.to(device)
                out = model(detections, captions, context_feats)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, vocab_size), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                # break
                # if i >25:
                #     break
                # i +=1
    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, transform_tok = None):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    print("now doing eval metrics")
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), disable=False) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            caps_gt, context_feats = caps_gt[0], torch.stack(caps_gt[1])
            images, context_feats = images.to(device), context_feats.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, context_feats, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            # caps_gen = text_field.decode(out, join_words=False)
            caps_gen = [transform_tok.decode(sent) for sent in out] 
            # print("capsgen:", caps_gen)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field, vocab_size):
    # Training with cross-entropy
    model.train()
    # scheduler.step()
    running_loss = .0
    i = 0
    print("training XE")
    # profile = cProfile.Profile()
    # profile.enable()
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader),  disable=False) as pbar:
        for it, (detections, captions, context_feats) in enumerate(dataloader):
            # print("detections is:", detections.size())
            # print("captions:", captions, captions.size(), "cocoid:", context_feats.size(), context_feats)
            detections, captions, context_feats = detections.to(device), captions.to(device), context_feats.to(device)
            # print("det ofxe:", detections.size(), captions.size())

            out = model(detections, captions, context_feats)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            # print("output of dec is:", out, out.size())
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, vocab_size), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()
            # break
            # if i >1000:
            #     break
            i +=1
    # profile.disable()
    # ps = pstats.Stats(profile)
    # ps.print_stats()
    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5
    print("trainin SCTS")
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader),  disable=True) as pbar:
        for it, (detections, caps_gt, cocoid) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


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
    parser.add_argument('--onlisa', type=str, default="True")

    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--feat_size', type=int, default=2048)

    args = parser.parse_args()
    print(args)

    print('KG context Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # load transformer numericalizer/tokenizer
    tokenizerBW = AutoTokenizer.from_pretrained("bert-base-uncased")
    allrel = ['Antonym', 'AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy', 'DefinedAs', 'DerivedFrom', 'Desires', 'DistinctFrom', 'Entails', 'EtymologicallyDerivedFrom', 'EtymologicallyRelatedTo', 'FormOf', 'HasA', 'HasContext', 'HasFirstSubevent', 'HasLastSubevent', 'HasPrerequisite', 'HasProperty', 'HasSubevent', 'InstanceOf', 'IsA', 'LocatedNear', 'MadeOf', 'MannerOf', 'MotivatedByGoal', 'NotCapableOf', 'NotDesires', 'NotHasProperty', 'PartOf', 'ReceivesAction', 'RelatedTo', 'SimilarTo', 'SymbolOf', 'Synonym', 'UsedFor', 'capital', 'field', 'genre', 'genus', 'influencedBy', 'knownFor', 'language', 'leader', 'occupation', 'product', "[IMG]"]
    tokenizerBW.add_tokens(allrel, special_tokens=True)

    # tokenizerBW = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
    # get the 1d clip emb features
    clipemb_field = ClipEmbDetectionsField(detections_path=args.contextfeat_path, load_in_tmp=False)

    # Pipeline for text
    # text_field = TextField(init_token='[SOS]', eos_token='[EOS]', pad_token='[PAD]', lower=True, tokenize='spacy',
    #                        remove_punctuation=True, nopoints=False, transform_tok = tokenizerBW, use_vocab= False)
    text_field = TextField(pad_token='[PAD]', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False, transform_tok = tokenizerBW, use_vocab= False)
    # cocoid_field = RawField()
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
    # encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
    #                                  attention_module_kwargs={'m': args.m})
    inp_feat_size = args.feat_size
    print("size is of feats set :", args.feat_size)
    encoder = MemoryAugmentedEncoder(3, 0, d_in=inp_feat_size,  attention_module=ScaledDotProductAttention,
                                     attention_module_kwargs={'m': args.m})
    onlisa = args.onlisa == "True"
    decoder = MeshedDecoder(len(tokenizerBW), 128, 3, text_field.vocab.stoi['<pad>'], d_k=128, d_v=128, transform_tok = tokenizerBW, device = device, on_lisa=onlisa)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

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
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
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

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=False,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        # scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        # scores = evaluate_metrics(model, dict_dataloader_val, text_field, transform_tok = tokenizerBW)
        # print("these scores be all like:", scores)
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field, len(tokenizerBW))
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, len(tokenizerBW))
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        # scores = evaluate_metrics(model, dict_dataloader_val, text_field, transform_tok = tokenizerBW)
        # print("Validation scores", scores)
        # val_cider = scores['CIDEr']
        # writer.add_scalar('data/val_cider', val_cider, e)
        # writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        # writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        # writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        # writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        # scores = evaluate_metrics(model, dict_dataloader_test, text_field, transform_tok = tokenizerBW)
        # print("Test scores", scores)
        # writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        # writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        # writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        # writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        # writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        # best = False
        # if val_cider >= best_cider:
        #     best_cider = val_cider
        val_cider = 42
        best_cider = val_cider
        patience = 0
        best = True
        # else:
        #     patience += 1

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
