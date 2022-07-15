

from zmq import device


def evaluate_loss(model, dataloader, loss_fn, spec, vocab_size):
    # Validation loss
    model.eval()
    running_loss = .0
    print("now doing eval loss")
    i = 0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader),  disable=spec['tdqm_disable']) as pbar:
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

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics_standard(model, dataloader, spec, transform_tok = None):
    model.eval()
    gen = {}
    gts = {}
    seq_len = 20

    print("now doing eval metrics")
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), disable=spec['tdqm_disable']) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            caps_gt, context_feats = caps_gt[0], torch.stack(caps_gt[1])
            context_feats = context_feats[:,0,:,:]
            images, context_feats = images.to(device), context_feats.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, context_feats, seq_len, spec['eos_tokenid'], 5, out_size=1)

            caps_gen = [transform_tok.decode(sent) for sent in out] 
            caps_gen = [sent.split("<|endoftext|>")[0] for sent in caps_gen]

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def evaluate_metrics_gpt2(model, dataloader, spec, transform_tok = None):
    model.eval()
    gen = {}
    gts = {}

    print("now doing eval metrics")
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), disable=spec['tdqm_disable']) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            caps_gt, context_feats = caps_gt[0], torch.stack(caps_gt[1])
            context_feats = context_feats[:,0,:,:]
            images, context_feats = images.to(device), context_feats.to(device)
            caps_gen = []
            with torch.no_grad():
                enc_output, mask_enc = model.encoder(images)
                prefix_embed = model.decoder(torch.ones((len(caps_gt), 6), dtype=int, device=device).to(device), enc_output, mask_enc, context_feats, gen_sent=True)
                for prefix_i in prefix_embed:
                    out = generate2(model.decoder, transform_tok, embed=prefix_i[None,:])
                    caps_gen.append(out)

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()


    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, spec, vocab_size):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    i = 0
    print("training XE")

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader),  disable=spec['tdqm_disable']) as pbar:
        for it, (detections, captions, context_feats) in enumerate(dataloader):
            detections, captions, context_feats = detections.to(device), captions.to(device), context_feats.to(device)
            out = model(detections, captions, context_feats)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()

            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, vocab_size), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()
            # if it > 1000:
            #     break
    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, spec, transform_tok):
    # Training with self-critical
    set_start_method('forkserver')
    tokenizer_pool = Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5
    print("trainin SCTS")
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader),  disable=spec['tdqm_disable']) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            caps_gt, context_feats = caps_gt[0], torch.stack(caps_gt[1])
            context_feats = context_feats[:,0,:,:]
            detections, context_feats = detections.to(device) , context_feats.to(device)
            outs, log_probs = model.beam_search(detections, context_feats, seq_len, spec["eos_tokenid"],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen = [transform_tok.decode(sent) for sent in outs.view(-1, seq_len)] 
            caps_gen = [sent.split("<|endoftext|>")[0] for sent in caps_gen] 
            # total its:14161
            # if it == 2265:
            #     print("\n",caps_gen, "\n")
            #     break
            # this puts it in lists or something:
            # tokenizer_pool = Pool()
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])

            # caps_gt = evaluation.PTBTokenizer.tokenize(caps_gt)
            # caps_gen = evaluation.PTBTokenizer.tokenize(caps_gen)

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
    tokenizer_pool.close()
    tokenizer_pool.join()
    return loss, reward, reward_baseline
