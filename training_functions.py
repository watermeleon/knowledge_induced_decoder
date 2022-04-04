from zmq import device


def evaluate_loss(model, dataloader, loss_fn, spec, vocab_size):
    # Validation loss
    model.eval()
    running_loss = .0
    
    print("now doing eval loss")
    i = 0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader),  disable=spec['tdqm_disable']) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                contextfeat = None
                out = model(detections, captions, contextfeat)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, vocab_size), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, spec, transform_tok = None):
    model.eval()
    gen = {}
    gts = {}
    print("now doing eval metrics")

    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), disable=spec['tdqm_disable']) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            # caps_gt, context_feats = caps_gt[0], torch.stack(caps_gt[1])
            context_feats = torch.zeros_like(images).to(device)
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, context_feats, 20, spec['eos_tokenid'], 5, out_size=1)
            # print(" \n GT caps:", caps_gt)
            # out = [[word.it for word in sent if word !=0] for sent in out]
            # print("out is:", out)
            
            # if using fast tokenizer, replace <|endoftext|> token id with the [EOS] tokenid
            endtexttokid = 49407
            if endtexttokid in out:
                out1 = torch.tensor([[tokenid if tokenid != endtexttokid else 49409 for tokenid in sent ] for sent in out])
                out = out1

            caps_gen = [transform_tok.decode(sent) for sent in out] 
            # print("caps mid", caps_gen1)
            caps_gen = [sent.split("[EOS]")[0] for sent in caps_gen]

            # caps_gen = [sent.replace("[BOS]","") for sent in caps_gen]

            # print("caps_gen is now:", caps_gen)
            # print("\n \n")
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                # gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    # print("after toks:", gen, gts)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, spec, vocab_size):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    i = 0
    print("training XE")
    # profile = cProfile.Profile()
    # profile.enable()
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader),  disable=spec['tdqm_disable']) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            contextfeat = None
            out = model(detections, captions, contextfeat)
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

    #         if i >25:
    #             break
    #         i +=1
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
            outs, log_probs = model.beam_search(detections, seq_len, eos_tokenid,
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