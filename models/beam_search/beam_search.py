import torch
import utils
import torch.nn.functional as F


class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int, sampling_method:str, sampling_temp:float):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None
        self.sampling_method = sampling_method
        self.sampling_temp = sampling_temp

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    def _expand_visual(self, visual: utils.TensorOrSequence, context_feat: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        # print("inp visual; expand:", visual.size())
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)

            contfeat_shape = context_feat.shape
            contfeat_exp_shape = (self.b_s, cur_beam_size) + contfeat_shape[1:]
            contfeat_red_shape = (self.b_s * self.beam_size,) + contfeat_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(contfeat_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + contfeat_exp_shape[2:]
            contfeat_exp = context_feat.view(contfeat_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            context_feat = torch.gather(contfeat_exp, 1, selected_beam_exp).view(contfeat_red_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual , context_feat

    def apply(self, visual: utils.TensorOrSequence, contextfeat: utils.TensorOrSequence, out_size=1, return_probs=False, **kwargs):
        self.b_s = utils.get_batch_size(visual)
        self.device = utils.get_device(visual)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                visual,contextfeat, outputs = self.iter(t, visual,contextfeat, outputs, return_probs, **kwargs)

        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                          self.max_len,
                                                                                          all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True) # [bs, 1, vocab_size]
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob

    def select_sample(self,t, candidate_logprob, **kwargs):
        # use original select
        if self.sampling_method == "beam":
            # print("using beam")
            return self.select(t, candidate_logprob,  **kwargs)
        # print("doign whatev, royally fucked", self.sampling_method)
        candlog = candidate_logprob.view(self.b_s, -1)
        # samp_probs = F.softmax(candlog, dim=-1)
        samp_probs = candlog.exp()
        samp_probs_orig = samp_probs.clone()
        if self.sampling_temp != 1:
            # sharpens peaks by lower temp
            samp_probs = F.softmax(candlog.div_(self.sampling_temp), dim=-1)

        if self.sampling_method == "topk":
            # use top-k sampling , sample k**2 = 25 beams, keep 5
            k = self.beam_size ** 2
            indices_to_remove = samp_probs < torch.topk(samp_probs, k)[0][..., -1, None]
            samp_probs[indices_to_remove] = 0
            selected_idx = samp_probs.multinomial(self.beam_size)
            selected_logprob = samp_probs_orig.gather(1, selected_idx)
        elif self.sampling_method == "nucleus":
            # use nucleus sampling with fixed p
            p = 0.9 
            sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
            sorted_orig_probs, _ = torch.sort(samp_probs, descending=True)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            sorted_samp_probs = sorted_probs.clone()
            sorted_samp_probs[sorted_indices_to_remove] = 0

            sorted_next_indices = sorted_samp_probs.multinomial(self.beam_size)
            selected_idx = sorted_indices.gather(1, sorted_next_indices)
            selected_logprob = sorted_orig_probs.gather(1, sorted_next_indices).log()
        return selected_idx, selected_logprob

    def iter(self, t: int, visual: utils.TensorOrSequence, contextfeat: utils.TensorOrSequence, outputs, return_probs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size
        word_logprob = self.model.step(t, self.selected_words, visual, None, mode='feedback', contextfeat = contextfeat, **kwargs)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)

        selected_idx, selected_logprob = self.select_sample(t, candidate_logprob, **kwargs)
        selected_beam = selected_idx  / candidate_logprob.shape[-1]
        error_marg = 1/(4*candidate_logprob.shape[-1])
        selected_beam += error_marg
        selected_beam = selected_beam.type(torch.int64)

        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
        visual, contextfeat = self._expand_visual(visual,contextfeat, cur_beam_size, selected_beam)

        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        outputs.append(selected_words.unsqueeze(-1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))

        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        return visual,contextfeat, outputs
