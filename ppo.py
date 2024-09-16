# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02-ppo.ipynb (unless otherwise specified).

__all__ = ['AdaptiveKLController', 'FixedKLController', 'PPOTrainer']

# Cell
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
import random
import os
import math
from tqdm import tqdm

from load_finetuned_model import load_model_and_tokenizer
from modeling_value_head import AutoModelForCausalLMWithValueHead

from core import (logprobs_from_logits,
                         whiten,
                         clip_by_value,
                         entropy_from_logits,
                         flatten_dict,
                         average_torch_dicts,
                         stats_to_np,
                         stack_dicts,
                         add_suffix, 
                         build_bert_batch_from_txt)


# Cell

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

# Cell

class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to tune a language model.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 64,
        "forward_batch_size": 8,
        "ppo_epochs": 4,
    }

    def __init__(self, model, ref_model, **ppo_params):
        """
        Initialize PPOTrainer.

        Args:
            model (torch.model): Huggingface GPT2 model
            ref_model (torch.model): Huggingface GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000

        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        self.ref_model = ref_model
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])

        # learning rate scheduler
        self.epochs = self.ppo_params["epochs"] / self.ppo_params["batch_size"]
        self.max_steps = int(self.epochs * self.ppo_params["ppo_epochs"] * self.ppo_params["batch_size"])
        self.warmup_steps = int(0.1 * self.max_steps)
        self.max_lr = self.ppo_params['lr']
        self.min_lr = 0.1 * self.ppo_params['lr']
        self.current_step = 0

        self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                           self.ppo_params['target'],
                                           self.ppo_params['horizon'])
        

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it+1) /  self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it -  self.warmup_steps) / (self.max_steps -  self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


    def step(self, query, response, scores):
        """
        Run a PPO optimisation step.

        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.ppo_params['batch_size']
        timing = dict()
        t0 = time.time()

        gen_len = response.shape[1]
        model_input = torch.cat((query, response), axis=1)

        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(model_input, gen_len)
        timing['time/ppo/forward_pass'] = time.time()-t

        t = time.time()
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time()-t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        # off policy learning 
        for _ in range(self.ppo_params['ppo_epochs']):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                train_stats = self.train_minibatch(logprobs[idx:idx+1], values[idx:idx+1],
                                                   rewards[idx:idx+1], query[idx:idx+1],
                                                   response[idx:idx+1], model_input[idx:idx+1], step=self.current_step)
                all_stats.append(train_stats)
                self.current_step += 1
        timing['time/ppo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0
        stats.update(timing)
        return stats

    def batched_forward_pass(self, model_input, gen_len):
        """Calculate model outputs in multiple batches.
        
        In PPO, we update the policy π_θ. The logprobs output represents log(π_θ(a_t|s_t)), which is crucial for computing the probability ratio:
            r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            In log space, this becomes:
            log(r_t(θ)) = log(π_θ(a_t|s_t)) - log(π_θ_old(a_t|s_t))

        Value Function (V_θ):
            The values output represents V_θ(s_t), which is used in computing advantages:
            A_t = δ_t + (γλ)δ_(t+1) + ... + (γλ)^(T-t+1)δ_(T-1)
            where δ_t = r_t + γV(s_(t+1)) - V(s_t)

        The ref_logprobs are used to compute the KL divergence between the current policy and the reference policy:
            KL(π_ref || π_θ) = E_s[Σ_a π_ref(a|s) * (log π_ref(a|s) - log π_θ(a|s))]
            This is used as a soft constraint to prevent the policy from deviating too much from the reference policy.
                """
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        logprobs = []
        ref_logprobs = []
        values = []

        for i in range(int(self.ppo_params['batch_size']/fbs)):
            m_input = model_input[i*fbs:(i+1)*fbs]
            logits, _, v = self.model(m_input)
            ref_logits, _, _ = self.ref_model(m_input)

            values.append(v[:, -gen_len-1:-1].detach()) #  By selecting -gen_len-1:-1, we ensure that we’re only considering the value predictions for the generated tokens and not for future tokens that haven’t been generated yet.
            logprobs.append(logprobs_from_logits(logits[:,:-1,:], m_input[:,1:])[:, -gen_len:].detach()) # we are interested in the log probability of the actual tokens that were generated or observed, not the predictions for tokens that haven’t been generated yet.
            ref_logprobs.append(logprobs_from_logits(ref_logits[:,:-1,:], m_input[:,1:])[:, -gen_len:].detach()) # m_input[:, 1:] is used to shift the input tokens to align with the target tokens that the model should predict.

        return torch.cat(logprobs), torch.cat(ref_logprobs), torch.cat(values)

    def train_minibatch(self, logprobs, values, rewards, query, response, model_input, step):
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats  = self.loss(logprobs, values, rewards, query, response, model_input)
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        # lr = self.get_lr(step)
        # print(f"learning rate {lr} at step {step}")
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr
        self.optimizer.step()
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty.
        Per-token Rewards:
        The non_score_reward provides a per-token penalty based on KL divergence. 
        This is a form of dense reward, giving feedback for each action (token) in the sequence.
        
        """
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl # tune this into a penalty 
        rewards = non_score_reward.clone().detach()
        rewards[:, -1] += scores # add the score to the last token because we compute advatage starting from last token going backwards
        return rewards, non_score_reward, self.kl_ctl.value

    def loss(self, old_logprobs, values, rewards, query, response, model_input):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        for t in reversed(range(gen_len)): # boostrapping 
            # estimating the advantage term
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t] # Measure the difference between what we expected and what actually happened.
            # total expected return from taking the chosen action
            # values[:, t] represents the average expected return from being in the current state, regardless of which action is taken. This is essentially what we'd expect if we chose actions randomly.
            # By subtracting values[:, t] from the total expected return, we're calculating how much better (or worse) the chosen action is compared to this average expectation.
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            # It allows advantages from future time steps to influence the advantage estimate of earlier time steps.
            # We add to this a discounted (gamma) version of our previous estimate (lastgaelam).
            # The 'lam' (lambda) parameter lets us control how much we trust future estimates vs. current feedback.
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        logits, _, vpred = self.model(model_input)
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
            logp=dict(logp=logprob, logp_old=old_logprobs, logp_diff=logprob-old_logprobs)
        )
        print(f"pg_loss {pg_loss} | vf_loss {self.ppo_params['vf_coef'] * vf_loss} | returns: {return_mean} | value {value_mean} | policykl: {policykl} | advantages: {torch.mean(advantages)}")
        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward =torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats

config = {
    "tb_path": "runs",
    "hidden_size": 768,
    "cls_model_name": "lvwerra/bert-imdb",
    "tk_name": "gpt2",
    "txt_in_len": 5,
    "txt_out_len": 15,
    "lr": 1.41e-5 * 0.7,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":1000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
    "epochs": 25600,
    "batch_size": 64,
    "forward_batch_size": 4,
    "ppo_epochs": 4,    
}


df = pd.read_csv('data/imdb-dataset.csv')
df = df.loc[df['review'].str.len() > 500]
df['review'] = df['review'].apply(lambda x: x[:1000])


model_path = "checkpoints/checkpoint_57.pt"
config_path = "checkpoints/config.json"

tokenizer, model, device, _  = load_model_and_tokenizer(model_path, config_path)
_, ref_model, _, model_config = load_model_and_tokenizer(model_path, config_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df['tokens'] = df['review'].apply(lambda x: tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])
df['query'] = df['tokens'].apply(lambda x: tokenizer.decode(x))


model = AutoModelForCausalLMWithValueHead(model, model_config, device).to(device)
ref_model = AutoModelForCausalLMWithValueHead(ref_model, model_config, device).to(device)


ppo_trainer = PPOTrainer(model, ref_model, device=device, **config)

def respond_to_batch(model, input_ids, text_length, pad_token_id, device="cuda"):

    input_ids = input_ids.to(device)
    
    batch_size = input_ids.size(0)
    input_length = input_ids.shape[1] 
    
    
    while input_ids.size(1) < text_length:
        with torch.no_grad():
            logits, loss, last_hidden_state = model(input_ids)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)  
            next_token = torch.gather(topk_indices, -1, ix) 
        
            input_ids = torch.cat((input_ids, next_token), dim=1)
    
    if input_ids.size(1) < text_length:
        padding = torch.full((batch_size, text_length - input_ids.size(1)), 
                             pad_token_id, 
                             dtype=torch.long, 
                             device=device)
        input_ids = torch.cat((input_ids, padding), dim=1)
    
    generated_sequence = input_ids[:, input_length:]

    return generated_sequence


#TODO -> train distil bert on imdb dataset for sentiment analysis
sentiment_model = AutoModelForSequenceClassification.from_pretrained(config["cls_model_name"])
sentiment_tokenizer = AutoTokenizer.from_pretrained(config["cls_model_name"])

text = 'this movie was really good!!'
output = sentiment_model.forward(sentiment_tokenizer.encode(text, return_tensors="pt"))

print("predicted sentiment :", output)

fbs = config['forward_batch_size']

model.to(device)
sentiment_model.to(device)
ref_model.to(device)

all_stats = []
for epoch in tqdm(range(int(np.ceil(config["epochs"]/config['batch_size'])))):
    torch.cuda.empty_cache()
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()
    
    #### get a batch from the dataset
    df_batch = df.sample(config['batch_size'])
    game_data['query'] = df_batch['query'].tolist()
    query_tensors = torch.stack(df_batch['tokens'].tolist())
    
    #### get response from gpt2
    t = time.time()
    total_length = config['txt_in_len']+config['txt_out_len']
    response_tensors = []
    for i in range(int(config['batch_size']/fbs)):
        response  = respond_to_batch(model, query_tensors[i*fbs:(i+1)*fbs],
                                     text_length=total_length, pad_token_id=tokenizer.eos_token_id, device=device)
        response_tensors.append(response)
    response_tensors = torch.cat(response_tensors)

    game_data['response'] = [tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
    timing['time/get_response'] = time.time()-t

    #### tokenize text for sentiment analysis
    t = time.time()
    texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]
    sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
    timing['time/build_input_sentiment'] = time.time()-t

    #### get sentiment score
    t = time.time()
    rewards = []
    for i in range(int(config['batch_size']/fbs)):
        res = sentiment_model.forward(sentiment_inputs[i*fbs:(i+1)*fbs],
                                      attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
        rewards.append(res)
    rewards = torch.cat(rewards)
    timing['time/get_sentiment_preds'] = time.time()-t

    #### Run PPO training 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything

    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    all_stats.append(logs)


checkpoint_dir = "ppo_checkpoints"
model.save_pretrained(os.path.join(checkpoint_dir, f"policy_model.pt"), config=config, safe_serialization=True)
ref_model.save_pretrained(os.path.join(checkpoint_dir, f"ref_model.pt"), config=config, safe_serialization=True)

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv("ppo_training_stats.csv", index=False)


#### get a batch from the dataset
bs = 16
game_data = dict()
df_batch = df.sample(bs)
game_data['query'] = df_batch['query'].tolist()
query_tensors = torch.stack(df_batch['tokens'].tolist())

#### get response from gpt2 and gpt2_ref
total_length = config['txt_in_len']+config['txt_out_len']
response_tensors_ref  = respond_to_batch(ref_model, query_tensors, text_length=config['txt_out_len'],  pad_token_id=tokenizer.eos_token_id, device=device)
game_data['response (before)'] = [tokenizer.decode(response_tensors_ref[i, :]) for i in range(bs)]

response_tensors  = respond_to_batch(model, query_tensors, text_length=config['txt_out_len'],  pad_token_id=tokenizer.eos_token_id, device=device)
game_data['response (after)'] = [tokenizer.decode(response_tensors[i, :]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q,r in zip(game_data['query'], game_data['response (before)'])]
sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
rewards = sentiment_model.forward(sentiment_inputs, attention_masks)[0][:, 1].detach()
game_data['rewards (before)'] = rewards.cpu().numpy()

texts = [q + r for q,r in zip(game_data['query'], game_data['response (after)'])]
sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
rewards = sentiment_model.forward(sentiment_inputs, attention_masks)[0][:, 1].detach()
game_data['rewards (after)'] = rewards.cpu().numpy()

# store results in a dataframe
df_results = pd.DataFrame(game_data)
df_results.to_csv("ref_model_response.csv", index=False)