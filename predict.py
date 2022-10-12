# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from PIL import Image
from models.clipcap import *
from parse_clevr import get_image_model
import random

import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device("cpu")

gens = list()

class Predictor():
    def __init__(self, ckpt_path = "./checkpoints/clevr_prefix-004_128.pt"):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda")
        self.feature_extractor, self.image_model = get_image_model()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.prefix_length = 128
        model = ClipCaptionModel(prefix_length = 128, 
                                 clip_length   = 128, 
                                 prefix_size   = 512,
                                 num_layers    = 8,
                                 mapping_type='transformer')
        model.load_state_dict(torch.load(ckpt_path, map_location=CPU))
        model = model.eval()
        self.image_model = self.image_model.cuda().eval()
        self.model = model.cuda()

    def predict(self, image, question ,use_beam_search=False):
        """Run a single prediction on the model"""
        image = Image.open(image).convert('RGB')

        with torch.no_grad():
            #DINO
            #inputs = self.feature_extractor(images=image, return_tensors="pt")
            #inputs['pixel_values'] = inputs['pixel_values'].cuda()
            #outputs = self.image_model(**inputs)      # 224,224
            #prefix = outputs.last_hidden_state[:,-1,:]  # [1, 785, 768]

            #CLIP
            inputs = self.feature_extractor(image).unsqueeze(0).cuda()
            with torch.no_grad():
                prefix = self.image_model.encode_image(inputs).float() # [1,512]

            prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)

        if use_beam_search:
            return generate_beam(self.model,question, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(self.model, question, self.tokenizer, embed=prefix_embed)


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

def generate2(
    model,
    question,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=64,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
):
    model.eval()
    generated_num = 0
    generated_list = []
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                question_id = torch.tensor(tokenizer.encode(question))
                tokens = question_id.unsqueeze(0).cuda()
                text_emb = model.gpt.transformer.wte(tokens)
                generated = torch.cat((embed, text_emb), dim=1) # torch.Size([1, L, 768])
                
                """
                gens.append(generated.cpu())
                
                if(len(gens) > 50):
                    fewshot_ = random.choices(gens, k=50)
                    fewshot_emb = torch.cat(fewshot_,dim=1)
                    generated = torch.cat((fewshot_emb.cuda(), generated), dim=1)
                #print(generated.shape)
                """

            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
            
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

if __name__=="__main__":
    import os,random
    import matplotlib.pyplot as plt

    index = 2
    question = "How many objects are metallic?"
    img_path = os.path.join("/home/seungyoun/dataset/CLEVR_v1.0/images/val", os.listdir("/home/seungyoun/dataset/CLEVR_v1.0/images/val")[index])
    predictor = Predictor()

    plt.imshow(plt.imread(img_path))
    plt.savefig("./results/predict_image_input.png")

    print("path:",img_path)
    pred = predictor.predict(image = img_path, question=question)

    print(pred)

