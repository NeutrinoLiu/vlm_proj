import argparse
import os
import json
from tqdm import tqdm
import torch
import transformers
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor
)
from peft import PeftModel
from data_utils import TestDataFeeder

MAX_NEW_TOKENS = 128

def load_model(args):

    # load vanilla qwen2.5 model
    print(f" >>> Loading vanilla model from {args.model_base}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_base,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16),
        device_map="auto",
    )

    if not args.base_only:
        # load finetuned non-lora part
        print(f" >>> Loading finetuned non-lora part from non_lora_trainables.bin")
        if os.path.exists(os.path.join(args.model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(args.model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        # load finetuned lora part
        print(f" >>> Loading finetuned lora part from {args.model_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()

        print(f"load model into {model.device}")

    # load tokenizer
    tokenizer_path = args.model_base if args.base_only else args.model_path
    print(f" >>> Loading tokenizer from {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # load image processor
    processor_path = args.model_base if args.base_only else args.model_path
    print(f" >>> Loading image processor from {processor_path}")
    image_processor = AutoProcessor.from_pretrained(
        processor_path,
    ).image_processor

    return model, tokenizer, image_processor

def load_qa_pairs(qa_pair_path):
    with open(qa_pair_path, "r") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} QA pairs from {qa_pair_path}")
    return pairs

def prepare_ans_file(ans_path):
    if os.path.exists(ans_path):
        overwrite = input(f"Answer file {ans_path} already exists. Overwriting it? (y/n): ")
        if overwrite.lower() != "y":
            print("Exiting without overwriting.")
            return
        os.remove(ans_path)
    os.makedirs(os.path.dirname(ans_path), exist_ok=True)

def eval(args):
    prepare_ans_file(args.ans_path)

    model, tokenizer, image_processor = load_model(args)
    feeder = TestDataFeeder(
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=args,
    )
    pairs = load_qa_pairs(args.qa_pair)
    ans_path = os.path.abspath(args.ans_path)
    for idx, pair in enumerate(tqdm(pairs)):
        data_dict = feeder.QA_to_ids(pair)

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to("cuda")
            elif isinstance(v, list):
                data_dict[k] = [i.to("cuda") for i in v]
            else:
                raise ValueError(f"Unsupported type: {type(v)}")

        input_dict = {k:v for k,v in data_dict.items() if k != "labels"}
        pred_ids = model.generate(
            **input_dict,
            max_new_tokens=MAX_NEW_TOKENS
        )
        ans_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_dict["input_ids"], pred_ids)
        ]
        ans_text = tokenizer.batch_decode(
            ans_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        q_text = tokenizer.decode(
            input_dict["input_ids"][0],
            skip_special_tokens=True)

        with open(ans_path, "a") as ans_file:
            ans_file.write(json.dumps({
                "question_id": idx,
                "gt_ans": pair["conversations"][-1]["value"],
                "ans": ans_text,
                "question": q_text,
            }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    model_path
    qa_pair
    ans_path
    model_max_length
    max_pixels
    min_pixels
    """
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--qa-pair", type=str, default=None)
    parser.add_argument("--ans-path", type=str, default="./answers.json")
    parser.add_argument("--model-max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--min-pixels", type=int, default=786)
    parser.add_argument("--base-only", action="store_true", help="Whether to load lora model")
    args = parser.parse_args()

    print(" >>> Arguments:")
    print("   | min_pixels: ", args.min_pixels)
    print("   | max_pixels: ", args.max_pixels)
    print("   | model_max_length: ", args.model_max_length)
    print(" >>> plz confirm above is the same as the training time")

    eval(args)
    
