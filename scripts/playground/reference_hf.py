"""
Usage:
python3 reference_hf.py --model TinyLlama/TinyLlama-1.1B-Chat-v0.4

Reference output:
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.
The capital of Canada is Ottawa.
The capital of Japan is Tokyo
prefill logits tensor([-8.3125, -7.1172,  3.3398,  ..., -4.9570, -4.1328, -3.4141],
       device='cuda:0')
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the United Kingdom is London.
The capital of the United Kingdom is London.
prefill logits tensor([-8.9062, -9.0156,  4.1406,  ..., -4.9922, -4.4961, -4.0742],
       device='cuda:0')
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park to play in the grass and water.
Today is a very
prefill logits tensor([-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4609],
       device='cuda:0')
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def normal_text(args):
    t = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    m.cuda()

    print(m)

    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    max_new_tokens = 32

    for p in prompts:
        if isinstance(p, str):
            input_ids = t.encode(p, return_tensors="pt").cuda()
        else:
            input_ids = torch.tensor([p], device="cuda")

        output_ids = m.generate(
            input_ids, do_sample=False, max_new_tokens=max_new_tokens
        )
        output_str = t.decode(output_ids[0])
        print(output_str)

        prefill_logits = m.forward(input_ids).logits[0][-1]
        print("prefill logits", prefill_logits)


@torch.inference_mode()
def synthetic_tokens(args):
    m = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    m.cuda()
    print(m)

    input_len = 256
    output_len = 8
    prompts = [list(range(5, 5 + input_len))]

    for p in prompts:
        input_ids = p
        for i in range(output_len + 1):
            prefill_logits = m.forward(torch.tensor([input_ids], device="cuda")).logits[
                0
            ][-1]

            if i == 0:
                print("prefill logits", prefill_logits)
            else:
                print("decode", i - 1, prefill_logits)

            input_ids.append(torch.argmax(prefill_logits).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
        # default="meta-llama/Llama-2-7b-chat-hf",
    )
    args = parser.parse_args()

    normal_text(args)
    # synthetic_tokens(args)
