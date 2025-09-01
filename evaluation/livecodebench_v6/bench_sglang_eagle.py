"""
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py
https://github.com/sgl-project/sglang/blob/main/benchmark/mtbench/bench_sglang_eagle.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding

Usage:
python3 bench_sglang_eagle.py \
  --question-file livecodebench_sampled_80_reformat_2.jsonl \
  --num-questions 80 \
  --temperature 0 \
  --max-gen-length 1024 \
  --answer-file "$answer_file" \
  --result-file "$result_file"
"""

import argparse
import json
import os
import time
import uuid

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions
            
    
def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w", encoding='utf-8') as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i]["question_id"],
                "question_title": questions[i]["question_title"],
                "contest_id": questions[i]["contest_id"],
                "platform": questions[i]["platform"],
                "difficulty": questions[i]["difficulty"],
                "question": questions[i]["question"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "answer": answers[i],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

    
@sgl.function
def answer_livecodebench_question(s, system, question):
    s += sgl.system(system)
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))

def main(args):
    # Construct prompts
    questions = load_questions(args.question_file)[: args.num_questions]
    arguments = [{"question": q["question"], "system": q["system"]} for q in questions]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.perf_counter() 
    rets = answer_livecodebench_question.run_batch(
        arguments,
        temperature=args.temperature,
        max_new_tokens=args.max_gen_length,
        num_threads=args.parallel,
        progress_bar=True,
    )
    answers = [s["answer"].strip() for s in rets]

    latency = time.perf_counter() - tic
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"]
        for s in rets
    )

    # NOTE: acceptance length is just completion_tokens / spec_verify_ct
    # {'id': '3bb9c5ead109488d8ed5ee9cbecaec29', 'finish_reason': {'type': 'length', 'length': 256}, 'prompt_tokens': 37, 'spec_verify_ct': 101, 'completion_tokens': 256, 'cached_tokens': 0}

    output_throughput = num_output_tokens / latency
        
    has_verify = "spec_verify_ct" in rets[0].get_meta_info("answer")
    if has_verify:
        num_verify_tokens = sum(
            s.get_meta_info("answer")["spec_verify_ct"]
            for s in rets
        )
        accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    print(
        f"#questions: {len(questions)}, Throughput: {output_throughput:.2f} token/s, Acceptance length: {accept_length:.2f}"
    )

    # Write results
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"tmp_output_{args.backend}.txt"
    write_answers(answer_file, model_id, questions, answers)
        
    with open(args.result_file, "a", encoding='utf-8') as fout:
        value = {
            "task": "ceval",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "throughput": round(output_throughput, 3),
            "accept_length": round(accept_length, 3),
            "num_requests": len(questions),
            "other": {
                "num_questions": len(questions),
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-gen-length", type=int, default=1024)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
    