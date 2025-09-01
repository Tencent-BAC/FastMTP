"""
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py
https://github.com/sgl-project/sglang/blob/main/benchmark/mtbench/bench_sglang_eagle.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding

Usage:
python3 bench_sglang_eagle.py \
  --question-file test_sampled.jsonl \
  --num-questions 80 \
  --temperature 0 \
  --max-gen-length 1024 \
  --answer-file "$answer_file" \
  --result-file "$result_file" \
  --port 30000
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


def format_prompt(problem):
    """Format the problem into a prompt"""
    prompt = f'{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'
    return prompt


def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i].get("unique_id", f"q_{i}"),  # Use unique_id
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "problem": questions[i]["problem"],  # Save original problem
                "ground_truth": questions[i]["answer"],  # Save ground truth answer
                "subject": questions[i].get("subject", "unknown"),
                "level": questions[i].get("level", 0),
                "model_answer": answers[i],  # Model's response
                "choices": {
                    "index": 0,
                    "answer": answers[i],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_math_problem(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))


def main(args):
    # Load questions
    questions = load_questions(args.question_file)
    if args.num_questions > 0:
        questions = questions[: args.num_questions]
    
    print(f"Loaded {len(questions)} questions")
    
    # Format problems into prompts
    arguments = [
        {"question": format_prompt(q["problem"])} for q in questions
    ]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    print("Running inference...")
    tic = time.perf_counter()
    rets = answer_math_problem.run_batch(
        arguments,
        temperature=args.temperature,
        max_new_tokens=args.max_gen_length,
        num_threads=args.parallel,
        progress_bar=True,
    )
    answers = [s["answer"] for s in rets]

    latency = time.perf_counter() - tic
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"]
        for s in rets
    )

    output_throughput = num_output_tokens / latency

    # Check for speculative decoding
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

    # Write answers
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"math_output_{args.backend}_{int(time.time())}.jsonl"
    write_answers(answer_file, model_id, questions, answers)
    print(f"Answers written to: {answer_file}")

    # Write results
    if args.result_file:
        with open(args.result_file, "a") as fout:
            value = {
                "task": "math",
                "backend": args.backend,
                "num_gpus": 1,
                "latency": round(latency, 3),
                "throughput": round(output_throughput, 3),
                "accept_length": round(accept_length, 3),
                "num_requests": len(questions),
                "num_output_tokens": num_output_tokens,
                "other": {
                    "num_questions": len(questions),
                    "parallel": args.parallel,
                    "temperature": args.temperature,
                    "max_gen_length": args.max_gen_length,
                },
            }
            
            fout.write(json.dumps(value) + "\n")
        print(f"Results written to: {args.result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="math_questions.jsonl",
                       help="Input JSONL file with MATH problems")
    parser.add_argument("--answer-file", type=str, default=None,
                       help="Output file for answers")
    parser.add_argument("--num-questions", type=int, default=-1,
                       help="Number of questions to process (-1 for all)")
    parser.add_argument("--temperature", type=float, default=0,
                       help="Generation temperature")
    parser.add_argument("--max-gen-length", type=int, default=2048,
                       help="Maximum generation length (increased for math problems)")
    args = add_common_sglang_args_and_parse(parser)
    main(args)
    