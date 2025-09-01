"""
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py
https://github.com/sgl-project/sglang/blob/main/benchmark/mtbench/bench_sglang_eagle.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding with multiple input files

Usage:
file1="spec_bench_rag.jsonl"
file2="spec_bench_qa.jsonl"
file3="spec_bench_summarization.jsonl"
python3 bench_sglang_eagle_multi.py \
  --question-files "$file1" "$file2" "$file3" \
  --num-questions 80 \
  --temperature 0 \
  --max-gen-length 1024 \
  --answer-file-suffix "$answer_suffix" \
  --result-file-suffix "$result_suffix" \
  --port 30000
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

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
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i]["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "answer": answers[i],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_question(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))


def main(args):
    # Select backend once
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)
    model_id = backend.model_info["model_path"]
    
    # Process each file independently
    for question_file in args.question_files:
        print(f"\n{'='*60}")
        print(f"Processing: {question_file}")
        print('='*60)
        
        # Load questions for this file
        questions = load_questions(question_file)
        if args.num_questions > 0:
            questions = questions[:args.num_questions]
            
        # Get category from first question
        category = questions[0].get("category", Path(question_file).stem)
        print(f"Category: {category}")
        print(f"Number of questions: {len(questions)}")
        
        # Prepare arguments
        arguments = [{"question": q["turns"][0]} for q in questions]
        
        # Run inference for this file
        print(f"Running inference...")
        tic = time.perf_counter()
        
        rets = answer_question.run_batch(
            arguments,
            temperature=args.temperature,
            max_new_tokens=args.max_gen_length,
            num_threads=args.parallel,
            progress_bar=True,
        )
        
        answers = [s["answer"] for s in rets]
        latency = time.perf_counter() - tic
        
        # Calculate metrics
        num_output_tokens = sum(
            s.get_meta_info("answer")["completion_tokens"]
            for s in rets
        )
        
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
        
        print(f"#questions: {len(questions)}, Throughput: {output_throughput:.2f} token/s, Acceptance length: {accept_length:.2f}")
        
        # Generate output filenames based on category
        category_name = category.replace("/", "_").replace(" ", "_")
        answer_file = f"spec-bench-{category_name}_{args.answer_file_suffix}.jsonl"
        result_file = f"spec-bench-{category_name}_{args.result_file_suffix}.jsonl"
        
        # Write answers
        write_answers(answer_file, model_id, questions, answers)
        
        # Write results
        with open(result_file, "w") as fout:
            value = {
                "task": f"spec-bench-{category}",
                "backend": args.backend,
                "num_gpus": 1,
                "latency": round(latency, 3),
                "throughput": round(output_throughput, 3),
                "accept_length": round(accept_length, 3),
                "num_requests": len(questions),
                "other": {
                    "num_questions": len(questions),
                    "parallel": args.parallel,
                    "file": question_file,
                    "category": category,
                },
            }
            fout.write(json.dumps(value) + "\n")
        print(f"Results written to: {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-files", nargs="+", required=True,
                       help="Input JSONL files (space-separated)")
    parser.add_argument("--answer-file-suffix", type=str, default="answers",
                       help="Prefix for answer files")
    parser.add_argument("--result-file-suffix", type=str, default="results",
                       help="Prefix for result files")
    parser.add_argument("--num-questions", type=int, default=-1,
                       help="Number of questions per file (-1 for all)")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-gen-length", type=int, default=1024)
    args = add_common_sglang_args_and_parse(parser)
    main(args)