"""
Adapted from 
https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
"""

import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class CodeGenerationProblem:
    question_content: str
    starter_code: Optional[str] = None

        
class PromptConstants:

    SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
    
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    
def generate_question_prompt(question: CodeGenerationProblem):

    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    
    if question.starter_code and question.starter_code.strip():
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    
    return prompt

def convert_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    
    problem = CodeGenerationProblem(
        question_content=item.get('question_content', ''),
        starter_code=item.get('starter_code')
    )
    
    system_content = PromptConstants.SYSTEM_MESSAGE_GENERIC
    question_content = generate_question_prompt(problem)
    
    new_item = item.copy()
    new_item['system'] = system_content
    new_item['question'] = question_content
    
    return new_item

def convert_jsonl_format(input_file: str, output_file: str):
    
    converted_data = []
    total_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line.strip())
                total_count += 1
                
                converted_item = convert_single_item(item)
                converted_data.append(converted_item)
                    
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Line {line_num} JSON parsing error: {e}")
            except Exception as e:
                error_count += 1
                print(f"Line {line_num} processing error: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nConversion complete!")
    print(f"Total processed: {total_count} records")
    print(f"Successfully converted: {len(converted_data)} records")
    print(f"Error count: {error_count} records")
    print(f"Output file: {output_file}")

def main():
    
    convert_jsonl_format("livecodebench_sampled_80.jsonl", "livecodebench_sampled_80_reformat_2.jsonl")


if __name__ == "__main__":
    main()