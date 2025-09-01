"""
Reference:
https://github.com/hkust-nlp/ceval/blob/main/resources/tutorial.md
"""
import json
import os

def load_category_mapping(mapping_file):
    
    category_to_chinese = {}
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
        for key, value in mapping_data.items():
            category_to_chinese[key] = value[1]
            
    return category_to_chinese


def format_to_prompt(input_file, output_file, mapping_file):

    category_to_chinese = load_category_mapping(mapping_file)
    
#     prompt_template = """以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。
# {question}
# A. {option_a}
# B. {option_b}
# C. {option_c}
# D. {option_d}
# 答案："""

    prompt_template = """以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。
{question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
"""
    
    formatted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    subject = category_to_chinese.get(data['category'], data['category'])
                    formatted_prompt = prompt_template.format(
                        subject=subject,
                        question=data['question'],
                        option_a=data['A'],
                        option_b=data['B'],
                        option_c=data['C'],
                        option_d=data['D']
                    )
                    
                    new_data = {
                        'id': data['id'],
                        'question': formatted_prompt,
                        'category': data['category'],
                        'answer': data['answer']
                    }
                    
                    formatted_data.append(new_data)
                    
                except Exception as e:
                    print(f"Line {line_num} processing error: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Done! Processed {len(formatted_data)} records in total.")
    
    return formatted_data

def main():
    input_file = 'ceval_sampled.jsonl'
    output_file = 'ceval_sampled_prompt_formatted.jsonl'
    
    # Reference:
    # https://github.com/hkust-nlp/ceval/blob/main/subject_mapping.json
    mapping_file = 'subject_mapping_zh.json'
    
    format_to_prompt(input_file, output_file, mapping_file)

if __name__ == "__main__":
    main()