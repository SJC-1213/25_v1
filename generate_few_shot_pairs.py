import json
import random

def load_templates(template_file="question_templates.json"):
    """템플릿 파일을 로드하고 모든 템플릿을 하나의 리스트로 반환"""
    with open(template_file, 'r', encoding='utf-8') as f:
        templates_dict = json.load(f)
    
    # 모든 카테고리의 템플릿을 하나의 리스트로 합치기
    all_templates = []
    for category_templates in templates_dict.values():
        all_templates.extend(category_templates)
    
    return all_templates

def load_research_fields(fields_file="research_fields.json"):
    """연구 분야와 키워드를 로드"""
    with open(fields_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_keywords(research_fields):
    """모든 키워드를 하나의 리스트로 반환"""
    all_keywords = []
    for field, data in research_fields.items():
        all_keywords.extend(data["main_keywords"])
        all_keywords.extend(data["sub_keywords"])
    return all_keywords

def get_field_for_keyword(keyword, research_fields):
    """키워드에 해당하는 연구 분야 반환"""
    for field, data in research_fields.items():
        if keyword in data["main_keywords"] or keyword in data["sub_keywords"]:
            return field
    return None

# 1. 템플릿과 연구 분야 로드
question_templates = load_templates()
research_fields = load_research_fields()
all_keywords = get_all_keywords(research_fields)

# 2. 결과 리스트
finetune_data = []

# 3. 생성 반복
for keyword in all_keywords:
    # 각 키워드당 5개의 서로 다른 질문 생성
    selected_templates = random.sample(question_templates, 5)
    for template in selected_templates:
        question = template.format(keyword)
        field = get_field_for_keyword(keyword, research_fields)
        answer = json.dumps({"topics": [field]}, ensure_ascii=False)
        
        # Fine-tuning용 포맷으로 추가
        finetune_data.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })

# 4. JSONL 파일로 저장
output_path = "fewshot_finetune_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in finetune_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Fine-tuning용 데이터셋 생성 완료: {output_path}")
print(f"총 {len(finetune_data)}개의 질문/답변 쌍이 생성되었습니다.")
