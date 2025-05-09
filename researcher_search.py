import json
import openai

openai.api_key = "your-api-key"  # 반드시 본인의 키로 대체

def load_few_shot_examples(filename="few_shot_pairs.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    return pairs

def create_few_shot_messages(system_prompt, few_shot_pairs, user_query):
    messages = [{"role": "system", "content": system_prompt}]
    
    # few-shot 예제들 추가
    for pair in few_shot_pairs:
        messages.append({"role": "user", "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["assistant"]})
    
    # 실제 사용자 질문 추가
    messages.append({"role": "user", "content": user_query})
    
    return messages

def main():
    # 시스템 프롬프트 정의
    system_prompt = """\
너는 사용자의 자연어 질문에서 연구자 정보를 찾기 위한 핵심 키워드나 주제를 추출하는 역할을 해.
응답은 JSON 형식으로 출력하고, 'topics' 필드에 관련 키워드 리스트를 담아줘.
"""
    
    # few-shot 예제 로드
    few_shot_pairs = load_few_shot_examples()
    
    # 사용자 질문
    user_query = input("연구자 검색을 위한 질문을 입력하세요: ")
    
    # few-shot 프롬프트 구성
    messages = create_few_shot_messages(system_prompt, few_shot_pairs[:3], user_query)
    
    # LLM 호출
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )
    
    # 응답 파싱 및 출력
    parsed = response['choices'][0]['message']['content']
    print("\n추출된 키워드:", parsed)

if __name__ == "__main__":
    main() 