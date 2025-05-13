# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="KISTI-KONI/KONI-Llama3-8B-Instruct-20240729")
pipe(messages)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델과 토크나이저 로드
model_id = "KISTI-KONI/KONI-Llama3-8B-Instruct-20240729"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def generate_response(instruction):
    messages = [
        {"role": "user", "content": instruction}
    ]
    
    # 채팅 템플릿 적용
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 입력을 토큰화하여 텐서로 변환 (attention_mask 포함)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # 생성 파라미터 설정
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # pad_token_id를 명시적으로 지정
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=2048,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):]

# 테스트
try:
    response = generate_response("양자 컴퓨팅의 기본 원리를 설명해주세요.")
    print("응답:", response)
except Exception as e:
    print("오류 발생:", str(e))