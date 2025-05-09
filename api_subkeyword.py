import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_base_fields():
    """기본 연구 분야 로드"""
    return {
        "반도체기술": {
            "main_keywords": ["반도체", "반도체기술", "반도체 연구"],
            "description": "반도체 소자, 회로, 공정 기술 및 관련 응용 분야"
        },
        "인공지능": {
            "main_keywords": ["인공지능", "AI", "머신러닝"],
            "description": "인공지능, 머신러닝, 딥러닝 및 관련 응용 분야"
        },
        "신경과학": {
            "main_keywords": ["신경과학", "뇌과학", "신경망"],
            "description": "뇌와 신경계 연구, 인지과학 및 관련 응용 분야"
        }
    }

def generate_sub_keywords(field_name, description, main_keywords):
    """LLM을 사용하여 서브 키워드 생성"""
    prompt = f"""
    다음 연구 분야에 대한 세부 키워드를 생성해주세요:
    
    분야명: {field_name}
    설명: {description}
    주요 키워드: {', '.join(main_keywords)}
    
    다음 형식으로 20개의 세부 키워드를 생성해주세요:
    - 각 키워드는 해당 분야의 구체적인 연구 주제나 기술을 나타내야 합니다
    - 키워드는 한글이나 영어로 작성 가능합니다
    - 키워드는 쉼표로 구분하여 나열해주세요
    """
    
    # 여기서는 예시로 하드코딩된 키워드를 반환합니다
    # 실제 구현시에는 LLM API를 호출하여 키워드를 생성해야 합니다
    if field_name == "반도체기술":
        return [
            "휴먼인터페이스 소자 및 센서",
            "나노바이오 소재 및 소자",
            "바이오 전자소자",
            "스핀트랜지스터",
            "스핀메모리",
            "반도체소자",
            "차세대 실리콘 태양전지",
            "실리콘/유기 이종접합 전자소자",
            "신경모사 인공 시냅스 소자",
            "반도체 신소재 개발",
            "반도체 신소재 합성",
            "반도체 신물성 연구",
            "스핀트로닉스",
            "B-based Superhard Coating",
            "C-based Superhard Coating",
            "N-based Superhard Coating",
            "Thin Film Solar Cell",
            "CIGS",
            "CZTS",
            "인공지능 광-반도체 소자"
        ]
    elif field_name == "인공지능":
        return [
            "딥러닝",
            "강화학습",
            "자연어처리",
            "컴퓨터비전",
            "음성인식",
            "머신러닝",
            "신경망",
            "전이학습",
            "메타러닝",
            "페더럴러닝",
            "설명가능한 AI",
            "AI 윤리",
            "AI 보안",
            "AI 하드웨어",
            "AI 최적화",
            "AI 응용",
            "AI 교육",
            "AI 의료",
            "AI 금융",
            "AI 제조"
        ]
    elif field_name == "신경과학":
        return [
            "인지신경과학",
            "분자신경과학",
            "계산신경과학",
            "신경영상",
            "신경공학",
            "신경인공지능",
            "신경회로",
            "신경전달물질",
            "신경발생학",
            "신경면역학",
            "신경퇴행성질환",
            "신경정신의학",
            "신경약리학",
            "신경생리학",
            "신경해부학",
            "신경유전학",
            "신경발달",
            "신경가소성",
            "신경재생",
            "신경보철"
        ]

def main():
    # 1. 기본 연구 분야 로드
    research_fields = load_base_fields()
    
    # 2. 각 분야별 서브 키워드 생성
    for field_name, field_data in research_fields.items():
        sub_keywords = generate_sub_keywords(
            field_name,
            field_data["description"],
            field_data["main_keywords"]
        )
        field_data["sub_keywords"] = sub_keywords
    
    # 3. 결과를 JSON 파일로 저장
    with open("research_fields.json", "w", encoding="utf-8") as f:
        json.dump(research_fields, f, ensure_ascii=False, indent=4)
    
    print("✅ 연구 분야 키워드 생성 완료")

if __name__ == "__main__":
    main()