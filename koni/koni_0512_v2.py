##0513 기준 느리지만, 답변은 잘나옴(가상의 html까지 구현성공)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. 모델 및 토크나이저 초기화
model_id = "KISTI-KONI/KONI-Llama3-8B-Instruct-20240729"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    pad_token="<|reserved_special_token_0|>"  # 패딩 토큰 명시적 지정
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2"  # 이 라인 주석 처리
)

# 2. 연구자 데이터베이스 클래스 (논문제목, 논문내용 예시)
class ResearcherDatabase:
    def __init__(self):
        self.researchers = [
            {
                "name": "김인공",
                "id": "RES-2024-AI01",
                "papers": [
                    {
                        "title": "대규모 언어 모델의 효율적 학습 방법론",
                        "content": "본 논문에서는 대규모 언어 모델의 학습 효율성을 개선하기 위한 새로운 방법론을 제안한다. 기존 학습 방식의 한계를 분석하고, 계산 자원을 최적화하는 새로운 학습 알고리즘을 개발하였다. 제안된 방법은 학습 시간을 40% 단축하면서도 모델 성능을 유지할 수 있음을 실험적으로 입증하였다. 또한 다양한 하드웨어 환경에서의 적용 가능성을 검증하였으며, 실제 산업 현장에서의 활용 사례를 제시하였다."
                    },
                    {
                        "title": "멀티모달 AI 시스템의 통합 학습 프레임워크",
                        "content": "텍스트, 이미지, 음성 등 다양한 형태의 데이터를 통합적으로 처리할 수 있는 AI 시스템을 개발하였다. 각 모달리티 간의 상호작용을 최적화하는 새로운 학습 프레임워크를 제안하고, 이를 통해 시스템의 인식 정확도를 크게 향상시켰다. 실시간 처리 속도와 메모리 효율성도 개선되었으며, 실제 응용 분야에서의 검증을 통해 그 효과를 입증하였다."
                    }
                ]
            },
            {
                "name": "이지능",
                "id": "RES-2024-AI02",
                "papers": [
                    {
                        "title": "강화학습 기반 로봇 제어 시스템",
                        "content": "복잡한 환경에서의 로봇 제어를 위한 강화학습 기반 시스템을 개발하였다. 새로운 보상 함수 설계 방법을 통해 학습 안정성을 크게 향상시켰으며, 실제 물리적 환경에서의 적응력을 검증하였다. 시스템은 다양한 작업에 대해 높은 성공률을 보였으며, 학습된 정책의 전이 가능성도 입증하였다. 또한 실시간 의사결정 속도를 개선하여 실제 산업 현장에서의 적용 가능성을 높였다."
                    },
                    {
                        "title": "자율주행 차량의 상황 인식 알고리즘",
                        "content": "자율주행 차량을 위한 고성능 상황 인식 알고리즘을 개발하였다. 센서 데이터의 통합적 처리를 통해 주변 환경을 정확하게 인식하고, 실시간으로 위험 상황을 예측할 수 있는 시스템을 구현하였다. 다양한 기상 조건과 도로 상황에서의 안정성을 검증하였으며, 실제 도로 주행 테스트를 통해 그 신뢰성을 입증하였다."
                    }
                ]
            },
            {
                "name": "박학습",
                "id": "RES-2024-AI03",
                "papers": [
                    {
                        "title": "딥러닝 기반 의료 영상 분석 시스템",
                        "content": "의료 영상 데이터를 분석하기 위한 딥러닝 기반 시스템을 개발하였다. 새로운 네트워크 아키텍처를 통해 영상의 세부적인 특징을 효과적으로 추출하고, 질병 진단의 정확도를 크게 향상시켰다. 실제 의료 데이터를 활용한 검증을 통해 시스템의 신뢰성을 입증하였으며, 의료진의 업무 효율성 향상에 기여할 수 있음을 보였다. 또한 다양한 의료 영상 장비와의 호환성도 검증하였다."
                    },
                    {
                        "title": "인공지능 기반 의료 데이터 보안 시스템",
                        "content": "의료 데이터의 보안을 강화하기 위한 AI 기반 시스템을 개발하였다. 개인정보 보호와 데이터 활용성 사이의 균형을 맞추는 새로운 암호화 방법을 제안하고, 이를 통해 안전한 데이터 공유가 가능함을 입증하였다. 실제 의료기관에서의 적용 사례를 통해 시스템의 실용성을 검증하였으며, 다양한 규제 요구사항을 충족할 수 있음을 보였다."
                    }
                ]
            },
            {
                "name": "최유전",
                "id": "RES-2024-BIO01",
                "papers": [
                    {
                        "title": "유전자 편집 기술의 정밀도 향상 연구",
                        "content": "CRISPR-Cas9 시스템의 정밀도를 크게 향상시키는 새로운 방법을 개발하였다. 오프타겟 효과를 최소화하는 새로운 가이드 RNA 설계 알고리즘을 제안하고, 이를 통해 유전자 편집의 정확도를 99% 이상으로 높였다. 다양한 세포주에서의 검증을 통해 방법의 신뢰성을 입증하였으며, 실제 치료제 개발에 적용 가능함을 보였다. 또한 안전성 평가를 통해 임상 적용의 가능성도 확인하였다."
                    },
                    {
                        "title": "개인 맞춤형 유전자 치료 플랫폼",
                        "content": "개인별 유전적 특성을 고려한 맞춤형 치료 플랫폼을 개발하였다. 환자의 유전체 정보를 분석하여 최적의 치료 방안을 제시하는 알고리즘을 구현하고, 실제 임상 사례를 통해 그 효과를 검증하였다. 다양한 유전 질환에 대한 적용 가능성을 확인하였으며, 치료 비용과 시간을 크게 단축할 수 있음을 입증하였다."
                    }
                ]
            },
            {
                "name": "정단백",
                "id": "RES-2024-BIO02",
                "papers": [
                    {
                        "title": "단백질 구조 예측의 정확도 향상 연구",
                        "content": "단백질의 3차원 구조를 예측하는 새로운 알고리즘을 개발하였다. 딥러닝과 물리 기반 모델을 결합하여 예측 정확도를 크게 향상시켰으며, 특히 복잡한 단백질 구조에서도 높은 성능을 보였다. 실제 단백질 구조 데이터베이스를 활용한 검증을 통해 방법의 신뢰성을 입증하였으며, 신약 개발 과정에 적용 가능함을 보였다. 또한 계산 효율성도 크게 개선되어 실용적인 활용이 가능함을 확인하였다."
                    },
                    {
                        "title": "단백질-약물 상호작용 예측 시스템",
                        "content": "단백질과 약물 후보 물질 간의 상호작용을 예측하는 시스템을 개발하였다. 새로운 머신러닝 모델을 통해 상호작용 강도와 결합 부위를 정확하게 예측할 수 있으며, 실제 약물 개발 사례를 통해 그 효과를 검증하였다. 시스템은 기존 방법 대비 예측 정확도가 30% 이상 향상되었으며, 신약 개발 기간을 크게 단축할 수 있음을 입증하였다."
                    }
                ]
            },
            {
                "name": "강세포",
                "id": "RES-2024-BIO03",
                "papers": [
                    {
                        "title": "줄기세포 분화 제어 기술 개발",
                        "content": "줄기세포의 분화를 정밀하게 제어하는 새로운 기술을 개발하였다. 특정 세포 유형으로의 분화를 유도하는 신호 전달 경로를 규명하고, 이를 통해 원하는 세포 유형으로의 분화 효율을 크게 향상시켰다. 다양한 세포주에서의 검증을 통해 방법의 재현성을 입증하였으며, 실제 조직 재생 치료에 적용 가능함을 보였다. 또한 분화된 세포의 기능성과 안정성도 검증하였다."
                    },
                    {
                        "title": "장기 칩 기술의 발전과 응용",
                        "content": "인체 장기를 모사하는 마이크로칩 기술을 개발하였다. 새로운 미세유체 시스템을 통해 실제 장기와 유사한 환경을 구현하고, 약물 반응을 정확하게 예측할 수 있는 플랫폼을 구축하였다. 실제 약물 개발 과정에 적용하여 그 유용성을 입증하였으며, 동물 실험을 대체할 수 있는 가능성도 확인하였다. 또한 다양한 장기 모델의 통합을 통해 전신 반응 예측도 가능함을 보였다."
                    }
                ]
            }
        ]
        self.vectorizer = TfidfVectorizer()
        self._build_index()

    def _build_index(self):
        """TF-IDF 인덱스 구축"""
        texts = [
            " ".join(
                [paper['title'] + " " + paper['content'] for paper in res['papers']]
            )
            for res in self.researchers
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5):
        """초기 키워드 기반 검색"""
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        sorted_indices = np.argsort(sim_scores)[::-1]
        return [self.researchers[i] for i in sorted_indices[:top_k]]

# 3. 논문 처리 파이프라인
def extract_keywords(paper_text, num_keywords=5):
    prompt = f"""당신은 논문의 핵심 키워드를 추출하는 전문가입니다.
다음 논문에서 가장 중요한 {num_keywords}개의 키워드를 추출해주세요.
키워드는 반드시 연구 분야, 기술, 방법론을 나타내는 명사여야 합니다.
키워드만 콤마로 구분하여 한 줄로 출력해주세요.

논문: {paper_text}

키워드:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 이후의 내용만 추출
    try:
        keywords_text = result.split("키워드:")[-1].strip()
        # 콤마로 구분된 키워드 추출
        keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
        # 키워드가 너무 길거나 문장 형태인 경우 제외
        filtered_keywords = [kw for kw in keywords if len(kw) <= 20 and not any(char in kw for char in ['.', '!', '?'])]
        return filtered_keywords[:num_keywords]
    except:
        return []

def get_researcher_keywords(researcher):
    all_keywords = []
    for paper in researcher['papers']:
        paper_text = paper['title'] + " " + paper['content']
        keywords = extract_keywords(paper_text)
        all_keywords.extend(keywords)
    
    # 중복 제거 및 빈도수 기반 정렬
    keyword_freq = {}
    for kw in all_keywords:
        if len(kw) > 1:  # 한 글자 키워드 제외
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
    
    # 빈도수가 높은 순으로 정렬하고 상위 10개만 선택
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:10]]

# 4. 검색 시스템 코어
def research_search(query):
    db = ResearcherDatabase()
    researchers = db.researchers
    ranked = rank_researchers_by_query(query, researchers)
    results = []
    for sim, res in ranked[:5]:  # 상위 5명
        res_keywords = get_researcher_keywords(res)
        results.append({
            "name": res['name'],
            "id": res['id'],
            "keywords": res_keywords,
            "similarity": float(sim)
        })
    return results

def rank_researchers_by_query(query, researchers):
    # 각 연구자별 키워드 추출
    researcher_keywords = [get_researcher_keywords(r) for r in researchers]
    
    # 쿼리에서 핵심 키워드 추출
    query_keywords = extract_keywords(query)
    
    # TF-IDF 기반 유사도 계산 (전체 맥락 기반)
    docs = [" ".join(keywords) for keywords in researcher_keywords]
    all_texts = [query] + docs
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_texts)
    query_vec = tfidf[0]
    doc_vecs = tfidf[1:]
    tfidf_scores = cosine_similarity(query_vec, doc_vecs).flatten()
    
    # 키워드 기반 유사도 계산 (정확한 매칭)
    keyword_scores = []
    for keywords in researcher_keywords:
        # 쿼리 키워드와 연구자 키워드 간의 일치도 계산
        matches = sum(1 for qk in query_keywords if any(qk in rk for rk in keywords))
        # 일치하는 키워드가 있으면 기본 점수 부여
        score = 0.2 if matches > 0 else 0.0
        # 일치하는 키워드 수에 따라 추가 점수 부여
        score += min(matches * 0.15, 0.6)  # 최대 0.6까지 추가 점수
        keyword_scores.append(score)
    
    # 최종 점수 계산 (TF-IDF 기반 60%, 키워드 기반 40%)
    final_scores = [0.6 * ts + 0.4 * ks for ts, ks in zip(tfidf_scores, keyword_scores)]
    
    # 점수와 연구자 정보를 결합하여 정렬
    ranked = sorted(zip(final_scores, researchers), key=lambda x: x[0], reverse=True)
    return ranked

# 5. 실행 예시
if __name__ == "__main__":
    try:
        query = "인공지능 관련 연구자를 찾아줘"
        print(f"검색 질문: {query}\n")
        results = research_search(query)
        print("검색 결과:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"오류 발생: {str(e)}")
