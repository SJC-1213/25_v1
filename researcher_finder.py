from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

# 연구자 데이터 (실제로는 DB에서 가져올 것)
researchers = [
    {
        "id": 1,
        "name": "김메모리",
        "keywords": ["메모리스터", "저항변화 메모리", "RRAM", "비휘발성 메모리"],
        "field": "반도체기술",
        "subfield": "소자",
        "affiliation": "서울대학교",
        "email": "memory@seoul.ac.kr"
    },
    {
        "id": 2,
        "name": "이뉴로",
        "keywords": ["뉴로모픽 소자", "인공신경망", "뇌형 컴퓨팅", "신경형 소자"],
        "field": "반도체기술",
        "subfield": "소자",
        "affiliation": "KAIST",
        "email": "neuro@kaist.ac.kr"
    },
    {
        "id": 3,
        "name": "박소재",
        "keywords": ["B-C-N 코팅", "CIGS 박막", "양자점 소재", "신소재 개발"],
        "field": "반도체기술",
        "subfield": "소재",
        "affiliation": "포항공대",
        "email": "material@postech.ac.kr"
    },
    {
        "id": 4,
        "name": "최공정",
        "keywords": ["나노패터닝", "ALD 증착", "에칭 최적화", "반도체 공정"],
        "field": "반도체기술",
        "subfield": "공정",
        "affiliation": "한양대학교",
        "email": "process@hanyang.ac.kr"
    }
]

# BERT 모델 초기화
kw_model = KeyBERT()
sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def calculate_name_similarity(query, name):
    """이름 유사도 계산"""
    # 기본 문자열 유사도
    basic_similarity = SequenceMatcher(None, query, name).ratio()
    
    # 문장 임베딩 기반 유사도
    query_embedding = sentence_model.encode(query)
    name_embedding = sentence_model.encode(name)
    embedding_similarity = cosine_similarity([query_embedding], [name_embedding])[0][0]
    
    # 두 유사도의 평균 반환
    return (basic_similarity + embedding_similarity) / 2

def is_name_search(query):
    """이름 검색인지 확인"""
    # 각 연구자 이름과의 유사도 계산
    name_similarities = []
    for researcher in researchers:
        similarity = calculate_name_similarity(query, researcher["name"])
        name_similarities.append((researcher, similarity))
    
    # 가장 높은 유사도를 가진 연구자 찾기
    best_match = max(name_similarities, key=lambda x: x[1])
    
    # 유사도가 0.5 이상이면 이름 검색으로 판단
    return best_match[1] > 0.5, best_match[0] if best_match[1] > 0.5 else None

def expand_query(query):
    """검색어를 확장하여 관련 키워드 추출"""
    # KeyBERT로 주요 키워드 추출
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1,2))
    
    # 문장 임베딩으로 유사 문장 생성
    query_embedding = sentence_model.encode(query)
    
    # 유사 문장 예시 (실제로는 LLM으로 생성)
    similar_queries = [
        "반도체 소자 연구자 찾기",
        "메모리 소자 전문가",
        "뉴로모픽 소자 연구자",
        "반도체 기술 전문가",
        "반도체 소재 연구자",
        "반도체 공정 전문가",
        "메모리 소자 개발자",
        "신소재 개발 연구자"
    ]
    
    # 유사도 계산
    similar_embeddings = sentence_model.encode(similar_queries)
    similarities = cosine_similarity([query_embedding], similar_embeddings)[0]
    
    # 상위 유사 문장 선택
    top_indices = np.argsort(similarities)[-3:][::-1]
    expanded_queries = [similar_queries[i] for i in top_indices]
    
    # 키워드와 유사 문장 결합
    expanded_keywords = [k[0] for k in keywords] + expanded_queries
    
    return expanded_keywords

def find_researchers(query):
    """연구자 검색 및 매칭"""
    # 이름 검색인지 확인
    is_name, matched_researcher = is_name_search(query)
    
    if is_name and matched_researcher:
        # 이름 검색인 경우 해당 연구자만 반환
        return [(matched_researcher, 1.0)]
    
    # 일반 검색인 경우 기존 로직 수행
    expanded_keywords = expand_query(query)
    print("\n확장된 검색어:")
    for kw in expanded_keywords:
        print(f"- {kw}")
    
    # 연구자 매칭 점수 계산
    researcher_scores = []
    for researcher in researchers:
        score = 0
        # 키워드 매칭
        for kw in expanded_keywords:
            if any(k in kw.lower() for k in researcher["keywords"]):
                score += 1
            if researcher["field"].lower() in kw.lower():
                score += 2
            if researcher["subfield"].lower() in kw.lower():
                score += 1
        researcher_scores.append((researcher, score))
    
    # 점수순 정렬
    sorted_researchers = sorted(researcher_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_researchers

def print_results(results):
    """검색 결과 출력"""
    print("\n검색 결과:")
    if not results:
        print("검색 결과가 없습니다.")
        return
        
    for researcher, score in results:
        if score > 0:  # 점수가 0보다 큰 경우만 출력
            print(f"\n연구자: {researcher['name']}")
            print(f"소속: {researcher['affiliation']}")
            print(f"전문 분야: {researcher['field']} - {researcher['subfield']}")
            print(f"주요 키워드: {', '.join(researcher['keywords'])}")
            print(f"이메일: {researcher['email']}")
            print(f"매칭 점수: {score:.2f}")

def main():
    print("반도체 연구자 검색 시스템")
    print("=" * 50)
    print("검색어를 입력하세요 (종료하려면 'q' 입력)")
    print("연구자 이름, 전문 분야, 키워드 등으로 검색 가능합니다.")
    
    while True:
        user_query = input("\n검색어: ").strip()
        
        if user_query.lower() == 'q':
            print("\n검색 시스템을 종료합니다.")
            break
            
        if not user_query:
            print("검색어를 입력해주세요.")
            continue
            
        print(f"\n사용자 검색어: {user_query}")
        results = find_researchers(user_query)
        print_results(results)
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 