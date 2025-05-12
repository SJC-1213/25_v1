from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

# 연구자 데이터 (30명)
researchers = [
    # 반도체 분야 (10명)
    {
        "id": 1,
        "name": "김메모리",
        "keywords": ["메모리스터", "RRAM", "비휘발성 메모리"],
        "field": "반도체",
        "subfield": "소자",
        "affiliation": "서울대학교",
        "email": "memory@seoul.ac.kr"
    },
    {
        "id": 2,
        "name": "이패키징",
        "keywords": ["3D 패키징", "실리콘 인터포저", "열 관리"],
        "field": "반도체",
        "subfield": "패키징",
        "affiliation": "KAIST",
        "email": "package@kaist.ac.kr"
    },
    {
        "id": 3,
        "name": "박소재",
        "keywords": ["2D 소재", "그래핀 트랜지스터", "초박막 소자"],
        "field": "반도체",
        "subfield": "소재",
        "affiliation": "POSTECH",
        "email": "material@postech.ac.kr"
    },
    {
        "id": 4,
        "name": "최공정",
        "keywords": ["나노패터닝", "ALD 증착", "에칭 공정"],
        "field": "반도체",
        "subfield": "공정",
        "affiliation": "한양대학교",
        "email": "process@hanyang.ac.kr"
    },
    {
        "id": 5,
        "name": "정설계",
        "keywords": ["3D IC 설계", "전력 최적화", "설계 자동화"],
        "field": "반도체",
        "subfield": "설계",
        "affiliation": "연세대학교",
        "email": "design@yonsei.ac.kr"
    },
    {
        "id": 6,
        "name": "강광전자",
        "keywords": ["GaN 소자", "전력 반도체", "고주파 소자"],
        "field": "반도체",
        "subfield": "소자",
        "affiliation": "광주과학기술원",
        "email": "opto@gist.ac.kr"
    },
    {
        "id": 7,
        "name": "윤테스트",
        "keywords": ["신뢰성 평가", "결함 분석", "자동 검사"],
        "field": "반도체",
        "subfield": "테스트",
        "affiliation": "한국과학기술원",
        "email": "test@kaist.ac.kr"
    },
    {
        "id": 8,
        "name": "한퓨처",
        "keywords": ["양자점 소자", "초전도체", "나노 와이어"],
        "field": "반도체",
        "subfield": "차세대기술",
        "affiliation": "UNIST",
        "email": "future@unist.ac.kr"
    },
    {
        "id": 9,
        "name": "송열관리",
        "keywords": ["열 분산 기술", "열전소자", "마이크로 쿨링"],
        "field": "반도체",
        "subfield": "열관리",
        "affiliation": "성균관대학교",
        "email": "thermal@skku.edu"
    },
    {
        "id": 10,
        "name": "류집적",
        "keywords": ["SoC 설계", "하드웨어 가속기", "병렬 처리"],
        "field": "반도체",
        "subfield": "집적회로",
        "affiliation": "고려대학교",
        "email": "ic@korea.ac.kr"
    },

    # 바이오의료 분야 (10명)
    {
        "id": 11,
        "name": "박바이오",
        "keywords": ["CRISPR", "유전자 편집", "세포 재프로그래밍"],
        "field": "바이오의료",
        "subfield": "유전공학",
        "affiliation": "KAIST",
        "email": "bio@kaist.ac.kr"
    },
    {
        "id": 12,
        "name": "김줄기세포",
        "keywords": ["iPS 세포", "세포 분화", "조직 재생"],
        "field": "바이오의료",
        "subfield": "세포공학",
        "affiliation": "서울대학교",
        "email": "stemcell@seoul.ac.kr"
    },
    {
        "id": 13,
        "name": "이진단",
        "keywords": ["바이오마커", "조기 진단", "액체 생검"],
        "field": "바이오의료",
        "subfield": "진단기술",
        "affiliation": "POSTECH",
        "email": "diagnosis@postech.ac.kr"
    },
    {
        "id": 14,
        "name": "최신약",
        "keywords": ["표적 치료제", "항체 의약품", "약물 전달 시스템"],
        "field": "바이오의료",
        "subfield": "신약개발",
        "affiliation": "연세대학교",
        "email": "drug@yonsei.ac.kr"
    },
    {
        "id": 15,
        "name": "정뇌과학",
        "keywords": ["신경 인터페이스", "뇌파 분석", "신경 회로"],
        "field": "바이오의료",
        "subfield": "신경과학",
        "affiliation": "KAIST",
        "email": "neuroscience@kaist.ac.kr"
    },
    {
        "id": 16,
        "name": "홍미생물",
        "keywords": ["장내 미생물", "미생물 군집", "프로바이오틱스"],
        "field": "바이오의료",
        "subfield": "미생물학",
        "affiliation": "서울대학교",
        "email": "microbiome@seoul.ac.kr"
    },
    {
        "id": 17,
        "name": "윤영상",
        "keywords": ["의료 영상 처리", "MRI 분석", "초음파 영상"],
        "field": "바이오의료",
        "subfield": "의료영상",
        "affiliation": "한양대학교",
        "email": "imaging@hanyang.ac.kr"
    },
    {
        "id": 18,
        "name": "장재생",
        "keywords": ["조직 공학", "인공 장기", "바이오 프린팅"],
        "field": "바이오의료",
        "subfield": "재생의학",
        "affiliation": "UNIST",
        "email": "regeneration@unist.ac.kr"
    },
    {
        "id": 19,
        "name": "조면역",
        "keywords": ["암 면역 치료", "CAR-T 세포", "면역 체계 조절"],
        "field": "바이오의료",
        "subfield": "면역학",
        "affiliation": "고려대학교",
        "email": "immuno@korea.ac.kr"
    },
    {
        "id": 20,
        "name": "한노화",
        "keywords": ["세포 노화", "테로미어", "항노화 치료"],
        "field": "바이오의료",
        "subfield": "노화연구",
        "affiliation": "성균관대학교",
        "email": "aging@skku.edu"
    },

    # 인공지능 분야 (10명)
    {
        "id": 21,
        "name": "최AI",
        "keywords": ["딥러닝", "강화학습", "컴퓨터 비전"],
        "field": "인공지능",
        "subfield": "머신러닝",
        "affiliation": "POSTECH",
        "email": "ai@postech.ac.kr"
    },
    {
        "id": 22,
        "name": "김자연어",
        "keywords": ["NLP", "텍스트 생성", "언어 모델"],
        "field": "인공지능",
        "subfield": "자연어처리",
        "affiliation": "서울대학교",
        "email": "nlp@seoul.ac.kr"
    },
    {
        "id": 23,
        "name": "이강화",
        "keywords": ["심층 강화학습", "게임 AI", "다중 에이전트"],
        "field": "인공지능",
        "subfield": "강화학습",
        "affiliation": "KAIST",
        "email": "rl@kaist.ac.kr"
    },
    {
        "id": 24,
        "name": "박지식",
        "keywords": ["지식 그래프", "추론 시스템", "온톨로지"],
        "field": "인공지능",
        "subfield": "지식표현",
        "affiliation": "연세대학교",
        "email": "kg@yonsei.ac.kr"
    },
    {
        "id": 25,
        "name": "정최적화",
        "keywords": ["하이퍼파라미터 튜닝", "분산 학습", "모델 압축"],
        "field": "인공지능",
        "subfield": "최적화",
        "affiliation": "한양대학교",
        "email": "optimize@hanyang.ac.kr"
    },
    {
        "id": 26,
        "name": "강생성",
        "keywords": ["GAN", "이미지 생성", "창의적 AI"],
        "field": "인공지능",
        "subfield": "생성모델",
        "affiliation": "UNIST",
        "email": "gan@unist.ac.kr"
    },
    {
        "id": 27,
        "name": "윤윤리",
        "keywords": ["AI 윤리", "편향 완화", "투명한 AI"],
        "field": "인공지능",
        "subfield": "AI윤리",
        "affiliation": "고려대학교",
        "email": "ethics@korea.ac.kr"
    },
    {
        "id": 28,
        "name": "한추천",
        "keywords": ["개인화 추천", "콘텐츠 필터링", "협업 필터링"],
        "field": "인공지능",
        "subfield": "추천시스템",
        "affiliation": "성균관대학교",
        "email": "recsys@skku.edu"
    },
    {
        "id": 29,
        "name": "송로봇",
        "keywords": ["로봇 제어", "물리적 상호작용", "강화학습 적용"],
        "field": "인공지능",
        "subfield": "로보틱스",
        "affiliation": "POSTECH",
        "email": "robot@postech.ac.kr"
    },
    {
        "id": 30,
        "name": "류음성",
        "keywords": ["음성 인식", "화자 식별", "음성 합성"],
        "field": "인공지능",
        "subfield": "음성처리",
        "affiliation": "KAIST",
        "email": "speech@kaist.ac.kr"
    }
]


# BERT 모델 초기화
kw_model = KeyBERT()
sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def calculate_name_similarity(query, name):
    """이름 유사도 계산 (변경 없음)"""
    basic_similarity = SequenceMatcher(None, query, name).ratio()
    query_embedding = sentence_model.encode(query)
    name_embedding = sentence_model.encode(name)
    embedding_similarity = cosine_similarity([query_embedding], [name_embedding])[0][0]
    return (basic_similarity + embedding_similarity) / 2

def is_name_search(query):
    """이름 검색 확인 (변경 없음)"""
    name_similarities = [(r, calculate_name_similarity(query, r["name"])) for r in researchers]
    best_match = max(name_similarities, key=lambda x: x[1])
    return best_match[1] > 0.5, best_match[0] if best_match[1] > 0.5 else None

def expand_query(query):
    """도메인 독립적 검색어 확장"""
    # 1. KeyBERT를 이용한 초기 키워드 추출
    keywords = kw_model.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 3),
        top_n=7,
        use_mmr=True,
        diversity=0.7
    )
    
    # 2. 의미 없는 키워드 필터링
    stop_words = ['관련', '알려줘', '관리', '대한', '위한', '있는', '하는']
    meaningful_keywords = [
        kw for kw, _ in keywords 
        if not any(sw in kw for sw in stop_words)
    ]
    
    # 3. 연구자 키워드 기반 동적 확장
    all_keywords = list({kw for r in researchers for kw in r["keywords"]})
    query_embedding = sentence_model.encode(query)
    
    # 상위 5개 유사 키워드 선택
    keyword_embeddings = sentence_model.encode(all_keywords)
    similarities = cosine_similarity([query_embedding], keyword_embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    # 4. 최종 확장 키워드 생성
    expanded_keywords = meaningful_keywords + [all_keywords[i] for i in top_indices]
    return list(set(expanded_keywords))  # 중복 제거

def find_researchers(query):
    """개선된 연구자 검색 로직"""
    # 이름 검색 처리 (변경 없음)
    is_name, matched_researcher = is_name_search(query)
    if is_name and matched_researcher:
        return [(matched_researcher, 1.0)]

    # 확장 키워드 생성
    expanded_keywords = expand_query(query)
    print("\n확장된 검색어:", ", ".join(expanded_keywords))

    # 유사도 기반 점수 계산
    researcher_scores = []
    query_embedding = sentence_model.encode(query)
    
    for researcher in researchers:
        # 키워드 임베딩
        keyword_embeddings = sentence_model.encode(researcher["keywords"])
        
        # 유사도 계산
        similarities = cosine_similarity([query_embedding], keyword_embeddings)[0]
        max_similarity = np.max(similarities)
        
        # 점수 계산 (유사도 가중치 적용)
        score = max_similarity * 10  # 0~10점 범위로 스케일링
        
        researcher_scores.append((researcher, score))
    
    # 정렬 및 상위 5개 결과 반환
    return sorted(researcher_scores, key=lambda x: x[1], reverse=True)[:5]

def print_results(results):
    """결과 출력 방식 개선"""
    print("\n 검색 결과:")
    if not results:
        print("결과가 없습니다.")
        return
    
    for researcher, score in results:
        print(f"\n {researcher['name']} 박사")
        print(f"  소속: {researcher['affiliation']}")
        print(f" 분야: {researcher['field']} > {researcher['subfield']}")
        print(f" 키워드: {', '.join(researcher['keywords'][:3])}...")
        print(f" 연락처: {researcher['email']}")
        print(f" 매칭 점수: {score:.2f}/10.0")

def main():
    print(" 범용 연구자 검색 시스템")
    print("="*50)
    print("검색어 예시: '메모리 소자', '유전자 편집', '강화학습'")
    
    while True:
        query = input("\n 검색어 입력 (종료: q): ").strip()
        if query.lower() == 'q':
            print("\n검색 시스템을 종료합니다.")
            break
            
        if not query:
            print("검색어를 입력해주세요.")
            continue
            
        results = find_researchers(query)
        print_results(results)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
