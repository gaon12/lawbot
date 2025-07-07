#!/usr/bin/env python3
"""
📜 Streamlit – 법령·판례 Q&A (JSON 미리보기·환각 방지·각주 교정·COT 노출)

설치:  pip install -U streamlit chromadb google-genai python-dotenv
실행:  streamlit run law_qa_app.py
"""
from __future__ import annotations
import os, re, html, json, sqlite3, textwrap, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from dotenv import load_dotenv
from google import genai          # type: ignore
import chromadb                   # type: ignore
from chromadb.config import Settings
from google.genai import types
from google.genai.types import EmbedContentConfig

# ─────────────────── 환경값 ───────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("⚠️  .env 파일에 GEMINI_API_KEY가 필요합니다.")

CHROMA_DIR   = "./data/ChromaLegal"
COL_LAW      = "laws"
COL_PREC     = "precs"
EMBED_MODEL  = "gemini-embedding-exp-03-07"
GEN_MODEL    = "gemini-2.5-flash"
TOP_K        = 8
TEMP         = 0.2
MAX_TOK      = 65536
DB_PATH      = "./data/laws/laws.db"
PRE_LAW_DIR  = Path("./data/laws/preprocess")
PRE_PREC_DIR = Path("./data/precs/preprocess")

# ─────────────────── 클라이언트 ───────────────────
g_client = genai.Client(api_key=API_KEY)

ch_set  = Settings(is_persistent=True,
                   persist_directory=CHROMA_DIR,
                   anonymized_telemetry=False)
ch_cli  = chromadb.PersistentClient(settings=ch_set)
col_law = ch_cli.get_or_create_collection(COL_LAW)
col_prec= ch_cli.get_or_create_collection(COL_PREC)

# ─────────────────── Streamlit ───────────────────
st.set_page_config("📜 법령·판례 Q&A", "📜", layout="wide")
st.markdown("<h1 style='margin-top:0'>📜 법령·판례 기반 Q&A</h1>",
            unsafe_allow_html=True)

# 세션 상태 초기화
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'should_search' not in st.session_state:
    st.session_state.should_search = False

# ─────────────────── 헬퍼 ───────────────────
@st.cache_data(show_spinner=False, ttl="1d")
def embed_text(txt: str) -> List[float]:
    """Gemini 임베딩"""
    try:
        res = g_client.models.embed_content(
            model=EMBED_MODEL,
            contents=txt,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=3072,
            )
        )
        return res.embeddings[0].values  # type: ignore
    except Exception as e:
        st.error(f"임베딩 생성 오류: {e}")
        return []


def _query(col, v) -> List[Dict[str, Any]]:
    if not v:  # 빈 임베딩 처리
        return []
    r = col.query(query_embeddings=[v],
                  n_results=TOP_K,
                  include=["documents", "metadatas", "distances"])
    out = []
    for i in range(len(r["ids"][0])):
        out.append({
            "id"   : r["ids"][0][i],
            "text" : r["documents"][0][i],
            "meta" : r["metadatas"][0][i],
            "score": r["distances"][0][i],
        })
    return out

# ─────────────────── 데이터 매핑 구축 ───────────────────
@st.cache_data(show_spinner=False)
def build_name_map() -> Dict[str, str]:
    """법령·판례 '제목 → parent_id' 매핑"""
    mp: Dict[str, str] = {}
    for p in PRE_LAW_DIR.glob("*.json"):
        try:
            with p.open(encoding="utf-8") as f:
                title = json.load(f).get("title") or p.stem
            mp[title] = p.stem
        except Exception:
            continue
    for p in PRE_PREC_DIR.glob("*.json"):
        try:
            with p.open(encoding="utf-8") as f:
                title = json.load(f).get("title") or p.stem
            mp[title] = p.stem
        except Exception:
            continue
    return mp

NAME2ID = build_name_map()

@st.cache_data(show_spinner=False)
def valid_ids_from_db() -> set[str]:
    """존재하는 parent_id 집합"""
    ids: set[str] = set()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            ids.update(r[0] for r in conn.execute("SELECT law_id FROM laws"))
            ids.update(r[0] for r in conn.execute("SELECT prec_id FROM precedents"))
    except Exception:
        pass
    return ids

VALID_IDS = valid_ids_from_db()

# ─────────────────── 텍스트·JSON 유틸 ───────────────────
def load_json(parent_id: str) -> Dict[str, Any] | None:
    try:
        law_file = PRE_LAW_DIR / f"{parent_id}.json"
        if law_file.exists():
            return json.load(law_file.open("r", encoding="utf-8"))
        prec_file = PRE_PREC_DIR / f"{parent_id}.json"
        if prec_file.exists():
            return json.load(prec_file.open("r", encoding="utf-8"))
    except Exception:
        pass
    return None


def json_to_fulltext(data: Dict[str, Any]) -> str:
    """JSON → 본문 전체 문자열"""
    if "articles" in data:                         # 법령
        return "\n".join(a["text"] for a in data["articles"])
    if "judgement" in data:                        # 판례
        return data["judgement"]
    return ""


def sanitize_html(raw: str) -> str:
    """불필요 HTML 태그 차단"""
    allowed = {"sup", "sub", "span", "br", "b", "strong", "i", "em", "table", "tr", "td", "th", "thead", "tbody"}
    return re.sub(
        r"</?([a-zA-Z0-9]+)[^>]*>",
        lambda m: m.group(0) if m.group(1).lower() in allowed
        else html.escape(m.group(0)),
        raw,
    )


def filter_hallucinated(txt: str) -> str:
    """존재하지 않는 parent_id를 [INVALID-…]로 치환"""
    pat = r"(LAW|PREC)_([0-9]+)"
    def repl(m):
        pid = m.group(2)
        if pid in VALID_IDS:
            return m.group(0)
        return f"[INVALID-{pid}]"
    return re.sub(pat, repl, txt)

# ─────────────────── COT(think 태그) 및 각주 처리 ───────────────────
THINK_RE = re.compile(r"<think(\d+)>(.*?)</think\1>", re.S)
FINALANSWER_RE = re.compile(r"<finalanswer>(.*?)</finalanswer>", re.S)
FOOT_RE  = re.compile(r"\[\^\s*([0-9]+)\s*\]")

def extract_thinking_process(text: str) -> Tuple[str, List[Tuple[int, str]], str]:
    """추론 과정, 최종 답변 분리"""
    # Think 블록들 추출
    think_matches = THINK_RE.findall(text)
    thinking_steps = [(int(num), content.strip()) for num, content in think_matches]
    
    # Final answer 추출
    final_match = FINALANSWER_RE.search(text)
    final_answer = final_match.group(1).strip() if final_match else ""
    
    # 원본에서 태그들 제거한 텍스트
    cleaned = THINK_RE.sub("", text)
    cleaned = FINALANSWER_RE.sub("", cleaned)
    
    return cleaned, thinking_steps, final_answer

def convert_to_html_footnotes(text: str) -> str:
    """각주를 Streamlit 호환 HTML로 변환 (출처 섹션 제외)"""
    # 출처 섹션 찾기
    source_section_match = re.search(r'(## 출처\s*\n.*?)(?=\n##|\Z)', text, re.S)
    
    if source_section_match:
        # 출처 섹션 이전 부분만 변환
        before_sources = text[:source_section_match.start()]
        source_section = source_section_match.group(0)
        after_sources = text[source_section_match.end():]
        
        # 본문의 각주만 HTML로 변환
        def footnote_repl(match):
            num = match.group(1)
            return f'<sup><a href="#footnote-{num}" style="text-decoration:none; color:#007bff;">[{num}]</a></sup>'
        
        converted_before = re.sub(r'\[\^(\d+)\]', footnote_repl, before_sources)
        
        # 출처 섹션은 변환하지 않고 그대로 유지
        return converted_before + source_section + after_sources
    else:
        # 출처 섹션이 없으면 전체 변환
        def footnote_repl(match):
            num = match.group(1)
            return f'<sup><a href="#footnote-{num}" style="text-decoration:none; color:#007bff;">[{num}]</a></sup>'
        
        return re.sub(r'\[\^(\d+)\]', footnote_repl, text)

def render_footnotes_section(footnote_sources: List[Tuple[int, str]]) -> str:
    """각주 섹션을 HTML로 렌더링"""
    if not footnote_sources:
        return ""
    
    footnotes_html = []
    for num, content in footnote_sources:
        footnotes_html.append(
            f'<div id="footnote-{num}" style="margin-bottom: 8px;">'
            f'<sup><strong>[{num}]</strong></sup> '
            f'<span style="margin-left: 4px;">{content}</span>'
            f'</div>'
        )
    return '<div style="border-top: 1px solid #ddd; margin-top: 20px; padding-top: 15px;">' + \
           '<h4>📚 각주</h4>' + \
           ''.join(footnotes_html) + '</div>'

def extract_footnote_sources(text: str) -> List[Tuple[int, str]]:
    """각주 소스 추출 (개선된 버전)"""
    sources = []
    
    # ## 출처 섹션 찾기
    source_section_match = re.search(r'## 출처\s*\n(.*?)(?=\n##|\n\*\*|\Z)', text, re.S)
    if not source_section_match:
        return sources
    
    source_content = source_section_match.group(1)
    lines = source_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('[^') and ']' in line:
            # 각주 시작
            match = re.match(r'\[\^(\d+)\]\s*(.*)', line)
            if match:
                num = int(match.group(1))
                content = match.group(2)
                sources.append((num, content))
    
    return sources

def is_reasoning_model_response(text: str) -> bool:
    """추론 모델의 응답인지 확인"""
    return bool(THINK_RE.search(text) or FINALANSWER_RE.search(text))

def generate_multi_turn_reasoning(query: str, docs: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, str]]]:
    """Gemini 공식 멀티턴 API를 사용한 체계적 추론"""
    thinking_steps = []
    
    try:
        # 채팅 세션 생성
        chat = g_client.chats.create(model=GEN_MODEL)
        
        # Turn 1: 질문 분석 및 쟁점 파악
        step1_prompt = f"""
        질문: {query}
        
        다음을 수행하세요:
        1. 핵심 쟁점 3개 이내로 정리
        2. 필요한 법령 영역 식별
        3. 추가 검토 필요 사항 명시
        
        간단명료하게 답변하세요.
        """
        
        step1_resp = chat.send_message(step1_prompt)
        step1_text = step1_resp.text if step1_resp.text is not None else "쟁점 분석을 완료할 수 없습니다."
        thinking_steps.append((1, "쟁점 분석: " + step1_text))
        
        # Turn 2: 법령 분석
        step2_prompt = f"""
        관련 자료: {docs[:4]}  
        
        위 관련 자료를 바탕으로 법령을 상세 분석하세요:
        1. 적용 조문 확인
        2. 구성요건 분석
        3. 해석상 쟁점 정리
        """
        
        step2_resp = chat.send_message(step2_prompt)
        step2_text = step2_resp.text if step2_resp.text is not None else "법령 분석을 완료할 수 없습니다."
        thinking_steps.append((2, "법령 분석: " + step2_text))
        
        # Turn 3: 판례 분석
        prec_docs = [d for d in docs if 'PREC_' in d.get('id', '')]
        step3_prompt = f"""
        판례 자료: {prec_docs}
        
        판례를 분석하여:
        1. 유사 사안 판례 식별
        2. 판시 요지 정리
        3. 구별되는 사실관계 검토
        """
        
        step3_resp = chat.send_message(step3_prompt)
        step3_text = step3_resp.text if step3_resp.text is not None else "판례 분석을 완료할 수 없습니다."
        thinking_steps.append((3, "판례 분석: " + step3_text))
        
        # Turn 4: 종합 결론
        final_prompt = f"""
        지금까지의 분석을 종합하여 최종 답변을 작성하세요.
        
        반드시 다음 구조를 준수하세요:
        
        【3줄 요약】
        1. [핵심 쟁점 한 줄]
        2. [적용 법령/판례 한 줄] 
        3. [결론 방향 한 줄]

        **I. 쟁점 정리**
        **II. 관련 법령 분석**  
        **III. 판례 검토**
        **IV. 법리적 분석**
        
        【예상 결과/형량】
        - 구체적 예상 결과 제시
        
        ## 출처
        [^1] 법령명/판례 + 핵심 내용
        [^2] 법령명/판례 + 핵심 내용
        """
        
        final_resp = chat.send_message(final_prompt)
        final_text = final_resp.text if final_resp.text is not None else "최종 답변을 생성할 수 없습니다."
        thinking_steps.append((4, "최종 답변 생성 완료"))
        
        return final_text, thinking_steps
        
    except Exception as e:
        error_text = f"멀티턴 추론 중 오류 발생: {str(e)}"
        thinking_steps.append((1, error_text))
        return error_text, thinking_steps

# ─────────────────── 검색 & 프롬프트 ───────────────────
def retrieve(q: str) -> List[Dict[str, Any]]:
    """① 명시적 이름 호출 → 원문, ② 임베딩 검색"""
    explicit_docs: List[Dict[str, Any]] = []
    for name, pid in NAME2ID.items():
        if name in q:
            data = load_json(pid)
            if data:
                explicit_docs.append({
                    "id": pid,
                    "text": json_to_fulltext(data),
                    "meta": {"parent_id": pid, "title": name},
                    "score": 0.0,
                })

    v    = embed_text(q)
    docs = _query(col_law, v) + _query(col_prec, v)
    docs = explicit_docs + docs
    docs.sort(key=lambda d: d["score"])
    return docs[:TOP_K]

def get_additional_documents(requested_laws: List[str]) -> List[Dict[str, Any]]:
    """LLM이 요청한 추가 법령/판례 전체 제공"""
    additional_docs = []
    for law_name in requested_laws:
        if law_name in NAME2ID:
            pid = NAME2ID[law_name]
            data = load_json(pid)
            if data:
                additional_docs.append({
                    "id": pid,
                    "text": json_to_fulltext(data),
                    "meta": {"parent_id": pid, "title": law_name},
                    "score": 0.0,
                })
    return additional_docs

def build_prompt(q: str, docs: List[Dict[str, Any]]) -> Tuple[str, str]:
    """LLM에 넣을 컨텍스트 구성"""
    ctx, id_set = [], set[str]()
    for i, d in enumerate(docs, 1):
        title = d["meta"].get("title", d["id"])
        ctx.append(f"[{i}] ({title}) {d['text']}")
        id_set.add(d["meta"]["parent_id"])
    joined = "\n\n".join(ctx)
    return f"[질문]\n{q}\n\n[자료]\n{joined}", ", ".join(sorted(id_set))

# ─────────────────── 시스템 프롬프트 ───────────────────
SYSTEM_PROMPT_REASONING = textwrap.dedent("""
    너는 한국의 **법령과 판례**를 동시에 참고하여 법률 질문에 답하는 전문 AI이다.

    ## 답변 구조 (필수 준수):
    **【3줄 요약】**
    1. [핵심 쟁점 한 줄]
    2. [적용 법령/판례 한 줄] 
    3. [결론 방향 한 줄]

    **I. 쟁점 정리**
    **II. 관련 법령 분석**
    **III. 판례 검토**  
    **IV. 법리적 분석**
    
    **【예상 결과/형량】**
    - 민사: 손해배상액, 승소 가능성 등
    - 형사: 예상 형량, 기소 가능성 등
    - 행정: 처분 적법성, 구제 방법 등

    ## 서식 및 구조화 지침:
    1) **하위 제목 활용**: 각 분석 단계에서 ### 소제목 사용
       예: ### 구성요건 검토, ### 위법성 판단, ### 책임 검토

    2) **표 활용 (비교 시)**: 
       - 법령 vs 판례 비교
       - 구성요건별 해당 여부
       - 유사 판례 비교 분석

    3) **리스트 구조**: 복합적 쟁점은 번호나 불릿 포인트 활용

    ## 단계별 추론 과정 (필수):
    1) **반드시** 답변 전에 단계별 추론을 수행한다:
       - <think1>질문 분석 및 쟁점 파악</think1>
       - <think2>관련 법령 및 판례 검토</think2>
       - <think3>법리적 분석 및 해석</think3>
       - <think4>결론 도출 및 검증</think4>
       - 필요시 <think5>, <think6> 등 추가 단계 수행
    
    2) 모든 추론이 완료된 후 **최종 답변**을 작성한다:
       <finalanswer>
       [위의 답변 구조에 따른 완전한 답변]
       </finalanswer>

    ## 각주 작성 규칙:
    - 각주는 [^1], [^2] 형태로 작성
    - 각주 내용은 답변 마지막 '## 출처' 섹션에 정리
    - 각주 번호는 1부터 순차적으로 증가
    - 각 각주 뒤에는 반드시 줄바꿈 추가

    ## 주의사항:
    - 법률 자문이 아닌 일반적 정보 제공임을 필요시 명시
    - 제공된 [자료]의 범위에서만 인용
    - 확실하지 않은 내용은 추측하지 말고 한계를 명시
""")

SYSTEM_PROMPT_STEP1 = textwrap.dedent("""
    너는 한국의 **법령과 판례**를 참고하여 법률 질문을 분석하는 AI이다.
    
    주어진 질문과 자료를 바탕으로 다음을 수행하라:
    
    1. **질문의 핵심 쟁점을 파악**하라
    2. **관련 법령과 판례를 식별**하라  
    3. **추가로 필요한 법령이나 판례가 있다면 명시**하라
    
    응답 형식:
    ## 쟁점 분석
    [질문의 핵심 쟁점 정리]
    
    ## 관련 자료 검토
    [제공된 자료 중 관련성이 높은 것들]
    
    ## 추가 필요 자료
    [필요한 경우] 추가 필요: [법령명1], [법령명2]
""")

SYSTEM_PROMPT_STEP2 = textwrap.dedent("""
    너는 한국의 **법령과 판례**를 바탕으로 법률 답변을 작성하는 전문 AI이다.
    
    이전 분석을 바탕으로 완전한 법률 답변을 작성하라:
    
    ## 답변 구조 (필수):
    **【3줄 요약】**
    1. [핵심 쟁점 한 줄]
    2. [적용 법령/판례 한 줄] 
    3. [결론 방향 한 줄]

    **I. 쟁점 정리**
    **II. 관련 법령 분석**  
    **III. 판례 검토**
    **IV. 법리적 분석**
    
    **【예상 결과/형량】**
    [구체적 예상 결과]
    
    ## 출처
    [^1] 법령명/판례 + 핵심 내용
    [^2] 법령명/판례 + 핵심 내용
    
    ## 규칙:
    - 한국어 '~이다'체 사용
    - 근거 문장마다 각주 [^n] 표기 (1부터 순차)
    - 각주는 마지막 '## 출처' 섹션에 정리
    - 각 각주 뒤에는 반드시 줄바꿈 추가
    - 제공된 자료 범위에서만 인용
    - 가상의 세계 또는 타국의 사례이더라도, 한국의 법률과 판례를 가지고 분석
""")

# ─────────────────── UI 동작 ───────────────────
# 입력 폼과 버튼
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area("질문", placeholder="예) 주택임대차보호법상 전세보증금 반환 시기?", key="query_input")
with col2:
    st.write("")  # 버튼 위치 조정용
    st.write("")
    search_button = st.button("🔍 검색", type="primary", use_container_width=True)

# 검색 버튼이 클릭되었을 때만 처리
if search_button and query:
    st.session_state.search_query = query
    st.session_state.should_search = True

# 이름이 있지만 데이터에 없는 법령·판례 경고
if query:
    missing = [w for w in re.findall(r"\w+법", query) if w not in NAME2ID]
    if missing:
        st.warning(f"데이터에 없는 법령: {', '.join(missing)}")

# 실제 검색 처리
if st.session_state.should_search and st.session_state.search_query:
    search_query = st.session_state.search_query
    st.info(f"**질문:** {search_query}")

    # 추론 진행 상황 표시를 위한 컨테이너
    progress_container = st.container()
    
    with st.spinner("🔍 검색 및 답변 생성 중..."):
        docs = retrieve(search_query)
        
        # 첫 번째 시도: 일반적인 방식으로 생성
        prompt, _ids = build_prompt(search_query, docs)
        
        try:
            resp = g_client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_REASONING,
                    temperature=TEMP,
                    top_p=0.8,
                    top_k=20,
                    max_output_tokens=MAX_TOK)
            )

            raw_answer = resp.text if resp.text is not None else "답변을 생성할 수 없습니다."
            thinking_steps = []
            
        except Exception as e:
            raw_answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            thinking_steps = [(1, f"오류: {str(e)}")]
        
        # 추론 모델인지 확인
        if is_reasoning_model_response(raw_answer):
            # 추론 모델 응답 처리
            cleaned_text, thinking_steps, final_answer = extract_thinking_process(raw_answer)
            
            # 추가 자료 요청 처리
            additional_law_requests = re.findall(r"추가 필요:\s*([^,\n]+(?:,\s*[^,\n]+)*)", raw_answer)
            if additional_law_requests:
                requested_laws = [law.strip() for request in additional_law_requests 
                                for law in request.split(',')]
                additional_docs = get_additional_documents(requested_laws)
                if additional_docs:
                    docs.extend(additional_docs)
                    thinking_steps.append((len(thinking_steps) + 1, f"추가 자료 확보: {', '.join(requested_laws)}"))
                    
                    # 추가 자료로 재생성
                    prompt, _ids = build_prompt(search_query, docs)
                    try:
                        resp = g_client.models.generate_content(
                            model=GEN_MODEL,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT_REASONING,
                                temperature=TEMP,
                                top_p=0.8,
                                top_k=20,
                                max_output_tokens=MAX_TOK)
                        )
                        raw_answer = resp.text if resp.text is not None else "재생성 실패"
                        cleaned_text, new_thinking_steps, final_answer = extract_thinking_process(raw_answer)
                        thinking_steps.extend(new_thinking_steps)
                    except Exception as e:
                        thinking_steps.append((len(thinking_steps) + 1, f"재생성 오류: {str(e)}"))

            # 최종 답변 처리
            body = final_answer if final_answer else cleaned_text
            # 각주 소스는 원본에서 추출
            footnote_sources = extract_footnote_sources(raw_answer)
            
        else:
            # 비추론 모델: 멀티턴 추론 생성
            st.info("🔄 단계별 분석을 진행합니다...")
            body, thinking_steps = generate_multi_turn_reasoning(search_query, docs)
            raw_answer = body  # 원본 보존
            # 각주 소스는 원본에서 추출
            footnote_sources = extract_footnote_sources(raw_answer)
            
        # 공통 후처리
        body = convert_to_html_footnotes(body)
        body = sanitize_html(body)
        body = filter_hallucinated(body)

        reasoning_summary = ""
        try:
            reasoning_summary = (resp.candidates[0].citation_metadata.summary
                               if resp.candidates
                               and resp.candidates[0].citation_metadata else "")
        except:
            pass

    # ───────────────── 출력 ─────────────────
    st.markdown("### 📋 답변", unsafe_allow_html=True)
    st.markdown(body, unsafe_allow_html=True)
    
    # 각주 처리 - 개선된 방식
    if footnote_sources:
        footnotes_html = render_footnotes_section(footnote_sources)
        st.markdown(footnotes_html, unsafe_allow_html=True)

    # 추론 과정 표시
    if thinking_steps:
        with st.expander("🧠 AI 추론 과정 (접기/펼치기)"):
            for step_num, content in thinking_steps:
                st.markdown(f"### 🔄 {step_num}차 추론 완료")
                st.markdown(sanitize_html(content), unsafe_allow_html=True)
                st.markdown("---")
                
        if reasoning_summary:
            st.write("**추론 요약:**")
            st.write(reasoning_summary)

    # ── 참고 문서 (JSON 미리보기) ──
    with st.expander("🔍 참고 문서 보기"):
        for i, d in enumerate(docs, 1):
            pid = d["meta"]["parent_id"]
            title = d["meta"].get("title", pid)
            snippet = d["text"][:400].replace("\n", " ")

            if st.button(f"📄 [{i}] {title}", key=f"btn_{pid}"):
                st.session_state[f"show_{pid}"] = not st.session_state.get(
                    f"show_{pid}", False)

            if st.session_state.get(f"show_{pid}", False):
                data = load_json(pid)
                if data:
                    st.json(data, expanded=False)
                else:
                    st.warning("원본 JSON을 찾을 수 없습니다.")

            st.markdown(f"<span style='font-size:0.9rem;color:#666'>{snippet}...</span>",
                        unsafe_allow_html=True)

    # 검색 완료 후 플래그 초기화
    st.session_state.should_search = False

# ─────────────────── 푸터 ───────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem'>"
    "Made with Streamlit · Gemini · ChromaDB.<br/>"
    "AI can make mistakes. Please verify the information with a reliable source."
    "</div>",
    unsafe_allow_html=True
)