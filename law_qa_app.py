#!/usr/bin/env python3
"""
📜 Streamlit – 법령·판례 Q&A (개선된 버전)
- Think/Non-think 모드 (Gemini 2.5 Pro thinking API 활용)
- 간단/상세 입력 모드  
- 개선된 각주 렌더링
- 동적 법령/판례 요청 기능
- 판례 데이터 전달 개선

설치:  pip install -U streamlit chromadb google-genai python-dotenv
실행:  streamlit run law_qa_app.py
"""
from __future__ import annotations
import os, re, html, json, sqlite3, textwrap, time, asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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
GEN_MODEL_THINK = "gemini-2.5-flash"  # thinking API 지원
GEN_MODEL_NORMAL = "gemini-2.5-flash"
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

# ─────────────────── Streamlit 설정 ───────────────────
st.set_page_config("📜 법령·판례 Q&A", "📜", layout="wide")

# 사이드바 설정
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    
    # Think 모드 설정
    think_mode = st.selectbox(
        "🧠 추론 모드",
        options=["Think 모드", "Non-Think 모드"],
        help="Think 모드: Gemini 2.5 Flash + 상세 추론 과정\nNon-Think 모드: Gemini 2.5 Flash + 빠른 답변"
    )
    
    # 입력 모드 설정
    input_mode = st.selectbox(
        "📝 입력 모드", 
        options=["간단 입력", "상세 입력"],
        help="간단 입력: 자유 텍스트\n상세 입력: 구조화된 양식"
    )
    
    # 동적 검색 설정
    dynamic_search = st.checkbox(
        "🔄 동적 검색",
        value=True,
        help="AI가 추론 중 추가 법령/판례를 요청할 수 있도록 허용"
    )
    
    st.markdown("---")
    st.markdown("### 📊 통계")
    
    # DB 통계 표시
    try:
        with sqlite3.connect(DB_PATH) as conn:
            law_count = conn.execute("SELECT COUNT(*) FROM laws").fetchone()[0]
            prec_count = conn.execute("SELECT COUNT(*) FROM precedents").fetchone()[0]
            st.metric("법령 수", f"{law_count:,}")
            st.metric("판례 수", f"{prec_count:,}")
    except:
        st.error("DB 연결 실패")

# 메인 타이틀
st.markdown("<h1 style='margin-top:0'>📜 법령·판례 기반 Q&A</h1>", unsafe_allow_html=True)

# 세션 상태 초기화
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'should_search' not in st.session_state:
    st.session_state.should_search = False
if 'thinking_requests' not in st.session_state:
    st.session_state.thinking_requests = []

# ─────────────────── 헬퍼 함수 ───────────────────
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

def _query(col, v, n_results: int = TOP_K) -> List[Dict[str, Any]]:
    if not v:
        return []
    try:
        r = col.query(query_embeddings=[v],
                      n_results=n_results,
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
    except Exception as e:
        st.error(f"검색 오류: {e}")
        return []

# ─────────────────── 데이터 매핑 구축 ───────────────────
@st.cache_data(show_spinner=False)
def build_name_map() -> Dict[str, str]:
    """법령·판례 '제목 → parent_id' 매핑"""
    mp: Dict[str, str] = {}
    for p in PRE_LAW_DIR.glob("*.json"):
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
                title = data.get("title") or p.stem
                mp[title] = data.get("law_id", p.stem)
        except Exception:
            continue
    for p in PRE_PREC_DIR.glob("*.json"):
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
                title = data.get("title") or p.stem
                mp[title] = data.get("prec_id", p.stem)
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
    content = data.get("content", "")
    if content:
        return content
    # 백업: 다른 필드에서 텍스트 추출
    if "articles" in data:
        return "\n".join(a.get("text", "") for a in data["articles"])
    if "judgement" in data:
        return data["judgement"]
    return ""

def is_law_document(doc_id: str, doc_meta: Dict[str, Any]) -> bool:
    """문서가 법령인지 판례인지 정확히 판단"""
    # 1. ID 기반 판단 (가장 확실)
    if doc_id.startswith("PREC_") or doc_id.startswith("prec_"):
        return False
    if doc_id.startswith("LAW_") or doc_id.startswith("law_"):
        return True
    
    # 2. 메타데이터 기반 판단
    doc_type = doc_meta.get("type", "").lower()
    if doc_type in ["precedent", "판례", "prec"]:
        return False
    if doc_type in ["law", "법령", "법률"]:
        return True
    
    # 3. parent_id 기반 판단
    parent_id = doc_meta.get("parent_id", "")
    if parent_id.startswith("PREC_") or parent_id.startswith("prec_"):
        return False
    if parent_id.startswith("LAW_") or parent_id.startswith("law_"):
        return True
    
    # 4. 제목 기반 판단
    title = doc_meta.get("title", "")
    if "판례" in title or "대법원" in title or "헌법재판소" in title:
        return False
    if title.endswith("법") or title.endswith("령") or title.endswith("규칙"):
        return True
    
    # 5. 내용 기반 판단 (JSON 데이터 확인)
    data = load_json(parent_id or doc_id)
    if data:
        if "judgment" in data or "judgement" in data or "court" in data:
            return False
        if "articles" in data or "provisions" in data:
            return True
    
    # 기본값: 법령으로 간주
    return True

def fix_markdown_bold(text: str) -> str:
    """마크다운 볼드체 문제 해결 - 개선된 버전"""
    def bold_replacer(match):
        content = match.group(1).strip()
        if content:
            return f'**{content}**'
        return match.group(0)
    
    # **내용** 패턴 매칭 (특수문자 포함)
    text = re.sub(r'\*\*([^*\n]+?)\*\*', bold_replacer, text)
    return text

def sanitize_html(raw: str) -> str:
    """불필요 HTML 태그 차단 및 마크다운 볼드체 수정"""
    allowed = {"sup", "sub", "span", "br", "b", "strong", "i", "em", "table", "tr", "td", "th", "thead", "tbody", "a"}
    sanitized = re.sub(
        r"</?([a-zA-Z0-9]+)[^>]*>",
        lambda m: m.group(0) if m.group(1).lower() in allowed
        else html.escape(m.group(0)),
        raw,
    )
    return fix_markdown_bold(sanitized)

def filter_hallucinated(txt: str) -> str:
    """존재하지 않는 parent_id를 [INVALID-…]로 치환"""
    pat = r"(LAW|PREC)_([0-9]+)"
    def repl(m):
        pid = m.group(2)
        if pid in VALID_IDS:
            return m.group(0)
        return f"[INVALID-{pid}]"
    return re.sub(pat, repl, txt)

# ─────────────────── 각주 처리 개선 ───────────────────
def convert_to_html_footnotes(text: str) -> str:
    """각주를 Streamlit 호환 HTML로 변환"""
    def footnote_repl(match):
        num = match.group(1)
        return f'<sup><a href="#footnote-{num}" onclick="scrollToFootnote({num})" style="text-decoration:none; color:#1f77b4; font-weight:bold; cursor:pointer;">[{num}]</a></sup>'
    
    return re.sub(r'\[\^(\d+)\]', footnote_repl, text)

def extract_footnote_sources(text: str) -> List[Tuple[int, str]]:
    """각주 소스 추출"""
    sources = []
    source_section_match = re.search(r'## 출처\s*\n(.*?)(?=\n##|\Z)', text, re.S)
    if not source_section_match:
        return sources
    
    source_content = source_section_match.group(1)
    lines = source_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('[^') and ']' in line:
            match = re.match(r'\[\^(\d+)\]\s*(.*)', line)
            if match:
                num = int(match.group(1))
                content = match.group(2)
                sources.append((num, content))
    
    return sources

def render_footnotes_section(footnote_sources: List[Tuple[int, str]]) -> str:
    """각주 섹션을 HTML로 렌더링"""
    if not footnote_sources:
        return ""
    
    footnotes_html = []
    for num, content in footnote_sources:
        footnotes_html.append(
            f'<div id="footnote-{num}" style="margin-bottom: 15px; padding: 12px; background-color: #f8f9fa; border-left: 4px solid #1f77b4; border-radius: 6px; line-height: 1.6;">'
            f'<div style="margin-bottom: 8px;">'
            f'<sup><strong style="color: #1f77b4; font-size: 14px;">[{num}]</strong></sup> '
            f'<a href="#footnote-ref-{num}" onclick="scrollToFootnoteRef({num})" style="float: right; color: #666; text-decoration: none; font-size: 12px; cursor: pointer;">↑ 돌아가기</a>'
            f'</div>'
            f'<div style="margin-left: 8px; color: #333;">{content}</div>'
            f'</div>'
        )
    
    scroll_js = """
    <script>
    function scrollToFootnote(num) {
        const element = document.getElementById('footnote-' + num);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            element.style.backgroundColor = '#e3f2fd';
            setTimeout(() => { element.style.backgroundColor = '#f8f9fa'; }, 2000);
        }
    }
    
    function scrollToFootnoteRef(num) {
        const refs = document.querySelectorAll('a[href="#footnote-' + num + '"]');
        if (refs.length > 0) {
            refs[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
            refs[0].style.backgroundColor = '#fff3cd';
            setTimeout(() => { refs[0].style.backgroundColor = 'transparent'; }, 2000);
        }
    }
    </script>
    """
    
    return (
        '<div style="border-top: 2px solid #ddd; margin-top: 30px; padding-top: 25px;">'
        '<h3 style="color: #1f77b4; margin-bottom: 20px; font-size: 20px; border-bottom: 1px solid #ddd; padding-bottom: 10px;">📚 참고 자료</h3>'
        + ''.join(footnotes_html) + 
        '</div>' + scroll_js
    )

def remove_source_section(text: str) -> str:
    """## 출처 섹션 제거"""
    return re.sub(r'## 출처\s*\n.*?(?=\n##|\Z)', '', text, flags=re.S).strip()

# ─────────────────── 동적 검색 기능 ───────────────────
def search_specific_law_or_precedent(query: str, doc_type: str = "both") -> List[Dict[str, Any]]:
    """특정 법령이나 판례 검색"""
    v = embed_text(query)
    results = []
    
    if doc_type in ["law", "both"]:
        law_results = _query(col_law, v, n_results=3)
        for doc in law_results:
            doc["meta"]["type"] = "law"
        results.extend(law_results)
    
    if doc_type in ["precedent", "both"]:
        prec_results = _query(col_prec, v, n_results=3)
        for doc in prec_results:
            doc["meta"]["type"] = "precedent"
        results.extend(prec_results)
    
    return sorted(results, key=lambda d: d["score"])[:5]

def process_thinking_requests(thinking_text: str) -> str:
    """추론 과정에서 추가 자료 요청 처리"""
    if not st.session_state.get('dynamic_search', True):
        return thinking_text
    
    # 요청 패턴 찾기
    request_patterns = [
        r"(?:추가로|더|또한)\s*(?:필요한|관련된|찾아보고\s*싶은)\s*(?:법령|판례|자료):\s*(.+?)(?:\n|$)",
        r"(?:검색|찾기)\s*요청:\s*(.+?)(?:\n|$)",
        r"(?:더\s*자세한|구체적인)\s*(?:법령|판례)(?:가|을)\s*(?:필요로|원함):\s*(.+?)(?:\n|$)"
    ]
    
    enhanced_thinking = thinking_text
    
    for pattern in request_patterns:
        matches = re.findall(pattern, thinking_text, re.IGNORECASE)
        for match in matches:
            query = match.strip()
            if len(query) > 5:  # 의미있는 쿼리만 처리
                additional_docs = search_specific_law_or_precedent(query)
                if additional_docs:
                    doc_summary = "\n".join([
                        f"- {doc['meta'].get('title', doc['id'])[:100]}..."
                        for doc in additional_docs[:3]
                    ])
                    enhanced_thinking += f"\n\n[추가 검색 결과: {query}]\n{doc_summary}"
    
    return enhanced_thinking

# ─────────────────── 검색 & 프롬프트 ───────────────────
def retrieve(q: str) -> List[Dict[str, Any]]:
    """개선된 검색: 법령과 판례 균형있게 가져오기"""
    # 1. 명시적 이름 호출
    explicit_docs: List[Dict[str, Any]] = []
    for name, pid in NAME2ID.items():
        if name in q:
            data = load_json(pid)
            if data:
                explicit_docs.append({
                    "id": pid,
                    "text": json_to_fulltext(data),
                    "meta": {"parent_id": pid, "title": name, "type": "explicit"},
                    "score": 0.0,
                })

    # 2. 임베딩 검색 - 법령과 판례 별도로
    v = embed_text(q)
    law_docs = _query(col_law, v, n_results=6)
    prec_docs = _query(col_prec, v, n_results=6)
    
    # 타입 정보 추가
    for doc in law_docs:
        doc["meta"]["type"] = "law"
    for doc in prec_docs:
        doc["meta"]["type"] = "precedent"
    
    # 결합 및 정렬
    all_docs = explicit_docs + law_docs + prec_docs
    all_docs.sort(key=lambda d: d["score"])
    
    return all_docs[:TOP_K]

def build_prompt_improved(q: str, docs: List[Dict[str, Any]]) -> str:
    """개선된 프롬프트 구성 - 법령과 판례 구분"""
    law_docs = [d for d in docs if is_law_document(d["id"], d["meta"])]
    prec_docs = [d for d in docs if not is_law_document(d["id"], d["meta"])]
    
    # 법령 섹션
    law_ctx = []
    for i, d in enumerate(law_docs, 1):
        title = d["meta"].get("title", d["id"])
        law_ctx.append(f"[법령{i}] {title}\n{d['text'][:2000]}")  # 길이 제한
    
    # 판례 섹션  
    prec_ctx = []
    for i, d in enumerate(prec_docs, 1):
        title = d["meta"].get("title", d["id"])
        prec_ctx.append(f"[판례{i}] {title}\n{d['text'][:2000]}")  # 길이 제한
    
    law_section = "\n\n".join(law_ctx) if law_ctx else "관련 법령 없음"
    prec_section = "\n\n".join(prec_ctx) if prec_ctx else "관련 판례 없음"
    
    dynamic_note = ""
    if st.session_state.get('dynamic_search', True):
        dynamic_note = "\n\n[참고] 추론 중 추가 법령이나 판례가 필요하면 '추가 필요한 법령: [법령명]' 또는 '추가 필요한 판례: [키워드]' 형태로 요청할 수 있습니다."
    
    return f"""[질문]
{q}

[관련 법령]
{law_section}

[관련 판례]  
{prec_section}{dynamic_note}"""

# ─────────────────── 시스템 프롬프트 ───────────────────
SYSTEM_PROMPT_THINK = textwrap.dedent("""
    너는 한국의 **법령과 판례**를 동시에 참고하여 법률 질문에 답하는 전문 AI이다.
    
    ## 상세한 추론 과정 (단계 제한 없음):
    각 단계별로 깊이 있게 분석하며, 필요한 만큼 단계를 늘려도 된다:
    
    1) **질문 분석**: 법적 쟁점, 핵심 키워드, 관련 법 영역 파악
    2) **법령 검토**: 제공된 법령의 조문별 상세 분석
    3) **판례 분석**: 판례의 사실관계, 법리, 판시사항 검토
    4) **법리적 종합**: 법령과 판례 간 관계, 해석론, 적용 기준
    5) **사안 적용**: 구체적 사실에 법리 적용
    6) **결론 도출**: 최종 법적 판단 및 근거
    7) **추가 고려사항**: 예외, 주의점, 실무상 고려사항
    
    추론 중 더 구체적인 법령이나 판례가 필요하면 요청할 수 있다.

    ## 최종 답변 구조:
    **【핵심 결론】**
    - 3줄 요약으로 핵심 결론 제시

    **I. 사안의 정리**
    **II. 관련 법령 분석**
    **III. 판례 검토**  
    **IV. 법리적 분석**
    **V. 결론 및 실무 조치**
    
    ## 규칙:
    - 제공된 자료를 최대한 활용하되, 부족하면 추가 요청
    - 근거가 있는 내용만 단정적으로 기술
    - 각주 [^n]로 출처 명시
    - 실무적 조치방안까지 제시
""")

SYSTEM_PROMPT_NON_THINK = textwrap.dedent("""
    너는 한국의 **법령과 판례**를 동시에 참고하여 법률 질문에 답하는 전문 AI이다.

    ## 답변 구조:
    **【핵심 결론】**
    - 3줄 요약으로 핵심 결론 제시

    **I. 사안의 정리**
    **II. 관련 법령 분석**
    **III. 판례 검토**  
    **IV. 법리적 분석**
    **V. 결론 및 실무 조치**

    ## 규칙:
    - 제공된 [관련 법령]과 [관련 판례] 자료를 **반드시** 활용
    - 근거 문장마다 각주 [^n] 표기 (1부터 순차)
    - 확실하지 않은 내용은 추측하지 말고 한계를 명시
    - 실무적 조치방안까지 제시
""")

# ─────────────────── Think 모드 처리 (Gemini 2.5 Flash API) ───────────────────
def generate_with_thinking(prompt: str, system_prompt: str) -> Tuple[str, str]:
    """Gemini 2.5 Flash thinking API를 사용한 생성"""
    try:
        thoughts = ""
        answer = ""
        
        # Thinking API 사용
        for chunk in g_client.models.generate_content_stream(
            model=GEN_MODEL_THINK,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMP,
                top_p=0.8,
                top_k=20,
                max_output_tokens=MAX_TOK,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                )
            )
        ):
            for part in chunk.candidates[0].content.parts:
                if not part.text:
                    continue
                elif part.thought:
                    thoughts += part.text
                else:
                    answer += part.text
        
        # 추론 과정에서 추가 자료 요청 처리
        if st.session_state.get('dynamic_search', True):
            enhanced_thoughts = process_thinking_requests(thoughts)
            if enhanced_thoughts != thoughts:
                thoughts = enhanced_thoughts
        
        return thoughts.strip(), answer.strip()
        
    except Exception as e:
        return f"추론 과정 오류: {str(e)}", f"답변 생성 중 오류가 발생했습니다: {str(e)}"

def generate_normal(prompt: str, system_prompt: str) -> str:
    """일반 생성 (Flash 모델)"""
    try:
        resp = g_client.models.generate_content(
            model=GEN_MODEL_NORMAL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMP,
                top_p=0.8,
                top_k=20,
                max_output_tokens=MAX_TOK
            )
        )
        return resp.text if resp.text is not None else "답변을 생성할 수 없습니다."
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# ─────────────────── UI 입력 부분 ───────────────────
if input_mode == "간단 입력":
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_area("질문", placeholder="예) 주택임대차보호법상 전세보증금 반환 시기?", key="query_input", height=100)
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 검색", type="primary", use_container_width=True)
        
else:
    # 상세 입력 모드
    st.markdown("### 📝 상세 사례 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        when_info = st.text_input("⏰ 언제", placeholder="2024년 1월 15일, 계약 만료 1개월 전")
        where_info = st.text_input("📍 어디서", placeholder="서울시 강남구 OO아파트")
    
    with col2:
        who_info = st.text_input("👥 누가", placeholder="임차인 A, 임대인 B")
        what_info = st.text_input("❓ 무엇을", placeholder="전세보증금 반환 요구")
    
    main_question = st.text_area("📋 상세 상황 및 질문", 
                                placeholder="계약 만료 후 보증금을 돌려받지 못하고 있습니다. 어떤 법적 조치를 취할 수 있나요?",
                                height=120)
    
    other_info = st.text_area("📎 기타 참고사항", 
                             placeholder="특약사항, 관련 서류, 기타 중요 정보",
                             height=80)
    
    # 구조화된 질문 생성
    query_parts = []
    if when_info: query_parts.append(f"시점: {when_info}")
    if where_info: query_parts.append(f"장소: {where_info}")
    if who_info: query_parts.append(f"당사자: {who_info}")
    if what_info: query_parts.append(f"행위: {what_info}")
    if main_question: query_parts.append(f"질문: {main_question}")
    if other_info: query_parts.append(f"기타: {other_info}")
    
    query = "\n".join(query_parts)
    
    search_button = st.button("🔍 검색", type="primary", use_container_width=True)

# 검색 버튼 처리
if search_button and query.strip():
    st.session_state.search_query = query
    st.session_state.should_search = True
    st.session_state.thinking_requests = []

# 이름이 있지만 데이터에 없는 법령·판례 경고
if query:
    missing = [w for w in re.findall(r"\w+법", query) if w not in NAME2ID]
    if missing:
        st.warning(f"데이터에 없는 법령: {', '.join(missing)}")

# ─────────────────── 검색 및 답변 생성 ───────────────────
if st.session_state.should_search and st.session_state.search_query:
    search_query = st.session_state.search_query
    st.info(f"**질문:** {search_query[:200]}..." if len(search_query) > 200 else f"**질문:** {search_query}")

    with st.spinner("🔍 검색 및 답변 생성 중..."):
        docs = retrieve(search_query)
        
        # 검색 결과 요약 표시 (정확한 분류)
        law_docs = [d for d in docs if is_law_document(d["id"], d["meta"])]
        prec_docs = [d for d in docs if not is_law_document(d["id"], d["meta"])]
        
        st.info(f"🔍 검색 완료: 법령 {len(law_docs)}건, 판례 {len(prec_docs)}건")
        
        prompt = build_prompt_improved(search_query, docs)
        
        # Think 모드에 따른 생성
        is_think_mode = think_mode.startswith("Think")
        system_prompt = SYSTEM_PROMPT_THINK if is_think_mode else SYSTEM_PROMPT_NON_THINK
        
        if is_think_mode:
            thinking_process, final_answer = generate_with_thinking(prompt, system_prompt)
            body = final_answer
        else:
            body = generate_normal(prompt, system_prompt)
            thinking_process = ""
        
        # 후처리
        body = remove_source_section(body)
        footnote_sources = extract_footnote_sources(body)
        body = convert_to_html_footnotes(body)
        body = sanitize_html(body)
        body = filter_hallucinated(body)

    # ───────────────── 출력 ─────────────────
    st.markdown("### 📋 답변")
    st.markdown(body, unsafe_allow_html=True)
    
    # 각주 렌더링
    if footnote_sources:
        footnotes_html = render_footnotes_section(footnote_sources)
        st.markdown(footnotes_html, unsafe_allow_html=True)

    # Think 모드일 때만 추론 과정 표시
    if is_think_mode and thinking_process:
        with st.expander("🧠 AI 상세 추론 과정", expanded=False):
            # 추론 과정을 섹션별로 나누어 표시
            thinking_sections = thinking_process.split('\n\n')
            for i, section in enumerate(thinking_sections, 1):
                if section.strip():
                    st.markdown(f"**🔄 추론 단계 {i}**")
                    st.markdown(sanitize_html(section.strip()), unsafe_allow_html=True)
                    st.markdown("---")

    # ── 참고 문서 (정확한 분류로 표시) ──
    with st.expander("🔍 참고 문서 보기"):
        if law_docs:
            st.markdown("#### 📜 법령")
            for i, d in enumerate(law_docs, 1):
                pid = d["meta"].get("parent_id", d["id"])
                title = d["meta"].get("title", pid)
                snippet = d["text"][:300].replace("\n", " ")

                if st.button(f"📄 [{i}] {title}", key=f"law_btn_{pid}_{i}"):
                    st.session_state[f"show_law_{pid}"] = not st.session_state.get(f"show_law_{pid}", False)

                if st.session_state.get(f"show_law_{pid}", False):
                    data = load_json(pid)
                    if data:
                        st.json(data, expanded=False)
                    else:
                        st.warning("원본 JSON을 찾을 수 없습니다.")

                st.markdown(f"<span style='font-size:0.9rem;color:#666'>{snippet}...</span>", unsafe_allow_html=True)
        
        if prec_docs:
            st.markdown("#### ⚖️ 판례")
            for i, d in enumerate(prec_docs, 1):
                pid = d["meta"].get("parent_id", d["id"])
                title = d["meta"].get("title", pid)
                snippet = d["text"][:300].replace("\n", " ")

                if st.button(f"📄 [{i}] {title}", key=f"prec_btn_{pid}_{i}"):
                    st.session_state[f"show_prec_{pid}"] = not st.session_state.get(f"show_prec_{pid}", False)

                if st.session_state.get(f"show_prec_{pid}", False):
                    data = load_json(pid)
                    if data:
                        st.json(data, expanded=False)
                    else:
                        st.warning("원본 JSON을 찾을 수 없습니다.")

                st.markdown(f"<span style='font-size:0.9rem;color:#666'>{snippet}...</span>", unsafe_allow_html=True)

    # 검색 완료 후 플래그 초기화
    st.session_state.should_search = False

# ─────────────────── 푸터 ───────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem'>"
    "Made with Streamlit · Gemini 2.5 Flash · ChromaDB<br/>"
    "⚠️ AI가 생성한 답변이므로 실제 법률 자문은 전문가에게 문의하세요.<br/>"
    "⚠ AI can make mistakes. Please verify the information with a reliable source."
    "</div>",
    unsafe_allow_html=True
)