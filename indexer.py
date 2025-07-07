#!/usr/bin/env python3
"""
전처리 JSON(법령 · 판례) → Gemini 임베딩 → Chroma DB 적재
────────────────────────────────────────────────────────
● .env
    GEMINI_API_KEY=<YOUR_KEY>

● 주요 특징
    1) ThreadPoolExecutor(기본 8)로 임베딩 병렬 처리
    2) 3-계층 분할(문단 → 문장 → 고정 글자 수)로 긴 텍스트 안전 분할
    3) Chroma 컬렉션 두 개(laws·precs) 업서트
       - id        :  LAW_{law_id}_{chunk:03d}  /  PREC_{prec_id}_{chunk:03d}
       - document  :  조각 텍스트
       - metadata  :  title, parent_id, file_path,
                      is_current(법령), court(판례)
    4) 적재 완료 후 각 테이블 indexed 플래그 갱신
    5) tqdm 진행 막대

설치:  pip install chromadb google-generativeai tqdm python-dotenv
"""

from __future__ import annotations

import os, glob, json, sqlite3
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────── 환경값 ────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("⚠️  .env 파일에 GEMINI_API_KEY가 필요하다.")

# 경로
PRE_LAW_DIR  = "./data/laws/preprocess"
PRE_PREC_DIR = "./data/precs/preprocess"
DB_PATH      = "./data/laws/laws.db"          # 법령·판례 모두 같은 DB
CHROMA_DIR   = "./data/ChromaLegal"           # 통합 저장소

# 파라미터
MAX_WORKERS = 8
BATCH       = 128
MAX_CHARS   = 6_000
MIN_CHARS   = 1_000
EMBED_MODEL = "gemini-embedding-exp-03-07"

# ─────────────────────── 외부 라이브러리 ────────────────────────
from google import genai          # type: ignore
import chromadb                   # type: ignore

genai_client = genai.Client(api_key=GEMINI_API_KEY)
ch_client    = chromadb.PersistentClient(path=CHROMA_DIR)
col_law      = ch_client.get_or_create_collection("laws",  metadata={"hnsw:space": "cosine"})
col_prec     = ch_client.get_or_create_collection("precs", metadata={"hnsw:space": "cosine"})

# ─────────────────────── 일반 함수 ────────────────────────
def split_text(text: str) -> List[str]:
    """문단→문장→고정 글자 수 3-계층 분할."""
    chunks, buf = [], []

    def flush():
        if buf:
            chunks.append(" ".join(buf).strip())
            buf.clear()

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        splitter = "。" if "。" in para else "."
        for sent in para.split(splitter):
            sent = sent.strip()
            if not sent:
                continue
            buf.append(sent)
            if sum(len(s) for s in buf) >= MAX_CHARS:
                flush()
    flush()

    # 너무 큰/작은 조각 재조정
    final = []
    for c in chunks:
        if len(c) <= MAX_CHARS:
            final.append(c)
        else:
            for i in range(0, len(c), MAX_CHARS):
                final.append(c[i : i + MAX_CHARS])

    merged = []
    for c in final:
        if merged and len(c) < MIN_CHARS:
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged


def embed_text(text: str) -> List[float]:
    """Gemini 임베딩 3072-d."""
    res = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        task_type="SEMANTIC_SIMILARITY",
    )
    return res.embeddings  # type: ignore


def mark_indexed(table: str, parent_ids: List[str]) -> None:
    """indexed = 1 갱신."""
    if not parent_ids:
        return
    col_name = "law_id" if table == "laws" else "prec_id"
    query = f"UPDATE {table} SET indexed = 1 WHERE {col_name} = ?"
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(query, [(pid,) for pid in parent_ids])
        conn.commit()

# ─────────────────────── 워커 ────────────────────────
def worker(path: str, kind: str) -> Tuple[str, List[Tuple[str, str, Dict[str,object], List[float]]]]:
    """
    JSON 하나를 읽어 (parent_id, [(id, text, meta, vec), ...]) 반환
    kind : 'law' | 'prec'
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if kind == "law":
            parent_id = data["law_id"]
            title     = data.get("title", "")
            is_cur    = bool(data.get("is_current", False))
            content   = data.get("content", "")
            prefix    = "LAW_"
        else:
            parent_id = data["prec_id"]
            title     = data.get("title", "")
            court     = data.get("court", "")
            content   = data.get("content", "")
            prefix    = "PREC_"

        chunks = split_text(content)
        results = []
        for idx, chunk in enumerate(chunks):
            vec = embed_text(chunk)
            chunk_id = f"{prefix}{parent_id}_{idx:03d}"

            meta: Dict[str, object] = {
                "parent_id": parent_id,
                "title":     title,
                "file_path": path,      # ←── 추가된 필드
            }
            if kind == "law":
                meta["is_current"] = is_cur
            else:
                meta["court"] = court
            results.append((chunk_id, chunk, meta, vec))
        return parent_id, results
    except Exception:
        return "", []

# ─────────────────────── 메인 ────────────────────────
def load_unindexed(table: str, id_col: str) -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        return {row[0] for row in conn.execute(f"SELECT {id_col} FROM {table} WHERE indexed = 0")}

def gather_candidates() -> List[Tuple[str,str]]:
    """indexed=0 인 항목들의 (path, kind) 목록 생성"""
    law_set  = load_unindexed("laws",       "law_id")
    prec_set = load_unindexed("precedents", "prec_id")

    cands: List[Tuple[str,str]] = []
    for p in glob.glob(os.path.join(PRE_LAW_DIR, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                jid = json.load(f).get("law_id", "")
            if jid in law_set:
                cands.append((p, "law"))
        except Exception:
            pass

    for p in glob.glob(os.path.join(PRE_PREC_DIR, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                jid = json.load(f).get("prec_id", "")
            if jid in prec_set:
                cands.append((p, "prec"))
        except Exception:
            pass
    return cands

def flush_buffers(ids, docs, meta, vecs):
    """법령/판례 항목을 컬렉션별로 분리해 한꺼번에 업서트"""
    law_idx  = [i for i, cid in enumerate(ids) if cid.startswith("LAW_")]
    prec_idx = [i for i, cid in enumerate(ids) if cid.startswith("PREC_")]

    if law_idx:
        col_law.add(
            ids        =[ids[i]  for i in law_idx],
            documents  =[docs[i] for i in law_idx],
            metadatas  =[meta[i] for i in law_idx],
            embeddings =[vecs[i] for i in law_idx],
        )
    if prec_idx:
        col_prec.add(
            ids        =[ids[i]  for i in prec_idx],
            documents  =[docs[i] for i in prec_idx],
            metadatas  =[meta[i] for i in prec_idx],
            embeddings =[vecs[i] for i in prec_idx],
        )

def main():
    candidates = gather_candidates()
    print(f"\n▷ 임베딩 대상: {len(candidates):,}건 — 병렬 {MAX_WORKERS} 스레드\n")

    # 배치 버퍼
    buf_ids, buf_docs, buf_meta, buf_vecs = [], [], [], []
    done_laws, done_precs = [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(worker, path, kind) for path, kind in candidates]

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Embedding", unit="file"):
            parent_id, pieces = fut.result()
            if not pieces:
                continue

            is_law = pieces[0][0].startswith("LAW_")

            for cid, text, meta, vec in pieces:
                buf_ids.append(cid)
                buf_docs.append(text)
                buf_meta.append(meta)
                buf_vecs.append(vec)

            (done_laws if is_law else done_precs).append(parent_id)

            if len(buf_ids) >= BATCH:
                flush_buffers(buf_ids, buf_docs, buf_meta, buf_vecs)
                mark_indexed("laws",       done_laws)
                mark_indexed("precedents", done_precs)

                # 버퍼 초기화
                buf_ids, buf_docs, buf_meta, buf_vecs = [], [], [], []
                done_laws, done_precs = [], []

    # 잔여 항목 처리
    if buf_ids:
        flush_buffers(buf_ids, buf_docs, buf_meta, buf_vecs)
    mark_indexed("laws",       done_laws)
    mark_indexed("precedents", done_precs)

    print("\n◎ Chroma 임베딩 완료")

if __name__ == "__main__":
    main()
