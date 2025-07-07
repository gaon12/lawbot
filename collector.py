#!/usr/bin/env python3
"""
🇰🇷 법령(법) + 판례(prec) DRF API 비동기 수집기
────────────────────────────────────────────
1. asyncio + aiohttp : 네트워크 I/O 완전 비동기
2. ThreadPool 제거 → GIL·락 이슈 감소, 메모리 사용↓
3. 법령·판례 공통 파이프라인 (target 매개변수 하나로 처리)
4. aiosqlite로 메인-루프 안에서도 DB 업서트 non-blocking
5. tqdm 세 단계 진행 막대 유지 (목록 → 다운로드 → DB)
6. rerequest.json에 명시된 ID는 강제 재다운로드
"""

import os, math, json, asyncio, aiohttp, aiosqlite
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# ──────────────── 환경값 & 경로 ────────────────────
load_dotenv()
API_KEY      = os.getenv("API_KEY") or exit("⚠️  .env의 API_KEY 누락")
MAX_CONC     = 32   # 동시 연결 수
DISPLAY      = 100  # DRF 한 페이지 당 개수
BASE_SEARCH  = "https://www.law.go.kr/DRF/lawSearch.do"
BASE_SERVICE = "https://www.law.go.kr/DRF/lawService.do"

RAW_DIR = "./data/laws/raw"          # 법령 raw
PRE_DIR = "./data/laws/preprocess"
RAW_P  = "./data/precs/raw"          # 판례 raw
PRE_P  = "./data/precs/preprocess"
DB_PATH = "./data/laws/laws.db"
REREQUEST_PATH = "./rerequest.json"  # 강제 갱신 목록

for p in (RAW_DIR, PRE_DIR, RAW_P, PRE_P, os.path.dirname(DB_PATH)):
    os.makedirs(p, exist_ok=True)

# ──────────────── rerequest 목록 로드 ───────────────
def load_rerequest() -> Tuple[set, set]:
    try:
        with open(REREQUEST_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        law_ids  = set(map(str, d.get("law", [])   + d.get("laws", [])))
        prec_ids = set(map(str, d.get("prec", [])  + d.get("precs", [])))
        return law_ids, prec_ids
    except FileNotFoundError:
        return set(), set()
    except Exception as e:
        print(f"⚠️  rerequest.json 파싱 오류: {e}")
        return set(), set()

RE_LAWS, RE_PRECS = load_rerequest()

# ──────────────── DB 초기화 ─────────────────────────
INIT_SQL = {
    "laws": """
        CREATE TABLE IF NOT EXISTS laws (
            law_id      TEXT PRIMARY KEY,
            law_title   TEXT,
            law_abbr    TEXT,
            first_seen  TEXT,
            last_saved  TEXT,
            is_current  INTEGER,
            indexed     INTEGER DEFAULT 0
        );
    """,
    "precs": """
        CREATE TABLE IF NOT EXISTS precedents (
            prec_id     TEXT PRIMARY KEY,
            first_seen  TEXT,
            last_saved  TEXT,
            indexed     INTEGER DEFAULT 0
        );
    """
}

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        # 테이블 생성
        for sql in INIT_SQL.values():
            await db.execute(sql)
        await db.commit()

        # 컬럼 자동 추가(이미 존재한다면 무시)
        async def ensure_column(table: str, col: str, typ: str):
            rows = [r async for r in await db.execute(f"PRAGMA table_info({table})")]
            col_names = [r[1] for r in rows]  # r[1]은 컬럼명
            if col not in col_names:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ};")
        await ensure_column("laws",  "law_title", "TEXT")
        await ensure_column("laws",  "law_abbr",  "TEXT")
        await db.commit()

# ──────────────── 공통 헬퍼 ─────────────────────────
def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def collect_text(value, acc: List[str]):
    if value is None: return
    if isinstance(value, str):
        txt = value.strip()
        if txt: acc.append(txt)
    elif isinstance(value, list):
        for v in value: collect_text(v, acc)
    elif isinstance(value, dict):
        for v in value.values(): collect_text(v, acc)

# ──────────────── 비동기 HTTP ──────────────────────
async def fetch_json(session: aiohttp.ClientSession, url: str, params: Dict[str,str], timeout=15):
    async with session.get(url, params=params, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json(content_type=None)

async def get_total(session, target: str) -> int:
    j = await fetch_json(session, BASE_SEARCH, {
        "OC": API_KEY, "target": target, "type": "JSON",
        "display": DISPLAY, "page": 1
    })
    key = "LawSearch" if target=="law" else "PrecSearch"
    return int(j[key]["totalCnt"])

# ──────────────── 전처리 규칙 ──────────────────────
def preprocess_law(mst:str, raw:Dict[str,Any]) -> Dict[str,Any]:
    root  = raw.get("법령", {})
    basic = root.get("기본정보", {})
    lines : List[str] = []
    # 주요 텍스트 필드 수집
    collect_text(root.get("개정문", {}).get("개정문내용"), lines)
    collect_text(root.get("제개정이유", {}).get("제개정이유내용"), lines)
    collect_text(root.get("부칙", {}).get("부칙단위"), lines)
    for cl in root.get("조문", {}).get("조문단위", []):
        collect_text(cl, lines)
    return {
        "mst": mst,
        "law_id": basic.get("법령ID", ""),
        "title":  basic.get("법령명_한글", ""),
        "abbr":   basic.get("법령명약칭") or basic.get("법령약칭명") or "",
        "content": "\n".join(lines)
    }

def preprocess_prec(pid:str, raw:Dict[str,Any]) -> Dict[str,Any]:
    root  = raw.get("PrecService", {})
    lines : List[str] = []
    for k in ("판시사항","판결요지","판례내용"):
        collect_text(root.get(k), lines)
    return {
        "prec_id": root.get("판례정보일련번호") or pid,
        "title":   root.get("사건명", ""),
        "court":   root.get("법원명", ""),
        "content": "\n".join(lines)
    }

# ──────────────── 파이프라인 ───────────────────────
async def collect_master(session, target:str) -> List[Tuple[str,bool]]:
    """target='law' → [(mst,is_current)], target='prec' → [(pid,True)]"""
    total = await get_total(session, target)
    pages = math.ceil(total / DISPLAY)
    out=[]
    desc = "목록(Law)" if target=="law" else "목록(Prec)"
    for page in tqdm(range(1, pages+1), desc=desc, unit="page"):
        res = await fetch_json(session, BASE_SEARCH, {
            "OC": API_KEY, "target": target, "type":"JSON",
            "display": DISPLAY, "page": page
        })
        key   = "law" if target=="law" else "prec"
        items = res["LawSearch" if target=="law" else "PrecSearch"].get(key,[])
        for it in items:
            if target=="law":
                mst = str(it.get("법령일련번호","")).strip()
                if mst: out.append((mst, it.get("현행연혁코드","")=="현행"))
            else:
                pid = str(it.get("판례일련번호") or it.get("id","")).strip()
                if pid: out.append((pid, True))
    return out

async def worker(item:str, flag:bool, target:str, session:aiohttp.ClientSession) -> Tuple[str,bool,Optional[str],Optional[str]]:
    """
    다운로드→전처리→저장 : (key,is_current,title,abbr)
    target=='law'  → (law_id,flag,title,abbr)
    target=='prec' → (prec_id,flag,None,None)
    """
    raw_dir, pre_dir = (RAW_DIR, PRE_DIR) if target=="law" else (RAW_P, PRE_P)
    raw_p = os.path.join(raw_dir, f"{item}.json")
    pre_p = os.path.join(pre_dir, f"{item}.json")

    # 강제 재다운로드 여부
    force_redl = (item in RE_LAWS) if target=="law" else (item in RE_PRECS)

    # 이미 전처리 파일이 있고, 강제 모드가 아니면 즉시 반환
    if not force_redl and os.path.exists(pre_p):
        try:
            with open(pre_p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if target=="law":
                return data.get("law_id",""), flag, data.get("title",""), data.get("abbr","")
            else:
                return data.get("prec_id",""), flag, None, None
        except Exception:
            pass  # 손상 시 재처리

    # fetch
    try:
        raw = await fetch_json(session, BASE_SERVICE, {
            "OC": API_KEY, "target": target,
            ("MST" if target=="law" else "ID"): item,
            "type":"JSON"
        })
    except Exception as e:
        print(f"⚠️  fetch 실패({item}): {e}")
        return "", flag, None, None
    await asyncio.to_thread(save_json, raw_p, raw)

    # preprocess
    pre = (preprocess_law if target=="law" else preprocess_prec)(item, raw)
    await asyncio.to_thread(save_json, pre_p, pre)

    if target=="law":
        return pre["law_id"], flag, pre["title"], pre["abbr"]
    else:
        return pre["prec_id"], flag, None, None

# ──────────────── DB 업서트 ────────────────────────
async def upsert_many(rows, table:str):
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        if table=="laws":
            await db.executemany(
                """INSERT INTO laws(law_id, law_title, law_abbr,
                                    first_seen, last_saved, is_current, indexed)
                   VALUES(?,?,?,?,?,?,0)
                   ON CONFLICT(law_id) DO UPDATE SET
                        law_title = excluded.law_title,
                        law_abbr  = excluded.law_abbr,
                        last_saved= excluded.last_saved,
                        is_current= excluded.is_current""",
                [(k,t,a,now,now,int(f)) for k,f,t,a in rows]
            )
        else:
            await db.executemany(
                """INSERT INTO precedents(prec_id, first_seen, last_saved, indexed)
                   VALUES(?,?,?,0)
                   ON CONFLICT(prec_id) DO UPDATE SET
                        last_saved=excluded.last_saved""",
                [(k,now,now) for k, *_ in rows]
            )
        await db.commit()

# ──────────────── 메인 루틴 ────────────────────────
async def pipeline(target:str):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=MAX_CONC)) as session:
        masters = await collect_master(session, target)
        kind = "법령" if target=="law" else "판례"
        print(f"◎ {kind} {len(masters):,}건 처리 시작 (동시 {MAX_CONC})")

        batch, tasks = [], []
        sem = asyncio.Semaphore(MAX_CONC)

        async def sem_worker(m,f):
            async with sem:
                return await worker(m,f,target,session)

        for m,f in masters:
            tasks.append(asyncio.create_task(sem_worker(m,f)))

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                         desc=f"처리중({kind})", unit="item"):
            key, flag, title, abbr = await coro
            if key:
                batch.append((key, flag, title, abbr))
            if len(batch) >= 500:
                await upsert_many(batch, "laws" if target=="law" else "precedents")
                batch.clear()

        if batch:
            await upsert_many(batch, "laws" if target=="law" else "precedents")
        print(f"★ {kind} 완료")

async def main():
    await init_db()
    await pipeline("law")
    await pipeline("prec")
    print("🎉 모든 작업 완료")

if __name__ == "__main__":
    asyncio.run(main())
