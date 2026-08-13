# -*- coding: utf-8 -*-
"""지역 검색 — 카카오 로컬 API.

두 엔드포인트를 **둘 다** 쓴다. 하나만으로는 사용자가 칠 법한 말을 못 덮는다.

  주소 검색  /v2/local/search/address.json   "성수동2가", "서울 성동구" 같은 행정구역
  키워드 검색 /v2/local/search/keyword.json   "성수역", "롯데월드" 같은 장소명

주소 검색을 먼저 보여준다 — 이 앱은 '지역'의 날씨를 묻는 것이라
동네 이름이 카페 이름보다 위에 와야 한다.

인증키
  배포:   Streamlit secrets 의 KAKAO_REST_KEY
  로컬:   환경변수 KAKAO_REST_KEY 또는 ~/.kakaorc 의 `key:` 뒤쪽
  ★ REST API 키다. JavaScript 키를 넣으면 401 이 난다 (카카오 개발자 콘솔에서
    같은 앱에 네 종류 키가 나오는데 서버 호출은 REST 키만 받는다).
  ★ 저장소에 커밋하지 마라.

응답 좌표는 WGS84 경위도이고 **x 가 경도, y 가 위도**다 (뒤집어 쓰기 쉽다).
"""
import os

import requests

KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
TIMEOUT = 6

_SESSION = requests.Session()


class GeocodeError(RuntimeError):
    pass


def load_key():
    """REST API 키. 없으면 None (앱은 '키 미설정' 안내를 띄운다)."""
    k = os.environ.get("KAKAO_REST_KEY")
    if k:
        return k.strip()
    try:                                   # 배포 환경에서만 존재
        import streamlit as st
        k = st.secrets.get("KAKAO_REST_KEY")
        if k:
            return str(k).strip()
    except Exception:                      # noqa: BLE001  로컬에선 secrets.toml 이 없다
        pass
    path = os.path.expanduser("~/.kakaorc")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8").read().splitlines():
            if line.strip().lower().startswith("key:"):
                return line.split(":", 1)[1].strip()
    return None


def _get(url, key, params):
    r = _SESSION.get(url, params=params, timeout=TIMEOUT,
                     headers={"Authorization": f"KakaoAK {key}"})
    if r.status_code == 401:
        raise GeocodeError("카카오 인증 실패(401) — REST API 키가 맞는지 확인하세요.")
    if r.status_code == 403:
        # ★ 키가 맞아도 403 이 난다. 로컬 API 는 **플랫폼(도메인) 등록**이 있어야 열린다.
        #   카카오 개발자 콘솔 > 앱 > 플랫폼 키 > REST API 키 > 수정 에서
        #   서비스 도메인(배포 URL, 로컬 테스트는 http://localhost:8501)을 등록할 것.
        raise GeocodeError(
            "카카오 로컬 API 접근 거부(403) — 앱에 서비스 도메인이 등록돼 있지 않습니다. "
            "개발자 콘솔에서 이 앱의 REST API 키에 배포 URL을 등록하세요.")
    if r.status_code == 429:
        raise GeocodeError("카카오 API 호출 한도를 넘었습니다. 잠시 뒤 다시 시도하세요.")
    r.raise_for_status()
    return r.json().get("documents", [])


def search(query, key=None, size=8):
    """지역명 -> [dict(name, detail, lat, lon)]. 주소 결과가 앞에 온다.

    빈 질의는 빈 목록. 키가 없으면 GeocodeError.
    """
    query = (query or "").strip()
    if not query:
        return []
    key = key or load_key()
    if not key:
        raise GeocodeError("카카오 REST API 키가 없습니다.")

    out, seen = [], set()

    def add(name, detail, lon, lat):
        # 같은 지점이 주소·키워드 양쪽에서 오면 한 번만 (약 100m 격자로 중복 판정)
        k = (name, round(float(lat), 3), round(float(lon), 3))
        if k in seen:
            return
        seen.add(k)
        out.append(dict(name=name, detail=detail,
                        lat=float(lat), lon=float(lon)))

    for d in _get(ADDRESS_URL, key, dict(query=query, size=size)):
        a = d.get("address") or d.get("road_address") or {}
        # region_3depth_name 이 동 이름. 없으면(시·구 단위 검색) 전체 주소를 쓴다.
        name = a.get("region_3depth_name") or d.get("address_name", query)
        add(name, d.get("address_name", ""), d["x"], d["y"])

    # ★ 키워드 검색(장소)은 쓰지 않는다. "성수동" 을 치면 카페거리·전망대·곤충식물원
    #   같은 POI 가 10건 중 8건을 채워서 정작 동네가 묻힌다. 이 앱이 답하는 건
    #   '이 지역에 비가 오나' 이지 '이 가게가 어디냐' 가 아니다.
    #   되살리려면 아래를 풀면 된다:
    #     for d in _get(KEYWORD_URL, key, dict(query=query, size=size)):
    #         add(d.get("place_name", query),
    #             d.get("road_address_name") or d.get("address_name", ""),
    #             d["x"], d["y"])
    return out
