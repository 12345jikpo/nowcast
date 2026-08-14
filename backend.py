# -*- coding: utf-8 -*-
"""예측 백엔드 — 실시간 GK2A 3채널 -> U-Net 3개 -> LightGBM 구간 분류기 4개.

    predict_point(lat, lon) -> dict(
        lv        : 0~4 단계
        probs     : {0.1,1.0,3.0,10.0: p}   단조성 보정 끝난 값
        valid_at  : datetime(KST) 예측 대상 시각 = 마지막 관측 + 30분
        base_at   : datetime(KST) 마지막 입력 프레임 시각
        live      : True 면 실제 위성, False 면 더미(키 없음)
    )

★ 캐싱이 핵심이다. 한 번 예측하려면 .nc 12장(약 10MB, 15초)을 받아야 하는데,
  30분 슬롯 안에서는 답이 바뀌지 않는다. 슬롯을 키로 캐시하면 그 30분 동안
  **모든 방문자가 한 번의 수집을 나눠 쓴다** — 기상청 하루 한도(5GB)를 지키는
  유일한 방법이기도 하다. 방문자마다 새로 받으면 300명이면 3GB다.
"""
import datetime as dt
import hashlib
import threading

from levels import BINS, categorize

KST = dt.timezone(dt.timedelta(hours=9))

# 도메인 (inference.py 의 lat/lon 격자와 같아야 한다)
LAT_MIN, LAT_MAX = 31.3, 40.9
LON_MIN, LON_MAX = 119.6, 132.1


def in_domain(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


# --------------------------------------------------------------------- 실제 예측
def _predictor():
    """무거운 객체(ONNX 3 + LightGBM 4 = 43MB)를 프로세스당 한 번만."""
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _mk():
        from inference import Predictor
        return Predictor()
    return _mk()


def _latest_obs():
    """3채널이 다 있는 최신 관측시각(UTC). 탐색은 가벼워서 2분만 캐시한다.

    ★ 이걸 캐시 키로 쓴다. 예전엔 **벽시계 30분 슬롯**을 키로 썼는데 그건 틀렸다 —
      위성 관측은 10분마다 오는데 캐시는 30분을 버티니, 슬롯 끝에 들어온 사용자는
      리드타임이 +20분에서 **−9분**까지 떨어진 예측(이미 지나간 시각!)을 봤다.
      관측 시각을 키로 잡으면 자료가 갱신될 때만 다시 계산하고 리드타임이
      +20 ~ +12분 사이에 머문다. 음수가 될 수 없다.
    """
    import streamlit as st

    @st.cache_data(ttl=120, show_spinner=False)
    def _probe(_bucket):
        import fetch
        k = fetch.load_key()
        if not k:
            raise fetch.FetchError("기상청 API 허브 인증키가 없습니다 "
                                   "(secrets 의 KMA_API_KEY).")
        t_end, lag, raws = fetch.latest_common_time(k)
        if t_end is None:
            raise fetch.FetchError("3시간 안에 3채널이 다 갖춰진 시각이 없습니다.")
        # ★ 탐색하며 받은 .nc(3장, 약 2.4MB)를 같이 들고 온다 — 버리면 수집 단계에서
        #   똑같은 걸 또 받는다(약 3.5초 낭비).
        return t_end, lag, raws
    # ttl 만으로도 되지만, 버킷을 키에 넣어야 2분 경계에서 확실히 새로 훑는다.
    return _probe(int(dt.datetime.now(dt.timezone.utc).timestamp()) // 120)


_GRID_CACHE = {}                       # t_iso -> 예측 격자
_GRID_LOCK = threading.Lock()
_KEEP = 3                              # 최근 관측 시각 3개만 들고 있는다


def _grid(t_end, progress=None, raws=None):
    """관측 시각 하나의 전 도메인 예측. 무거운 부분(12장 수집 + 추론)의 캐시 지점.

    ★ 여기에 `@st.cache_data` 를 쓰면 안 된다. 진행률 콜백이 함수 **안에서**
      바깥에서 만든 `st.empty()` 에 그리는데, cache_data 는 캐시 적중 시 내부
      UI 호출을 재생하려 하므로 "layout block created outside the function" 오류가 난다.
      Streamlit 은 프로세스 하나에 세션이 스레드로 붙으므로, 모듈 전역 dict 면
      방문자끼리 공유되면서 그 제약도 없다.
    ★ 락이 필요하다 — 동시에 두 명이 들어오면 12장을 두 번 받게 된다(하루 한도).
      뒤에 온 쪽은 앞사람 수집이 끝나기를 기다렸다가 캐시를 받는다.
    """
    key = t_end.isoformat()
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    with _GRID_LOCK:
        hit = _GRID_CACHE.get(key)     # 락을 기다리는 사이 앞사람이 채웠을 수 있다
        if hit is not None:
            return hit
        import fetch
        from inference import STEP_S
        k = fetch.load_key()
        irs, wvs, sws, times, _ = fetch.build_frames(k, t_end, progress=progress,
                                                     raws=raws)
        valid = t_end + dt.timedelta(seconds=STEP_S)
        out = _predictor().predict_grid(irs, wvs, sws, int(valid.timestamp()))
        out["valid_at"] = valid.astimezone(KST)
        out["base_at"] = t_end.astimezone(KST)
        _GRID_CACHE[key] = out
        for old in sorted(_GRID_CACHE)[:-_KEEP]:
            _GRID_CACHE.pop(old, None)
        return out


def predict(lat, lon, progress=None):
    """한 지점의 예측 + 표출용 격자. 격자는 캐시에서 공유된다."""
    t_end, lag, raws = _latest_obs()
    g = _grid(t_end, progress=progress, raws=raws)
    g = dict(g)
    g["lag_min"] = lag
    # ★ 리드타임은 '대상 시각 − 지금'이다. "30분 뒤"가 아니다 —
    #   30분은 마지막 관측 기준이고, 위성 지연만큼 이미 깎여 있다.
    g["lead_min"] = (g["valid_at"] - dt.datetime.now(KST)).total_seconds() / 60.0
    P = _predictor()
    px = P.pixel_of(lat, lon)
    if px is None:
        return None
    i, j = px
    probs = {b: float(g["probs"][b][i, j]) for b in BINS}
    # ★ 이웃 칸의 최고 단계. 화면이 스스로 모순돼 보이는 걸 막는 용도다 —
    #   격자가 4km 라 마커가 칸 경계에 걸리면 "주변은 온통 파란데 여기만 0" 이 되고
    #   사용자 눈에는 오류로 보인다. 예측값은 건드리지 않고 안내 문구만 덧붙인다.
    h, w = g["cat"].shape
    near = int(g["cat"][max(i - 1, 0):min(i + 2, h),
                        max(j - 1, 0):min(j + 2, w)].max())
    return dict(lv=categorize(probs, P.thrs), probs=probs, thresholds=P.thrs,
                valid_at=g["valid_at"], base_at=g["base_at"], lag_min=g["lag_min"],
                lead_min=g["lead_min"], near_lv=near, grid=g, live=True)


# --------------------------------------------------------------------- 더미
def predict_stub(lat, lon, now=None):
    """인증키가 없을 때만 쓰는 재현 가능한 가짜 값 (화면 확인용)."""
    t = (now or dt.datetime.now(KST)) - dt.timedelta(minutes=10)
    base = t.replace(minute=(t.minute // 30) * 30, second=0, microsecond=0)
    h = hashlib.sha256(f"{round(lat,3)}_{round(lon,3)}_{base:%Y%m%d%H%M}"
                       .encode()).digest()
    thr = {0.1: 0.1837, 1.0: 0.1519, 3.0: 0.1539, 10.0: 0.0991}
    p0, probs, prev = h[0] / 255.0, {}, 1.0
    for i, b in enumerate(BINS):
        p = min(p0 * (0.55 ** i) * (h[i + 1] / 255.0 * 0.6 + 0.7), prev)
        probs[b] = p
        prev = p
    valid = base + dt.timedelta(minutes=30)
    lv = categorize(probs, thr)
    return dict(lv=lv, probs=probs, thresholds=thr, near_lv=lv,
                valid_at=valid, base_at=base, lag_min=None, grid=None, live=False,
                lead_min=(valid - dt.datetime.now(KST)).total_seconds() / 60.0)


def predict_point(lat, lon, progress=None):
    """키가 있으면 실제, 없으면 더미."""
    import fetch
    if fetch.load_key():
        return predict(lat, lon, progress=progress)
    return predict_stub(lat, lon)
