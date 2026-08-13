# -*- coding: utf-8 -*-
"""실시간 GK2A 3채널 수집 — 기상청 API 허브.

`nowcast/realtime.py`(IR105 단일, 검증 완료)를 3채널로 넓힌 것. 배포본이
학습 저장소에 의존하지 않도록 여기로 독립시켰다.

  https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{CH}/KO/data?date=YYYYMMDDHHMM
  date 는 **UTC**. 200 + Content-Type 에 netcdf 면 성공, 404 면 결측.

시각 규칙 — 여기서 틀리기 제일 쉽다
  ★ KO(한반도) 관측은 **10분 간격**인데 모델은 **30분 간격 4장**으로 학습했다.
    10분 간격을 넣으면 이동량이 1/3 이라 모델이 전혀 다른 상황으로 읽는다.
  ★ 실시간 지연이 약 8분이라 '지금'을 그대로 요청하면 404 다 — 10분 격자를
    **거슬러 탐색**해야 한다.
  ★ 관측 분(minute)이 통째로 밀리는 날이 있다(2024-09-20 은 06:00 -> 06:10 UTC).
    그래서 ':00 만 있다'고 가정하지 않고 10분 격자를 다 훑는다. 30분 간격은
    어느 10분 눈금에서 시작하든 같은 눈금 위에 머무르므로 안전하다.

결측은 **채우지 않는다.** 한 장이라도 없으면 그 시각은 포기한다 — 이전 프레임으로
때우면 모델이 '구름이 멈췄다'고 착각해 조용히 persistence 로 퇴화한다.
"""
import datetime as dt
import os
import time

import numpy as np
import requests

from gk2a_cal import lut
from inference import (CH_NORM, CROP_X0, CROP_X1, CROP_Y0, CROP_Y1, POOL,
                       IN_FRAMES, STEP_S)

UTC = dt.timezone.utc
KST = dt.timezone(dt.timedelta(hours=9))
BASE = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B"
DOMAIN = "KO"
CHANNELS = ("IR105", "WV063", "SW038")
SCAN_STEP = dt.timedelta(minutes=10)          # KO 실제 관측 간격
TIMEOUT = 60

_SESSION = requests.Session()
_LUTS = {}


class FetchError(RuntimeError):
    pass


def load_key():
    """기상청 API 허브 인증키.

    ★ `.kmaapirc` 는 `key: xxxx` 형식이다. 파일 전체를 키로 쓰면 안 된다.
    """
    k = os.environ.get("KMA_API_KEY")
    if k:
        return k.strip()
    try:
        import streamlit as st
        k = st.secrets.get("KMA_API_KEY")
        if k:
            return str(k).strip()
    except Exception:                          # noqa: BLE001
        pass
    path = os.environ.get("KMA_APIRC", os.path.expanduser("~/.kmaapirc"))
    if os.path.exists(path):
        for line in open(path, encoding="utf-8").read().splitlines():
            if line.strip().lower().startswith("key:"):
                return line.split(":", 1)[1].strip()
    return None


def _lut(ch):
    """DN -> BT LUT. 무효(NaN)는 채널 상한으로 — build_cache 와 같은 규칙."""
    if ch not in _LUTS:
        L = lut(ch).astype(np.float32)
        L[np.isnan(L)] = CH_NORM[ch][1]
        _LUTS[ch] = L
    return _LUTS[ch]


def fetch_raw(ch, t_utc, key, tries=4):
    """한 채널·한 시각의 .nc 바이트. 결측이면 None.

    ★ 429 는 **물러섰다 다시** 해야 한다. 즉시 예외로 올리면 일시적인 초당 한도
      하나에 예측 전체가 죽는다. 12장을 연달아 받는 구조라 실제로 걸린다.
      다만 하루 한도를 넘긴 경우엔 아무리 기다려도 안 풀리므로 횟수는 제한한다.
    """
    url = f"{BASE}/{ch}/{DOMAIN}/data"
    last429 = False
    for i in range(tries):
        try:
            r = _SESSION.get(url, timeout=TIMEOUT,
                             params=dict(date=f"{t_utc:%Y%m%d%H%M}", authKey=key))
            if r.status_code == 200 and "netcdf" in r.headers.get("Content-Type", ""):
                return r.content
            if r.status_code == 404:
                return None
            last429 = r.status_code == 429
            if last429:
                time.sleep(3.0 * (i + 1))        # 3, 6, 9초
                continue
        except requests.RequestException:
            pass
        time.sleep(1.0 * (i + 1))
    if last429:
        raise FetchError("기상청 API 호출 한도(429)가 계속됩니다 — "
                         "하루 한도를 넘었을 수 있습니다.")
    return None


def nc_to_bt(raw, ch):
    """.nc 바이트 -> (256,256) 밝기온도(K).

    ★ 캐시(build_cache.py)와 **똑같은 순서**여야 한다: LUT -> crop -> 2x2 평균.
      순서를 바꾸면(예: 풀링 먼저) 극값이 달라져 성능이 조용히 무너진다.
    ★ netCDF4/HDF5 는 스레드 안전하지 않다 — 여기서 병렬화하지 말 것(SIGSEGV).
    """
    import netCDF4 as nc
    with nc.Dataset("inmem", mode="r", memory=raw) as ds:
        dn = np.asarray(ds.variables["image_pixel_values"][:])
    L = _lut(ch)
    bt = L[np.clip(dn, 0, L.size - 1).astype(np.int32)]
    bt = bt[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    h, w = bt.shape
    return bt.reshape(h // POOL, POOL, w // POOL, POOL).mean(axis=(1, 3))


def latest_common_time(key, back_minutes=180):
    """3채널이 **모두** 있는 가장 최신 관측시각(UTC)과 지연(분).

    IR105 로 먼저 훑고, 걸린 시각에서 나머지 둘을 확인한다. 매 후보마다 3채널을
    다 때리면 요청이 3배가 된다 — 지연이 8분이라 보통 1~2번 만에 걸린다.
    """
    now = dt.datetime.now(UTC).replace(second=0, microsecond=0)
    t = now - dt.timedelta(minutes=now.minute % 10)
    for _ in range(back_minutes // 10):
        if fetch_raw("IR105", t, key) is not None:
            if all(fetch_raw(ch, t, key) is not None for ch in ("WV063", "SW038")):
                return t, (now - t).total_seconds() / 60.0
        t -= SCAN_STEP
    return None, None


def build_frames(key, t_end=None, progress=None):
    """모델 입력 3채널 x 4프레임.

    반환 (irs, wvs, sws, times, lag) — 각 (4,256,256) 밝기온도(K), 과거->현재.
    한 장이라도 결측이면 FetchError.

    progress : 선택. progress(done, total, label) 로 불린다 (Streamlit 진행바용).
    """
    if t_end is None:
        t_end, lag = latest_common_time(key)
        if t_end is None:
            raise FetchError("3시간 안에 3채널이 다 갖춰진 시각이 없습니다.")
    else:
        lag = None

    times = [t_end - dt.timedelta(seconds=STEP_S * k)
             for k in range(IN_FRAMES - 1, -1, -1)]
    total = len(times) * len(CHANNELS)
    got = {ch: [] for ch in CHANNELS}
    n = 0
    for t in times:
        for ch in CHANNELS:
            raw = fetch_raw(ch, t, key)
            if raw is None:
                raise FetchError(
                    f"{ch} {t:%m-%d %H:%M}UTC 결측 — 이 시각은 예측할 수 없습니다. "
                    "(결측을 이전 프레임으로 채우면 모델이 '구름이 멈췄다'고 읽습니다)")
            got[ch].append(nc_to_bt(raw, ch))
            n += 1
            if progress:
                progress(n, total, f"{ch} {t:%H:%M}UTC")
    return (np.stack(got["IR105"]), np.stack(got["WV063"]),
            np.stack(got["SW038"]), times, lag)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = load_key()
    if not key:
        raise SystemExit("★ 인증키 없음 (~/.kmaapirc 또는 KMA_API_KEY)")

    print("[1] 3채널 공통 최신 시각 탐색")
    t0 = time.time()
    t_end, lag = latest_common_time(key)
    if t_end is None:
        raise SystemExit("★ 3시간 안에 자료가 없다")
    print(f"  {t_end:%Y-%m-%d %H:%M} UTC = {t_end.astimezone(KST):%H:%M} KST"
          f"   지연 {lag:.0f}분   (탐색 {time.time()-t0:.1f}초)")

    print(f"\n[2] 30분 간격 {IN_FRAMES}장 x 3채널 = 12장 수집")
    t0 = time.time()
    irs, wvs, sws, times, _ = build_frames(
        key, t_end, progress=lambda n, tot, lab: print(f"  {n:2d}/{tot} {lab}", flush=True))
    t_fetch = time.time() - t0
    print(f"  수집 {t_fetch:.1f}초")
    for nm, a in (("IR105", irs), ("WV063", wvs), ("SW038", sws)):
        print(f"  {nm}  {a.shape}  BT {a.min():.1f}~{a.max():.1f}K")
    print(f"  SW-IR {(sws-irs).min():+.1f}~{(sws-irs).max():+.1f}K   "
          f"WV-IR {(wvs-irs).min():+.1f}~{(wvs-irs).max():+.1f}K")

    print("\n[3] 2단 추론")
    from inference import Predictor
    t0 = time.time()
    P = Predictor()
    t_load = time.time() - t0
    t0 = time.time()
    valid = t_end + dt.timedelta(seconds=STEP_S)
    out = P.predict_grid(irs, wvs, sws, int(valid.timestamp()))
    t_inf = time.time() - t0
    cat = out["cat"]
    print(f"  대상 시각 {valid.astimezone(KST):%m-%d %H:%M} KST")
    print(f"  구간 분포 {[round(100*float((cat==i).mean()),2) for i in range(5)]} %")
    print(f"  모델 적재 {t_load:.1f}초 + 추론 {t_inf:.1f}초")
    print(f"\n웹앱 총 소요: 수집 {t_fetch:.1f} + 추론 {t_inf:.1f} = {t_fetch+t_inf:.1f}초")
