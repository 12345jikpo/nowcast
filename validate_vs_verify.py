# -*- coding: utf-8 -*-
"""배포용 추론(ONNX)이 검증된 원본(torch)과 **같은 답**을 내는지 대조한다.

왜 이게 2단계의 핵심인가
  특징 조립은 56열의 순서·창 방향(min/max/mean)·정규화가 전부 맞아야 한다.
  하나만 틀려도 예외가 안 나고 그럴듯하게 틀린 숫자가 나온다. 눈으로는 못 잡는다.
  그래서 이미 채점까지 끝난 날(2026-07-20)의 원본 출력과 화소 단위로 맞춰본다.

정답지
  runs/verify_20260720.npz  <- verify_20260720.py 가 저장한 것
    cat  (256,256) 구간 0~4
    p01  (256,256) P(>=0.1)
    time 그림 시각 (KST)

  python validate_vs_verify.py [--npz ../runs/verify_20260720.npz]
"""
import argparse
import datetime as dt
import os
import re
import sys

import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gk2a_cal import lut
from inference import (Predictor, CROP_X0, CROP_X1, CROP_Y0, CROP_Y1, POOL,
                       CH_NORM, IN_FRAMES, STEP_S, BINS)

# 원본 .nc 는 학습용 자료 폴더에 있다 (배포본에는 안 들어간다)
DATA_ROOT = r"C:\Users\12345\Downloads\apcc\2024_AWS_GK2A_RADAR"
KST = dt.timezone(dt.timedelta(hours=9))
FNAME_RE = re.compile(r"_(\d{12})KST\.nc$")


def scan(ch, prefix="2026"):
    d = os.path.join(DATA_ROOT, "GK2A", ch)
    out = {}
    for fn in os.listdir(d):
        m = FNAME_RE.search(fn)
        if m and m.group(1).startswith(prefix):
            out[m.group(1)] = os.path.join(d, fn)
    return out


def load_bt(path, bt_lut):
    """.nc -> (256,256) BT(K). ★ 캐시와 같은 순서: LUT -> crop -> 2x2 평균."""
    with nc.Dataset(path) as ds:
        dn = np.asarray(ds.variables["image_pixel_values"][:])
    bt = bt_lut[np.clip(dn, 0, bt_lut.size - 1).astype(np.int32)]
    bt = bt[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    h, w = bt.shape
    return bt.reshape(h // POOL, POOL, w // POOL, POOL).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "runs",
        "verify_20260720.npz"))
    a = ap.parse_args()

    ref = np.load(a.npz, allow_pickle=True)
    t = dt.datetime.fromisoformat(str(ref["time"]))
    print(f"정답지: {os.path.basename(a.npz)}  시각 {t:%Y-%m-%d %H:%M %Z}")

    F = {ch: scan(ch) for ch in ("IR105", "WV063", "SW038")}
    keys = [(t - dt.timedelta(seconds=STEP_S * j)).strftime("%Y%m%d%H%M")
            for j in (4, 3, 2, 1)]
    print(f"입력 프레임: {keys[0]} ~ {keys[-1]}")
    for ch in F:
        miss = [k for k in keys if k not in F[ch]]
        if miss:
            raise SystemExit(f"★ {ch} 결측 {miss} — 원본 .nc 가 있어야 대조가 된다")

    luts = {}
    for ch in ("IR105", "WV063", "SW038"):
        L = lut(ch)
        L[np.isnan(L)] = CH_NORM[ch][1]
        luts[ch] = L

    irs = np.stack([load_bt(F["IR105"][k], luts["IR105"]) for k in keys])
    wvs = np.stack([load_bt(F["WV063"][k], luts["WV063"]) for k in keys])
    sws = np.stack([load_bt(F["SW038"][k], luts["SW038"]) for k in keys])

    P = Predictor()
    out = P.predict_grid(irs, wvs, sws, int(t.timestamp()))

    # ---- 대조 ----
    # ★ 합격 기준을 "최대 절대차 < 1e-5" 로 잡으면 안 된다 — 2단이 **트리 모델**이라
    #   출력이 연속이 아니다. ONNX 와 torch 의 1e-6 차이가 erode/dilate(창 안 최소값)
    #   에서 다른 화소를 뽑게 만들면, 그 특징이 분기 문턱을 넘어 확률이 1e-2 만큼
    #   **점프**한다. 회귀 모델의 허용오차 기준을 트리에 갖다 대면 멀쩡한 이식을
    #   불합격으로 읽는다.
    #   대신 이렇게 본다:
    #     (1) 압도적 다수 화소가 **차이 정확히 0** 이어야 한다 — 56열 배선의 증거
    #     (2) 구간이 갈린 화소는 전부 **문턱 근처**여야 한다 (경계 흔들림뿐)
    ok = True
    cat_ref, cat_got = ref["cat"], out["cat"]
    same = float((cat_ref == cat_got).mean())
    print(f"\n구간(cat) 일치율   {100*same:7.4f}%   "
          f"({int((cat_ref != cat_got).sum())} / {cat_ref.size} 화소 불일치)")
    ok &= same >= 0.999

    p_ref = ref["p01"].astype(np.float64)
    p_got = out["probs"][0.1].astype(np.float64)
    d = np.abs(p_ref - p_got)
    exact = float((d == 0).mean())
    print(f"P(>=0.1) 차이 정확히 0  {100*exact:7.4f}%   최대 {d.max():.3e}")
    ok &= exact >= 0.99                       # (1)

    thr01 = float(ref["thr01"])
    m = cat_ref != cat_got
    if m.any():
        far = float(np.abs(np.concatenate([p_ref[m], p_got[m]]) - thr01).max())
        print(f"구간이 갈린 화소의 P — 문턱({thr01:.4f})에서 최대 {far:.4f} 거리")
        ok &= far < 0.05                      # (2) 경계 흔들림인지 확인

    print(f"\n구간 분포 (원본 -> 배포)")
    for i in range(5):
        print(f"  {i}: {100*float((cat_ref==i).mean()):6.2f}%  ->"
              f"  {100*float((cat_got==i).mean()):6.2f}%")

    # 단조성은 배포 경로에서도 지켜져야 한다
    pr = out["probs"]
    bad = sum(int((pr[BINS[i]] > pr[BINS[i-1]] + 1e-12).sum()) for i in range(1, 4))
    print(f"\n단조성 위반 화소 {bad}개 (0 이어야 한다)")
    ok &= bad == 0

    print("\n" + ("일치. 배포 경로가 검증된 원본과 같은 답을 낸다."
                  if ok else "★ 불일치 — 특징 조립 순서를 원본과 대조하라."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
