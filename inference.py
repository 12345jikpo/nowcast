# -*- coding: utf-8 -*-
"""2단 추론 — U-Net 3개(ONNX) -> 특징 56개 -> LightGBM 구간 분류기 4개.

★ 이 파일은 `verify_20260720.py` 의 특징 조립을 **그대로 옮긴 것**이다.
  순서가 1비트라도 어긋나면 예외 없이 조용히 틀린 값이 나온다. 고칠 일이 생기면
  반드시 원본과 나란히 놓고 대조할 것. `validate_vs_verify.py` 가 그 대조를 자동화한다.

옮기면서 지킨 것
  (1) `psw` 는 min, `pwv` 는 max, `pir` 는 min  — 채널마다 방향이 다르다
  (2) `cos_sza` 특징은 [-1,1] 로 되돌려 넣는다 (U-Net 입력은 [0,1] 로 넣지만)
  (3) `elev` 는 관측소 고도가 아니라 **DEM** (학습이 DEM 으로 됐다)
  (4) 단조성 보정 — 낮은 문턱부터 P[i] = min(P[i], P[i-1])
  (5) 모델2·4 의 잔차 기준 채널은 7 (ONNX 로 구울 때 --res-idx 7 로 이미 박혔다)

torch 를 쓰지 않는다 — onnxruntime 만으로 돈다(무료 클라우드 RAM 1GB 대비).
"""
import json
import os

import cv2
import numpy as np
import lightgbm as lgb
import onnxruntime as ort

from solar import cos_sza

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# --------------------------------------------------------------- 격자 (config.py 사본)
# ★ 학습 캐시와 반드시 같아야 하는 값들이다. 원본은 nowcast/config.py.
IMG = 256
CROP_Y0, CROP_Y1 = 294, 806
CROP_X0, CROP_X1 = 188, 700
POOL = 2
IN_FRAMES = 4
STEP_S = 1800                      # 30분

# 원본 GK2A KO 격자 (표출 재투영에만 쓴다)
PIXEL_M = 2000.0
UL_EASTING = -899000.0             # 화소 (0,0) 중심의 easting
UL_NORTHING = 899000.0
LCC_PROJ = dict(proj="lcc", lat_1=30.0, lat_2=60.0, lat_0=38.0, lon_0=126.0,
                x_0=0.0, y_0=0.0, ellps="WGS84", units="m")

CH_NORM = {"IR105": (180.0, 330.0),
           "WV063": (190.0, 280.0),
           "SW038": (200.0, 330.0)}
DIFF_NORM = (-75.0, 5.0)           # WV063 − IR105
SW_DIFF_NORM = (-15.0, 70.0)       # SW038 − IR105

KS = [1, 3, 5, 7, 9, 13]           # build_features.KS
SLOPE_KS = [3, 9, 13]              # build_features2.SLOPE_KS
BINS = [0.1, 1.0, 3.0, 10.0]


def _unit(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _from_unit(u, lo, hi):
    return (lo + (hi - lo) * u).astype(np.float32)


def multiscale(f, mode):
    """(H,W) -> {k: (H,W)}. min=erode, max=dilate, mean=blur. build_features.multiscale 사본."""
    out = {}
    for k in KS:
        if k == 1:
            out[k] = f
        elif mode == "min":
            out[k] = cv2.erode(f, np.ones((k, k), np.uint8),
                               borderType=cv2.BORDER_REPLICATE)
        elif mode == "max":
            out[k] = cv2.dilate(f, np.ones((k, k), np.uint8),
                                borderType=cv2.BORDER_REPLICATE)
        else:
            out[k] = cv2.blur(f, (k, k), borderType=cv2.BORDER_REPLICATE)
    return out


class Predictor:
    """무겁게 한 번만 만들고 재사용한다 (Streamlit 에서는 @st.cache_resource)."""

    def __init__(self, models_dir=MODELS):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2          # 무료 클라우드는 코어가 적다
        p = ["CPUExecutionProvider"]
        self.m1 = ort.InferenceSession(os.path.join(models_dir, "unet_ir105.onnx"),
                                       so, providers=p)
        self.m2 = ort.InferenceSession(os.path.join(models_dir, "unet_wvdiff.onnx"),
                                       so, providers=p)
        self.m4 = ort.InferenceSession(os.path.join(models_dir, "unet_swdiff.onnx"),
                                       so, providers=p)

        self.meta = json.load(open(os.path.join(models_dir, "lgbm_v2_bins_dem.json"),
                                   encoding="utf-8"))
        self.cols = self.meta["features"]
        self.clfs = {b: lgb.Booster(model_file=os.path.join(
            models_dir, f"lgbm_v2_bins_dem_ge{b}.txt")) for b in BINS}
        self.thrs = {b: self.meta["results"][str(b)]["fixed"]["thr"] for b in BINS}

        d = np.load(os.path.join(models_dir, "dem_256.npz"))
        self.dem_n = d["norm"].astype(np.float32)        # U-Net 채널 10 (0~1)
        self.elev = d["elev_m"].astype(np.float32)       # LightGBM 특징 (m)
        ll = np.load(os.path.join(models_dir, "latlon_256.npz"))
        self.lat = ll["lat"].astype(np.float64)
        self.lon = ll["lon"].astype(np.float64)

        # 경사는 정적이라 한 번만 만든다 (build_features2.slope_fields 사본)
        self.slopes = {}
        for k in SLOPE_KS:
            sm = cv2.blur(self.elev, (k, k), borderType=cv2.BORDER_REPLICATE)
            gy, gx = np.gradient(sm)
            self.slopes[k] = np.hypot(gx, gy).astype(np.float32)

    # ------------------------------------------------------------------ 1단
    def unets(self, irs, wvs, sws, t_epoch):
        """관측 BT 4프레임 x 3채널 -> 30분 뒤 예측장 (pir, pwv, psw).

        irs/wvs/sws : (4,256,256) 밝기온도(K), 과거->현재 순서
        t_epoch     : 예측 대상 시각(= 마지막 입력 프레임 + 30분)의 epoch 초
        """
        dwv, dsw = wvs - irs, sws - irs
        lo1, hi1 = CH_NORM["IR105"]
        x1 = _unit(irs, lo1, hi1)[None]
        x2 = np.concatenate([x1, _unit(dwv, *DIFF_NORM)[None]], 1)

        # ★ cos(SZA) 를 t0(마지막 입력)와 t+30(대상) 둘 다 넣는다.
        #   t0 만 주면 "지금 밤"은 알아도 "30분 뒤 낮"을 몰라 새벽·황혼에 잔차가 깨진다.
        sz0 = ((cos_sza(t_epoch - STEP_S, self.lat, self.lon) + 1) / 2
               ).astype(np.float32)[None, None]
        sz1 = ((cos_sza(t_epoch, self.lat, self.lon) + 1) / 2
               ).astype(np.float32)[None, None]
        x4 = np.ascontiguousarray(np.concatenate(
            [x1, _unit(dsw, *SW_DIFF_NORM)[None], sz0, sz1,
             self.dem_n[None, None]], 1))

        pir = _from_unit(self.m1.run(None, {"frames": x1})[0][0, 0], lo1, hi1)
        pwv = _from_unit(self.m2.run(None, {"frames": x2})[0][0, 0], *DIFF_NORM)
        psw = _from_unit(self.m4.run(None, {"frames": x4})[0][0, 0], *SW_DIFF_NORM)
        return pir, pwv, psw, dwv, dsw, sz1

    # ------------------------------------------------------------------ 특징
    def features(self, irs, wvs, sws, t_epoch):
        """(65536, 56) 특징 행렬. ★ 순서는 meta["features"] 를 따른다."""
        pir, pwv, psw, dwv, dsw, sz1 = self.unets(irs, wvs, sws, t_epoch)

        f = {}
        for mode, arr, pre in (("min", pir, "pir_min"), ("mean", pir, "pir_mean"),
                               ("max", pwv, "pwv_max"), ("mean", pwv, "pwv_mean"),
                               ("min", psw, "psw_min"), ("mean", psw, "psw_mean")):
            ms = multiscale(arr, mode)
            for k in KS:
                f[f"{pre}{k}"] = ms[k]

        # 관측(마지막 프레임) 쪽은 9화소 창만 쓴다
        f["oir_min9"] = multiscale(irs[3], "min")[9]
        f["oir_mean9"] = multiscale(irs[3], "mean")[9]
        f["owv_max9"] = multiscale(dwv[3], "max")[9]
        f["owv_mean9"] = multiscale(dwv[3], "mean")[9]
        f["osw_min9"] = multiscale(dsw[3], "min")[9]
        f["osw_mean9"] = multiscale(dsw[3], "mean")[9]
        # 예측 − 관측 = '앞으로 30분 동안의 변화'
        for pre in ("pir_min9", "pir_mean9", "pwv_max9", "pwv_mean9",
                    "psw_min9", "psw_mean9"):
            o = pre.replace("pir_", "oir_").replace("pwv_", "owv_").replace("psw_", "osw_")
            f["d_" + pre] = f[pre] - f[o]

        import datetime as dt
        t = dt.datetime.fromtimestamp(t_epoch, dt.timezone(dt.timedelta(hours=9)))
        hh = t.hour + t.minute / 60.0
        one = np.ones((IMG, IMG), np.float32)
        f["hr_sin"] = one * np.sin(2 * np.pi * hh / 24)
        f["hr_cos"] = one * np.cos(2 * np.pi * hh / 24)
        f["cos_sza"] = (sz1[0, 0] * 2 - 1).astype(np.float32)    # ★ [-1,1] 로 되돌린다
        f["domain_mean_ir"] = one * float(irs[3].mean())
        f["elev"] = self.elev
        for k in SLOPE_KS:
            f[f"slope_{k}"] = self.slopes[k]

        return np.stack([f[c].ravel() for c in self.cols], 1).astype(np.float32), pir

    # ------------------------------------------------------------------ 2단
    def predict_grid(self, irs, wvs, sws, t_epoch):
        """-> dict(cat (256,256) int8, probs {b: (256,256)}, pir (256,256))"""
        X, pir = self.features(irs, wvs, sws, t_epoch)
        P = {b: self.clfs[b].predict(X) for b in BINS}
        for i in range(1, len(BINS)):                    # ★ 단조성 보정
            P[BINS[i]] = np.minimum(P[BINS[i]], P[BINS[i - 1]])
        cat = np.zeros(X.shape[0], np.int8)
        for i, b in enumerate(BINS):
            cat[P[b] >= self.thrs[b]] = i + 1
        return dict(cat=cat.reshape(IMG, IMG),
                    probs={b: P[b].reshape(IMG, IMG).astype(np.float32) for b in BINS},
                    pir=pir)

    # ------------------------------------------------------------------ 표출
    def latlon_bounds(self):
        """오버레이용 위경도 사각형 [[남,서],[북,동]]."""
        return [[float(self.lat.min()), float(self.lon.min())],
                [float(self.lat.max()), float(self.lon.max())]]

    def to_latlon(self, grid, out_hw=512):
        """LCC 격자 -> 위경도 정사각 격자로 재투영. 격자 밖은 NaN.

        ★ 그냥 ImageOverlay 에 붙이면 안 된다. 이 격자는 **람베르트 정각원추**라
          위경도 사각형이 아니다. 도메인 가장자리에서 경선이 몇 도씩 기울어 있어
          그대로 붙이면 구름이 실제 위치에서 눈에 띄게 어긋난다.
          여기서는 목표 위경도 화소마다 LCC 로 정투영해 원본 화소를 찾는다(역매핑).

        최근접 표본을 쓴다 — 구간(cat)이 정수 범주라 보간하면 없는 값이 생긴다.
        """
        from pyproj import Transformer
        (s, w), (n, e) = self.latlon_bounds()
        la = np.linspace(n, s, out_hw)                 # 위에서 아래로 (영상 좌표)
        lo = np.linspace(w, e, out_hw)
        LO, LA = np.meshgrid(lo, la)

        tr = Transformer.from_crs("EPSG:4326", LCC_PROJ, always_xy=True)
        x, y = tr.transform(LO.ravel(), LA.ravel())
        col = (x - UL_EASTING) / PIXEL_M               # 원본 900x900 열
        row = (UL_NORTHING - y) / PIXEL_M
        j = np.round((col - CROP_X0 - (POOL - 1) / 2) / POOL).astype(np.int32)
        i = np.round((row - CROP_Y0 - (POOL - 1) / 2) / POOL).astype(np.int32)

        ok = (i >= 0) & (i < IMG) & (j >= 0) & (j < IMG)
        out = np.full(i.shape, np.nan, np.float32)
        out[ok] = np.asarray(grid, np.float32)[i[ok], j[ok]]
        return out.reshape(out_hw, out_hw)

    # ------------------------------------------------------------------ 지점
    def pixel_of(self, lat, lon):
        """위경도 -> 격자 (i, j). 격자 밖이면 None.

        ★ pyproj 로 LCC 역변환을 하는 대신 lat/lon 격자에서 최근접을 찾는다.
          두 방법이 같은 답을 주는데, 이쪽은 의존성이 하나 줄고 격자 밖 판정이 쉽다.
          4km 격자라 최근접 오차는 최대 2.8km — 시차(parallax)가 11km 인 문제에서
          무시할 수 있다.
        """
        d2 = (self.lat - lat) ** 2 + ((self.lon - lon) * np.cos(np.radians(lat))) ** 2
        i, j = np.unravel_index(int(d2.argmin()), d2.shape)
        # 최근접이 너무 멀면 도메인 밖이다 (격자 간격 약 0.036도)
        if d2[i, j] > 0.1 ** 2:
            return None
        return int(i), int(j)
