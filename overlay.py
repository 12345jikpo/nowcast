# -*- coding: utf-8 -*-
"""예측 격자 -> 지도에 얹을 반투명 PNG (data URI).

두 겹으로 그린다 — 시안의 '레이더/위성 영상' 패널이 배경 + 강수 셀 블롭 구조다.
  1) 구름  : 30분 뒤 예측 IR105 밝기온도. 차가울수록 하얗고 진하다.
  2) 강수  : 구간 1~4 를 색으로. 구름 위에 얹는다.

★ 정량 mm/h 는 어디에도 안 쓴다. 색은 **구간**이지 강수량이 아니다.
"""
import base64
import io

import numpy as np
from PIL import Image

from levels import LEVEL_COLORS

# 구름 투명도 눈금 (밝기온도 K).
#   280K 이상 = 지표/맑은 하늘 -> 완전 투명
#   220K 이하 = 깊은 대류운 정상 -> 거의 불투명
#   지도(도로·지명)가 비쳐야 하므로 최대 알파를 0.85 로 묶는다.
BT_CLEAR, BT_THICK = 280.0, 220.0
CLOUD_MAX_ALPHA = 0.85


def _png_data_uri(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cloud_png(bt):
    """예측 밝기온도 (H,W, NaN=격자밖) -> 흰 구름 PNG."""
    a = np.clip((BT_CLEAR - bt) / (BT_CLEAR - BT_THICK), 0.0, 1.0)
    a = np.nan_to_num(a, nan=0.0) * CLOUD_MAX_ALPHA
    h, w = a.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    # 차가울수록 순백에 가깝게, 따뜻한 얇은 구름은 약간 푸르스름하게
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return _png_data_uri(rgba)


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rain_png(cat, alpha=0.72):
    """구간 격자 (H,W, NaN=격자밖) -> 색 PNG. 구간 0(없음)은 투명."""
    c = np.nan_to_num(cat, nan=0.0).round().astype(np.int16)
    h, w = c.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    for lv in (1, 2, 3, 4):
        m = c == lv
        if not m.any():
            continue
        r, g, b = _hex_rgb(LEVEL_COLORS[lv])
        rgba[m] = (r, g, b, int(alpha * 255))
    return _png_data_uri(rgba)
