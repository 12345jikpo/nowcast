# -*- coding: utf-8 -*-
"""태양천정각(SZA) — SW038 채널을 쓰려면 반드시 함께 넣어야 하는 축.

왜 필요한가
  SW038(3.8μm)은 **주야로 물리량이 다르다**.
    야간 : 순수 방출. 물방울 구름은 방출률이 낮아 BT 가 IR105 보다 2~4K 낮다
           -> `SW038 − IR105` 강한 음수 = 하층 물구름 (안개/하층운 탐지의 표준 원리)
    주간 : 태양반사가 섞인다. 반사는 **구름 유효입자반경**에 지배되어 물방울이 커질수록
           어두워지고, 유효반경 ~14μm 이 따뜻한 비의 개시 문턱이다.
  같은 채널이 낮과 밤에 정반대 의미를 갖는다. 태양천정각 없이 U-Net 에 넣으면
  모델이 둘을 구별할 방법이 없어 **평균으로 뭉갠다**. 잔차 학습 구조라 더 취약하다.
  ★ 이걸 빠뜨리면 "SW038 이 안 든다"는 결론이 나올 텐데, 채널 탓이 아니게 된다.

왜 받지 않고 계산하나
  위성 자료가 아니라 시각·위경도만으로 나오는 천문 계산이다. 용량 0.
  캐시로 구울 필요도 없다 — 256x256 한 장이 마이크로초라 __getitem__ 에서 즉석 계산이 싸다.
  (구우면 IR105 캐시와 같은 1.06GiB 를 계산 가능한 값에 쓰는 셈)

알고리즘
  NOAA 저정밀 태양위치식. 적위 오차 ~0.01도로 이 용도엔 차고 넘친다.
  cos(SZA) 를 그대로 쓴다 — 각도(0~180도)보다 낫다:
    · 밤이 음수, 낮이 양수로 **부호가 주야를 가른다**
    · 정오 부근에서 완만해서 반사 성분 세기와 거의 비례한다
    · 이미 [-1,1] 이라 정규화가 필요 없다

  python solar.py        # 자체 검증(기하학적 항등식과 대조)
"""
import numpy as np

__all__ = ["cos_sza", "cos_sza_grid"]

_J2000 = 946728000.0        # 2000-01-01 12:00 UTC 의 epoch 초 (율리우스력 2451545.0)
_D2R = np.pi / 180.0


def cos_sza(t_epoch, lat, lon):
    """cos(태양천정각). 1 = 머리 위, 0 = 지평선, 음수 = 밤.

    t_epoch : epoch 초(스칼라 또는 배열). 캐시 meta["times"] 가 이 단위다.
    lat, lon: 도 단위. t_epoch 와 브로드캐스트된다
              (시각 스칼라 + 격자 (H,W) -> (H,W) 로 나온다).
    """
    n = (np.asarray(t_epoch, np.float64) - _J2000) / 86400.0     # J2000 이후 일수

    L = (280.460 + 0.9856474 * n) % 360.0                        # 평균황경
    g = ((357.528 + 0.9856003 * n) % 360.0) * _D2R                # 평균근점이각
    lam = (L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) * _D2R  # 황경
    eps = (23.439 - 3.6e-7 * n) * _D2R                            # 황도경사

    sin_dec = np.sin(eps) * np.sin(lam)                           # 적위
    cos_dec = np.sqrt(np.maximum(0.0, 1.0 - sin_dec ** 2))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))       # 적경(rad)

    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0          # 그리니치 항성시(h)
    # 시간각 = 지방항성시 − 적경
    H = (gmst * 15.0 + np.asarray(lon, np.float64)) * _D2R - ra

    phi = np.asarray(lat, np.float64) * _D2R
    return (np.sin(phi) * sin_dec +
            np.cos(phi) * cos_dec * np.cos(H))


def cos_sza_grid(t_epoch, lat2d, lon2d, dtype=np.float32):
    """256x256 격자용. 시각 스칼라 -> (H,W), 시각 배열 -> (T,H,W)."""
    t = np.atleast_1d(np.asarray(t_epoch, np.float64))
    out = cos_sza(t[:, None, None], lat2d[None], lon2d[None]).astype(dtype)
    return out[0] if np.isscalar(t_epoch) or np.ndim(t_epoch) == 0 else out


# --------------------------------------------------------------------- 자체 검증
def _selftest():
    """기하학적 항등식과 대조한다 — 외부 표가 필요 없다.

    지점 위도 phi, 태양 적위 dec 일 때 **남중 고도 = 90 − |phi − dec|**.
    하지(dec=+23.44) / 동지(dec=−23.44) 서울(37.57N)에서 각각 75.9도 / 29.0도여야 한다.
    """
    import datetime as dt
    KST = dt.timezone(dt.timedelta(hours=9))

    def ep(y, m, d, hh, mm):
        return dt.datetime(y, m, d, hh, mm, tzinfo=KST).timestamp()

    lat, lon = 37.5665, 126.9780        # 서울
    print("1) 남중 고도 — 기하학적 정답과 대조 (서울 37.57N)")
    for label, (y, m, d), expect in (("하지", (2024, 6, 21), 90 - abs(lat - 23.44)),
                                     ("동지", (2024, 12, 21), 90 - abs(lat + 23.44)),
                                     ("춘분", (2024, 3, 20), 90 - abs(lat - 0.0))):
        # 그날의 최대 고도를 1분 간격으로 찾는다(남중시각을 몰라도 된다)
        ts = np.array([ep(y, m, d, 0, 0) + 60 * k for k in range(1440)])
        elev = np.degrees(np.arcsin(np.clip(cos_sza(ts, lat, lon), -1, 1)))
        got = elev.max()
        k = int(elev.argmax())
        print(f"   {label} {y}-{m:02d}-{d:02d}  남중 {got:5.2f}도 "
              f"(정답 {expect:5.2f}도, 오차 {got-expect:+.2f}도)  "
              f"남중시각 {k//60:02d}:{k%60:02d} KST")

    print("\n2) 주야 판별 — cos(SZA) 부호")
    for label, (y, m, d, hh, mm) in (("한여름 정오", (2024, 7, 15, 12, 30)),
                                     ("한여름 자정", (2024, 7, 15,  0,  0)),
                                     ("한겨울 정오", (2024, 1, 15, 12, 30)),
                                     ("한겨울 자정", (2024, 1, 15,  0,  0))):
        c = float(cos_sza(ep(y, m, d, hh, mm), lat, lon))
        print(f"   {label:10s} {y}-{m:02d}-{d:02d} {hh:02d}:{mm:02d} KST  "
              f"cos(SZA) = {c:+.3f}  ({'낮' if c > 0 else '밤'})")

    print("\n3) 도메인 격자 — 남북 끝의 낮 길이 차 (여름엔 북쪽이 길다)")
    for label, la in (("북단 40.9N", 40.9), ("남단 31.3N", 31.3)):
        ts = np.array([ep(2024, 7, 1, 0, 0) + 600 * k for k in range(144)])
        day = (cos_sza(ts, la, 126.0) > 0).sum() * 10 / 60.0
        print(f"   {label}  7/1 낮 길이 {day:.1f} 시간")

    print("\n4) 격자 함수 모양 확인")
    lat2d = np.linspace(40.9, 31.3, 256)[:, None] * np.ones((1, 256))
    lon2d = np.ones((256, 1)) * np.linspace(119.6, 132.1, 256)[None]
    g = cos_sza_grid(ep(2024, 7, 1, 6, 30), lat2d, lon2d)
    print(f"   6:30 KST  shape {g.shape}  범위 {g.min():+.3f} ~ {g.max():+.3f}  "
          f"(동쪽이 먼저 밝음: 서 {g[128,0]:+.3f} < 동 {g[128,-1]:+.3f})")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
