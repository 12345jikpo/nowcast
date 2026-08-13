# -*- coding: utf-8 -*-
"""30분 뒤에 비가 올까? — GK2A 초단기 강수 예측 (Streamlit)

화면 구성 (design_handoff_rain30 시안)
  홈    eyebrow · 타이틀 · 검색 · 최근  — ★ 지도 없음
  결과  헤더 · 위성 영상(지도+예측 구름+강수 구간) · 구간 · 문구 · 게이지

시안의 '지역 검색'은 별도 풀스크린 화면인데, Streamlit 은 화면 전환이 어색해서
홈 안에 인라인으로 두고 결과만 따로 뺐다.
"""
import datetime as dt

import streamlit as st
import folium
from streamlit_folium import st_folium

import geocode
from levels import LEVELS
from backend import predict_point, in_domain, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, KST

st.set_page_config(page_title="30분 뒤에 비가 올까?", page_icon="🌧",
                   layout="centered", initial_sidebar_state="collapsed")

# 시안의 '날씨 일러스트' (96x66). 자리표시자였던 CSS 도형을 SVG 로 바꿨다.
#   해 #ffe58a 뒤 · 흰 구름 앞 · 빗방울 #8ec6f2 3개 14도 기울임 — 색·치수는 시안 그대로.
#   currentColor 를 안 쓰고 값을 박은 건 이 그림이 테마와 무관하게 고정이기 때문.
WX_SVG = """
<svg class="wx" viewBox="0 0 96 66" fill="none" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="구름과 해, 비">
  <g stroke="#ffe58a" stroke-width="3" stroke-linecap="round">
    <path d="M72 2v6M72 32v6M87 20h6M51 20h6M82.6 9.4l4.2-4.2M57.2 34.8l4.2-4.2M82.6 30.6l4.2 4.2M57.2 5.2l4.2 4.2"/>
  </g>
  <circle cx="72" cy="20" r="11" fill="#ffe58a"/>
  <g fill="#ffffff">
    <circle cx="33" cy="30" r="13"/>
    <circle cx="52" cy="27" r="15"/>
    <rect x="14" y="33" width="55" height="17" rx="8.5"/>
  </g>
  <g fill="#8ec6f2">
    <rect x="25" y="52" width="5" height="13" rx="2.5" transform="rotate(14 27.5 58.5)"/>
    <rect x="40" y="52" width="5" height="13" rx="2.5" transform="rotate(14 42.5 58.5)"/>
    <rect x="55" y="52" width="5" height="13" rx="2.5" transform="rotate(14 57.5 58.5)"/>
  </g>
</svg>
"""

PRESETS = [
    ("서울 성동구 성수동2가", 37.5445, 127.0557),
    ("부산 해운대구 우동", 35.1631, 129.1637),
    ("제주 제주시 이도이동", 33.4996, 126.5312),
    ("강원 강릉시 포남동", 37.7519, 128.8761),
]

CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css');
:root{
  --bg:#eaf4ff; --ink:#12324d; --ink2:#37536b; --sub:#5c7f9b; --muted:#8aa8c2;
  --surface:#ffffff; --accent:#ffe58a; --accent2:#ffd34d; --blue:#8ec6f2;
  --panel:#dbe9f6; --line:rgba(18,50,77,.14); --line2:rgba(18,50,77,.2);
  --mono:ui-monospace,Menlo,"SFMono-Regular",Consolas,monospace;
}
html,body,.stApp{background:var(--bg)!important;
  font-family:"Pretendard Variable",Pretendard,system-ui,sans-serif;}
.block-container{max-width:440px!important;padding:14px 20px 40px!important;}
#MainMenu,footer,header{visibility:hidden;}

/* ★ p 기반 스타일은 !important 가 필요하다 — Streamlit 의 .stMarkdown p 규칙이
   클래스 선택자를 이겨서 헤드라인이 24 -> 16px 로 눌린다. */
.eyebrow{font-family:var(--mono);font-size:11px!important;letter-spacing:2px;
         color:var(--muted);margin:0 0 6px;}
h1.title{font-size:30px!important;line-height:1.24;font-weight:800;
         letter-spacing:-1.4px;color:var(--ink);margin:0 0 14px;}
.rule{height:1px;background:var(--ink);opacity:.14;margin:14px 0;}

/* 타이틀 행 — 시안: 글씨 왼쪽 flex:1, 날씨 일러스트 오른쪽 96x66 flex:none */
.titlerow{display:flex;align-items:center;gap:8px;}
.titlerow h1.title{flex:1;margin:0;}
.titlerow .wx{flex:none;width:96px;height:66px;}
.hint{font-size:13px!important;color:var(--sub);margin:10px 0 0;}
.place{font-size:14.5px!important;font-weight:700;color:var(--ink);margin:0 0 10px;}

.meta{display:flex;justify-content:space-between;font-family:var(--mono);
      font-size:11px;color:var(--muted);margin:10px 0 6px;}
.bignum{font-size:44px;line-height:.95;font-weight:800;letter-spacing:-2px;
        color:var(--ink);display:flex;align-items:baseline;gap:6px;}
.bignum .unit{font-size:19px;font-weight:700;color:var(--muted);letter-spacing:0;}
.headline{font-size:24px!important;line-height:1.32;font-weight:800;
          letter-spacing:-0.6px;color:var(--ink);margin:14px 0 0;text-wrap:pretty;}
.subcopy{font-size:13.5px!important;line-height:1.55;font-weight:500;
         color:var(--sub);margin:9px 0 0;text-wrap:pretty;}
.gauge{display:flex;gap:8px;align-items:center;margin:18px 0 0;}
.gauge .cell{flex:1;height:5px;border-radius:3px;background:var(--line);}
.gauge .cell.on{background:var(--ink);}
.gauge .lab{font-family:var(--mono);font-size:10px;color:var(--muted);
            flex:none;padding-left:4px;}
.badge{display:inline-block;background:var(--accent2);color:var(--ink);
       border-radius:999px;padding:6px 12px;font-size:12px;font-weight:800;
       margin:0 0 10px;}
.stub{background:var(--surface);border:1px dashed var(--line2);border-radius:4px;
      padding:10px 12px;font-family:var(--mono);font-size:10.5px!important;
      color:var(--muted);margin:16px 0 0;line-height:1.7;}

/* 지도 범례 — ★ 한 줄에 다 넣으면 모바일 폭에서 '강한 비'만 다음 줄로 떨어져
   어긋나 보인다. 설명 한 줄 / 색 칩 한 줄로 **일부러** 두 줄로 나눈다.
   칩은 space-between 이라 폭이 변해도 줄바꿈 없이 고르게 퍼진다. */
.legend{margin:8px 0 0;font-family:var(--mono);font-size:10px;color:var(--muted);}
.legend .lgnote{margin-bottom:5px;}
.legend .lgchips{display:flex;justify-content:space-between;gap:6px;}
.legend .lgchips span{white-space:nowrap;}
.legend i{width:9px;height:9px;border-radius:2px;display:inline-block;
          margin-right:4px;vertical-align:-1px;}

div.stButton>button{width:100%;height:46px;border-radius:4px;
  border:2px solid var(--ink);background:var(--surface);color:var(--ink);
  font-weight:700;font-size:14px;}
/* 호버는 노란빛 대신 옅은 회색 — 노랑은 '선택됨'을 뜻하는 색이라 마우스만
   올려도 노래지면 이미 고른 것처럼 읽힌다. */
div.stButton>button:hover{background:#eef2f7;
  border-color:var(--ink);color:var(--ink);}
div.stButton>button:active{background:#e4eaf1;}
div.stButton>button[kind="primary"]{background:var(--ink);color:var(--bg);}

div[data-testid="stTextInput"] input{height:52px;background:var(--surface);
  border-radius:4px;font-size:15px!important;font-weight:600;color:var(--ink);
  padding:0 14px;}
div[data-testid="stTextInput"] div[data-baseweb="input"]{
  border:2px solid var(--ink)!important;border-radius:4px;background:var(--surface);}
div[data-testid="stTextInput"] input::placeholder{color:var(--muted);font-weight:500;}

/* ---------- 로딩: 구름이 아래에서 차오른다 ---------- */
.loadwrap{display:flex;flex-direction:column;align-items:center;gap:10px;
          padding:26px 0 18px;}
.loadwrap svg{width:120px;height:82px;}
.loadlab{font-family:var(--mono);font-size:11px!important;color:var(--muted);
         margin:0;letter-spacing:.5px;}
/* 탐색 구간(약 3.5초)은 받은 게 없어 게이지가 0% 에 멈춰 있다.
   진행률을 거짓으로 채우는 대신 구름만 은은히 숨쉬게 해 '살아있음'을 알린다. */
.loadwrap.searching svg{animation:cloudpulse 1.4s ease-in-out infinite;}
@keyframes cloudpulse{0%,100%{opacity:1;}50%{opacity:.42;}}

/* ---------- 결과 배경: 강도에 맞춘 비 ---------- */
/* 시안 지정 루프 — 약한 비 1.9s / 보통 1.15s / 폭우 0.62s. 그 사이는 보간했다. */
/* ★ 그라디언트 타일링(반복 배경)으로는 비가 안 된다 — 모든 줄이 같은 길이·속도로
   움직여서 '움직이는 벽지'로 읽힌다. 물방울을 개별 요소로 두고 속도·길이·투명도를
   제각각 줘야 깊이감이 생긴다. 기울기는 컨테이너를 통째로 14도 돌려서 준다
   (물방울마다 x 이동을 계산하는 것보다 싸고, 타이틀 일러스트의 빗방울 각도와 같다). */
.rainbg{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;}
.rainbg .fld{position:absolute;inset:-35%;transform:rotate(14deg);}
.rainbg s{position:absolute;top:0;display:block;border-radius:1px;
  text-decoration:none;
  background:linear-gradient(to bottom,rgba(142,198,242,0),rgba(142,198,242,1));
  animation:drop linear infinite;will-change:transform;}
@keyframes drop{from{transform:translateY(-18vh);}to{transform:translateY(118vh);}}
/* 내용은 비 위에 온다 */
.block-container{position:relative;z-index:1;}
/* ★ 시안 요구: 모션 민감 사용자에게는 애니메이션을 멈춘다 */
@media (prefers-reduced-motion: reduce){
  .rainbg i,.loadwrap.searching svg{animation:none;}
}
</style>
"""


def cloud_loader(frac, label, searching=False):
    """구름 실루엣이 아래에서 흰색으로 차오르는 로딩 표시.

    ★ clipPath 는 자식 도형들의 **합집합**으로 자른다 — 타이틀 일러스트와 똑같은
      원 2개 + 둥근 사각 1개를 그대로 재사용해 모양을 맞췄다.
      구름 바깥은 아예 칠하지 않아 투명이다(페이지 배경이 그대로 비친다).

    색: 빈 곳 #dbe9f6 / 채움 #ffffff (사용자 선택).
    ★ 둘 다 밝아서 25% 부근까지는 차오르는 게 잘 안 보인다 — 알고 고른 값이다.
      대비를 올리려면 채움을 #12324d(잉크)로 바꾸거나 네이비 테두리를 두르면 된다.
    """
    frac = max(0.0, min(1.0, frac))
    y = 56 - 44 * frac                       # 구름 아랫변 56 -> 윗변 12
    return f"""
<div class="loadwrap{' searching' if searching else ''}">
  <svg viewBox="0 0 96 66" xmlns="http://www.w3.org/2000/svg" role="img"
       aria-label="자료 받는 중 {int(frac*100)}퍼센트">
    <defs><clipPath id="cloudclip">
      <circle cx="33" cy="30" r="13"/><circle cx="52" cy="27" r="15"/>
      <rect x="14" y="33" width="55" height="17" rx="8.5"/>
    </clipPath></defs>
    <g clip-path="url(#cloudclip)">
      <rect x="0" y="0" width="96" height="66" fill="#dbe9f6"/>
      <rect x="0" y="{y:.1f}" width="96" height="{66 - y:.1f}" fill="#ffffff"/>
    </g>
  </svg>
  <p class="loadlab">{label}</p>
</div>"""


# 단계별 비 — (물방울 수, 낙하 초, 길이 px 범위, 굵기 px, 불투명도 범위).
# 낙하 시간은 시안이 제시한 값(약한 1.9 / 보통 1.15 / 폭우 0.62초)을 기준으로 삼되,
# 물방울마다 ±25% 흔들어 줄이 뭉치지 않게 한다.
RAIN_STYLE = {
    1: (52, 1.90, (8, 16), 1.2, (0.20, 0.45)),
    2: (96, 1.50, (10, 20), 1.4, (0.25, 0.55)),
    3: (170, 1.15, (13, 26), 1.6, (0.30, 0.65)),
    4: (280, 0.62, (18, 34), 2.0, (0.35, 0.80)),
}


def rain_bg(lv):
    """결과 화면 배경 비. 강할수록 촘촘·길고·빠르다. 0단계는 비 없음.

    ★ 난수는 **고정 시드**로 뽑는다 — Streamlit 은 상호작용마다 스크립트를 다시
      돌리므로, 매번 새로 뽑으면 화면을 건드릴 때마다 비 배치가 튄다.
    """
    if lv not in RAIN_STYLE:
        return ""
    import random
    n, dur, (lmin, lmax), w, (amin, amax) = RAIN_STYLE[lv]
    rng = random.Random(lv * 7919)
    drops = []
    for _ in range(n):
        left = rng.uniform(-4, 104)                  # 회전 때문에 화면 밖까지 깐다
        length = rng.uniform(lmin, lmax)
        d = dur * rng.uniform(0.78, 1.25)            # 속도 편차 = 깊이감
        delay = -rng.uniform(0, d)                   # 음수 지연: 처음부터 꽉 차 보인다
        op = rng.uniform(amin, amax)
        drops.append(
            f'<s style="left:{left:.2f}%;height:{length:.0f}px;width:{w}px;'
            f'animation-duration:{d:.2f}s;animation-delay:{delay:.2f}s;'
            f'opacity:{op:.2f}"></s>')
    return ('<div class="rainbg" aria-hidden="true"><div class="fld">'
            + "".join(drops) + '</div></div>')


def safe_msg(e):
    """오류 메시지를 화면에 띄우기 전에 소독한다.

    ★ 두 가지가 위험하다 —
      (1) 인증키 노출: requests 예외 메시지는 **URL 전문**을 담는데 기상청 API 는
          authKey 를 쿼리스트링으로 받는다. 그대로 띄우면 키가 화면에 찍힌다.
      (2) HTML 주입: unsafe_allow_html=True 로 넣으므로 메시지 안의 태그가 살아난다.
    """
    import html
    import re
    s = re.sub(r"(authKey|apiKey|serviceKey)=[^&\s\"']+", r"\1=***",
               str(e), flags=re.I)
    return html.escape(s)


st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ 상태
if "view" not in st.session_state:
    st.session_state.view = "home"
if "sel" not in st.session_state:
    st.session_state.sel = None
if "recent" not in st.session_state:
    # ★ 시안은 localStorage 영속인데 Streamlit 은 서버 세션이라 새로고침하면 날아간다.
    st.session_state.recent = list(PRESETS)


def go_result(nm, la, lo):
    st.session_state.sel = (nm, la, lo)
    rec = [r for r in st.session_state.recent if r[0] != nm]
    st.session_state.recent = [(nm, la, lo)] + rec[:4]
    st.session_state.view = "result"


# ================================================================== 홈
def home():
    st.markdown('<p class="eyebrow">NOWCAST / +30MIN</p>', unsafe_allow_html=True)
    st.markdown('<div class="titlerow"><h1 class="title">30분 뒤에<br>비가 올까?'
                f'</h1>{WX_SVG}</div>', unsafe_allow_html=True)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    q = st.text_input("지역 검색", key="q", placeholder="지역 검색 · 예) 성수동",
                      label_visibility="collapsed")

    if q.strip():
        try:
            hits = _search(q)
        except geocode.GeocodeError as e:
            hits = []
            st.markdown(f'<p class="hint">{safe_msg(e)}</p>', unsafe_allow_html=True)
        except Exception as e:                          # noqa: BLE001
            hits = []
            st.markdown(f'<p class="hint">검색 실패 — {type(e).__name__}</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="eyebrow">검색 결과 {len(hits)}</p>',
                        unsafe_allow_html=True)
        for i, h in enumerate(hits):
            label = h["name"] + (f'  ·  {h["detail"]}' if h["detail"] else "")
            if st.button(label, key=f"h{i}"):
                go_result(h["name"], h["lat"], h["lon"])
                st.rerun()

    st.markdown('<p class="eyebrow">최근</p>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (nm, la, lo) in enumerate(st.session_state.recent):
        if cols[i % 2].button(nm, key=f"r{i}", width="stretch"):
            go_result(nm, la, lo)
            st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def _search(q):
    return geocode.search(q)


# ================================================================== 결과
def result():
    name, lat, lon = st.session_state.sel

    if st.button("← 다른 지역 보기", key="back"):
        st.session_state.view = "home"
        st.rerun()
    st.markdown(f'<p class="place">{name}</p>', unsafe_allow_html=True)

    if not in_domain(lat, lon):
        st.markdown(
            '<p class="headline">예측 범위 밖입니다</p>'
            '<p class="subcopy">위성 도메인(31.3~40.9N, 119.6~132.1E) 안이어야 '
            '합니다. 경계 밖은 상류에서 들어올 구름을 못 봐서 예측이 성립하지 '
            '않습니다.</p>', unsafe_allow_html=True)
        return

    slot = st.empty()
    # 탐색 구간 — 아직 받은 게 없어 0%. 깜빡임으로만 진행 중임을 알린다.
    slot.markdown(cloud_loader(0.0, "위성 자료 찾는 중", searching=True),
                  unsafe_allow_html=True)

    def prog(n, tot, lab):
        slot.markdown(cloud_loader(n / tot, f"GK2A {n}/{tot} · {lab}"),
                      unsafe_allow_html=True)

    try:
        r = predict_point(lat, lon, progress=prog)
    except Exception as e:                              # noqa: BLE001
        slot.empty()
        st.markdown(f'<p class="headline">예측을 못 만들었습니다</p>'
                    f'<p class="subcopy">{safe_msg(e)}</p>', unsafe_allow_html=True)
        return
    slot.empty()

    if r is None:
        st.markdown('<p class="headline">예측 범위 밖입니다</p>',
                    unsafe_allow_html=True)
        return

    # ★ 배경 비는 지도·글씨 **뒤**에만 깔린다 (z-index 0, block-container 가 1)
    st.markdown(rain_bg(r["lv"]), unsafe_allow_html=True)

    _map(lat, lon, name, r)

    L = LEVELS[r["lv"]]
    if r["lv"] == 4:
        # ★ 시안은 '호우 주의' 였는데 최상위 구간이 10mm/h 이상이라 호우가 아니다
        #   (기상청 기준 '강한 비'는 15~30, '매우 강한 비'가 30 이상).
        st.markdown('<span class="badge">우산 필수</span>', unsafe_allow_html=True)

    unit = f'<span class="unit">{L["unit"]}</span>' if L["unit"] else ""
    cells = "".join(f'<div class="cell{" on" if i <= r["lv"] else ""}"></div>'
                    for i in range(5))
    st.markdown(f"""
<div class="meta"><span>30분 뒤 · {r['valid_at']:%H:%M} KST</span><span>LV.{r['lv']}</span></div>
<div class="bignum">{L['range_text']}{unit}</div>
<p class="headline">{L['headline']}</p>
<p class="subcopy">{L['sub']}</p>
<div class="gauge">{cells}<span class="lab">{r['lv']}단계</span></div>
""", unsafe_allow_html=True)

    if not r["live"]:
        p = r["probs"]
        st.markdown(f"""<div class="stub">
[ 인증키 미설정 — 이 값은 실제 예측이 아닙니다 ]<br>
더미 P(&ge;0.1/1/3/10) = {p[0.1]:.3f} / {p[1.0]:.3f} / {p[3.0]:.3f} / {p[10.0]:.3f}
</div>""", unsafe_allow_html=True)


def _map(lat, lon, name, r):
    """시안의 '레이더/위성 영상' 패널 — 지도 + 예측 구름 + 강수 구간."""
    m = folium.Map(location=[lat, lon], zoom_start=8, tiles="CartoDB positron",
                   control_scale=False, zoom_control=True)

    if r["grid"] is not None:
        import overlay
        import backend
        # ★ backend 와 **같은 인스턴스**를 써야 한다. 여기서 따로 만들면
        #   ONNX 3개 + LightGBM 4개가 메모리에 두 벌 올라간다(무료 클라우드 1GB).
        P = backend._predictor()
        bounds = P.latlon_bounds()
        # ★ LCC -> 위경도 재투영이 반드시 필요하다. 그냥 붙이면 구름이 어긋난다.
        cloud = P.to_latlon(r["grid"]["pir"])
        cat = P.to_latlon(r["grid"]["cat"])
        folium.raster_layers.ImageOverlay(
            overlay.cloud_png(cloud), bounds=bounds, opacity=1.0, zindex=1).add_to(m)
        folium.raster_layers.ImageOverlay(
            overlay.rain_png(cat), bounds=bounds, opacity=1.0, zindex=2).add_to(m)

    folium.CircleMarker([lat, lon], radius=7, color="#12324d", weight=2,
                        fill=True, fill_color="#ffe58a", fill_opacity=1,
                        tooltip=name).add_to(m)
    st_folium(m, height=230, use_container_width=True, returned_objects=[],
              key=f"map_{lat:.3f}_{lon:.3f}_{r['valid_at']:%H%M}")

    if r["grid"] is not None:
        from levels import LEVEL_COLORS
        chips = "".join(
            f'<span><i style="background:{LEVEL_COLORS[i]}"></i>{LEVELS[i]["label"]}</span>'
            for i in (1, 2, 3, 4))
        st.markdown(
            '<div class="legend">'
            '<div class="lgnote">흰색 = 30분 뒤 예측 구름</div>'
            f'<div class="lgchips">{chips}</div></div>',
            unsafe_allow_html=True)


# ================================================================== 라우팅
if st.session_state.view == "result" and st.session_state.sel:
    result()
else:
    home()

with st.expander("환경 점검"):
    import importlib
    import platform
    rows = [f"python {platform.python_version()}"]
    for mod in ("streamlit", "numpy", "onnxruntime", "netCDF4", "lightgbm",
                "cv2", "pyproj", "requests", "folium", "streamlit_folium",
                "PIL", "branca"):
        try:
            rows.append(f"✅ {mod} "
                        f"{getattr(importlib.import_module(mod), '__version__', '?')}")
        except Exception as e:                          # noqa: BLE001
            rows.append(f"❌ {mod} — {type(e).__name__}: {e}")
    import fetch
    rows.append(("✅" if fetch.load_key() else "❌") + " KMA_API_KEY "
                + ("설정됨" if fetch.load_key() else "미설정 — 실제 예측 불가"))
    rows.append(("✅" if geocode.load_key() else "❌") + " KAKAO_REST_KEY "
                + ("설정됨" if geocode.load_key() else "미설정 — 지역 검색 불가"))
    rows.append(f"KST now {dt.datetime.now(KST):%Y-%m-%d %H:%M}")
    st.code("\n".join(rows), language=None)
