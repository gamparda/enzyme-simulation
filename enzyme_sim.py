"""
효소 반응 속도 시뮬레이션 (고2 「세포와 물질대사」 탐구)

효소 분자 하나하나를 컴퓨터 안에서 움직여 보는 '확률 모델'이다.
매 시간 단계(step)마다 각 효소는 세 상태 중 하나에 있다.
  0 = 비어 있음(자유 효소 E)
  1 = 기질과 결합(효소-기질 복합체 ES)
  2 = 저해제와 결합(EI)  ※ 실험 3에서만 사용

규칙
  - ES는 매 단계 확률 k로 반응을 끝내고 생성물 1개를 내놓은 뒤 다시 빈 효소가 된다.
  - 빈 효소는 기질이 진할수록 결합할 확률이 커진다: p = a[S] / (1 + a[S])
    (농도가 낮을 때는 거의 a×[S]에 비례하고, 아무리 진해도 1을 넘지 않는다)
  - (실험 3) 저해제도 같은 방식으로 빈 효소와 결합하고, EI는 확률 r로 저해제를 놓는다.
    기질과 저해제는 같은 빈 효소(활성 부위)를 두고 경쟁한다.

'초기 반응 속도'를 재기 위해 기질 농도는 실험 동안 일정하다고 가정한다
(기질이 효소보다 훨씬 많아 처음 잠깐 동안은 거의 줄지 않는다는 가정).

실행: python enzyme_sim.py
필요 패키지: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 글꼴(없으면 기본 글꼴 사용). Windows는 'Malgun Gothic', macOS는 'AppleGothic'
for name in ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(name in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(seed=2026)   # 같은 결과가 다시 나오도록 난수 씨앗 고정

# ---- 모델 상수 ----
A = 0.002      # 결합 계수: 기질 농도 1당 결합 확률
K = 0.2        # 촉매 확률: ES가 한 단계에 생성물을 만들 확률
B = 0.002      # 저해제 결합 계수
R = 0.2        # 저해제가 떨어질 확률
STEPS = 2000   # 측정할 시간 단계 수
WARMUP = 200   # 처음 200단계는 안정될 때까지 버림


def simulate(n_enzyme, S, I=0.0):
    """효소 n_enzyme개, 기질 농도 S, 저해제 농도 I에서 단계당 평균 생성물 수(반응 속도)를 반환."""
    state = np.zeros(n_enzyme, dtype=int)
    products = 0
    x, y = A * S, B * I                 # 기질·저해제와 만나는 '기회'의 크기
    p_bind_s = x / (1 + x + y)          # 기질과 결합할 확률 (아무리 진해도 1을 넘지 않음)
    p_bind_i = y / (1 + x + y)          # 저해제와 결합할 확률
    for step in range(WARMUP + STEPS):
        # 1단계: 결합해 있던 효소가 떨어진다
        u = rng.random(n_enzyme)        # 효소마다 주사위 하나씩
        done = (state == 1) & (u < K)   # ES → E + 생성물
        release = (state == 2) & (u < R)  # EI → E + I
        state[done | release] = 0
        if step >= WARMUP:
            products += done.sum()

        # 2단계: 빈 효소가 기질 또는 저해제와 결합한다 (방금 빈 효소도 포함)
        u = rng.random(n_enzyme)
        free = state == 0
        state[free & (u < p_bind_s)] = 1
        state[free & (u >= p_bind_s) & (u < p_bind_s + p_bind_i)] = 2
    return products / STEPS


def theory(n_enzyme, S, I=0.0):
    """미카엘리스-멘텐 식: v = Vmax·[S] / (Km(1+[I]/Ki) + [S])"""
    vmax = n_enzyme * K
    km = K / A
    ki = R / B
    return vmax * S / (km * (1 + I / ki) + S)


S_LIST = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
S_FINE = np.linspace(0, 1000, 300)

# ---------- 실험 1: 기질 농도 → 반응 속도 ----------
v1 = [simulate(100, s) for s in S_LIST]

# ---------- 실험 2: 효소 양 2배 ----------
v2 = [simulate(200, s) for s in S_LIST]

# ---------- 실험 3: 경쟁적 저해제 ----------
v3 = [simulate(100, s, I=100) for s in S_LIST]

# ---------- 결과 표 출력 ----------
print(f"이론값: Vmax(효소100개) = {100*K:.0f}, Km = {K/A:.0f}, Ki = {R/B:.0f}")
print(f"{'[S]':>5} | {'효소100':>8} {'이론':>6} | {'효소200':>8} {'이론':>6} | {'저해제':>8} {'이론':>6}")
for s, a, b, c in zip(S_LIST, v1, v2, v3):
    print(f"{s:>5} | {a:8.2f} {theory(100,s):6.2f} | {b:8.2f} {theory(200,s):6.2f} | {c:8.2f} {theory(100,s,100):6.2f}")

err = [abs(v - theory(n, s, i)) / theory(n, s, i) * 100
       for vs, n, i in [(v1, 100, 0), (v2, 200, 0), (v3, 100, 100)]
       for s, v in zip(S_LIST, vs) if s > 0]
print(f"\n이론값과의 평균 오차: {np.mean(err):.1f}%, 최대 오차: {np.max(err):.1f}%")

# ---------- 그래프 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
blue, orange, red = "#2a6fdb", "#e8833a", "#c23b4a"

ax = axes[0]
ax.plot(S_FINE, theory(100, S_FINE), color=blue, alpha=0.5, label="이론 곡선 (미카엘리스-멘텐)")
ax.scatter(S_LIST, v1, color=blue, zorder=3, label="시뮬레이션 (효소 100개)")
ax.axhline(100 * K, ls="--", color="gray", lw=1)
ax.text(5, 100 * K + 0.4, "최대 속도 Vmax = 20", color="gray")
ax.axvline(K / A, ls=":", color="gray", lw=1)
ax.text(K / A + 15, 16.5, "Km = 100\n(속도가 Vmax의 절반)", color="gray")
ax.set_title("실험 1. 기질 농도와 반응 속도")

ax = axes[1]
ax.plot(S_FINE, theory(100, S_FINE), color=blue, alpha=0.5)
ax.scatter(S_LIST, v1, color=blue, zorder=3, label="효소 100개")
ax.plot(S_FINE, theory(200, S_FINE), color=orange, alpha=0.5)
ax.scatter(S_LIST, v2, color=orange, zorder=3, label="효소 200개")
ax.set_title("실험 2. 효소 양을 2배로")

ax = axes[2]
ax.plot(S_FINE, theory(100, S_FINE), color=blue, alpha=0.5)
ax.scatter(S_LIST, v1, color=blue, zorder=3, label="저해제 없음")
ax.plot(S_FINE, theory(100, S_FINE, 100), color=red, alpha=0.5)
ax.scatter(S_LIST, v3, color=red, zorder=3, label="경쟁적 저해제 있음")
ax.set_title("실험 3. 경쟁적 저해제")

for ax in axes:
    ax.set_xlabel("기질 농도 [S]")
    ax.set_ylabel("반응 속도 (단계당 생성물 수)")
    ax.set_ylim(0, 42 if ax is axes[1] else 22)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig("enzyme_results.png", dpi=180)
print("그래프 저장: enzyme_results.png")
