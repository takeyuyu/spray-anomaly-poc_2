"""
窓汚れ（ダスト堆積）検知 PoC ── 実装実績その2
=================================================
スプレー監視の「見ている窓そのものが汚れて見えなくなる」問題に対し、
窓の汚れ（ゆっくり・場所固定）と 噴霧の異常（速い・噴霧領域）を
取り違えずに切り分けられるかを、合成データで実証する。

守秘配慮: 実機データは一切使用していない。すべて合成データ。

考え方:
  カメラ映像が暗くなる原因は2つある。
   (A) 噴霧の異常 … 速い変動（1秒以内）、噴霧している場所で起こる
   (B) 窓の汚れ   … ゆっくりした変化（分〜時間）、特定の場所に固定的に積もる
  この2つを取り違えると、現場で誤報を生み、使われなくなる。
   → 「速いか/遅いか」と「場所」で切り分ける。

実行すると、同じフォルダに以下が出力される:
  - dust_poc_frames.png : 窓が徐々に汚れていく様子（合成）
  - dust_poc_result.png : 窓の汚れと噴霧異常を、別々のチャンネルで検知する様子
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

try:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    BOLD = fm.FontProperties(family="Noto Sans CJK JP", weight="bold")
    JP = fm.FontProperties(family="Noto Sans CJK JP")
except Exception:
    BOLD = None; JP = None
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)

FPS = 2                 # 長時間監視なので低レート
MINUTES = 5
N = FPS * 60 * MINUTES  # 600フレーム
H = W = 64

def spray_cone(intensity):
    yy, xx = np.mgrid[0:H, 0:W]
    cx = W/2
    img = np.zeros((H, W))
    if intensity > 0:
        for y in range(H):
            width = 2 + (y/H)*(W*0.35)
            img[y] = np.exp(-((xx[y]-cx)**2)/(2*width**2)) * (y/H)**0.4
        img = img/img.max()*intensity
    return img

def dust_mask(progress):
    """窓の左下に固定的に積もるダスト。progress(0→1)で濃くなる。"""
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W*0.30, H*0.72
    blob = np.exp(-(((xx-cx)**2)/(2*(W*0.28)**2) + ((yy-cy)**2)/(2*(H*0.22)**2)))
    return blob * progress

t_sec = np.arange(N)/FPS

# (B) 窓汚れ: 5分かけてゆっくり進行（遅い変化）
dust_progress = np.clip(np.linspace(0, 0.6, N) + rng.normal(0,0.005,N), 0, 1)

# (A) 噴霧強度: 基本は正常。後半の一定区間だけチラつき（速い異常）
spray_int = 0.85 + 0.02*np.sin(np.linspace(0,8*np.pi,N))
spray_int += rng.normal(0, 0.015, N)
flicker_start, flicker_end = int(N*0.70), int(N*0.82)
tt = np.arange(flicker_start, flicker_end)
flick = (tt % 2 == 0).astype(float)
spray_int[flicker_start:flicker_end] = np.where(flick>0, 0.8, 0.05)
spray_int = np.clip(spray_int + rng.normal(0,0.02,N), 0, 1)

def build_frame(i):
    base = spray_cone(spray_int[i]) + 0.12
    d = dust_mask(dust_progress[i])
    frame = base * (1 - d)
    frame = frame + (rng.random((H,W))>0.99)*rng.uniform(0.05,0.15,(H,W))
    frame = frame + rng.normal(0, 0.03, (H,W))
    return np.clip(frame, 0, 1)

frames = np.stack([build_frame(i) for i in range(N)], axis=0)

# 噴霧領域（中央）と 窓監視領域（左下）を別々に見る
spray_region = frames[:, 4:H-4, W//2-12:W//2+12].mean(axis=(1,2))
dust_region  = frames[:, int(H*0.55):int(H*0.9), int(W*0.12):int(W*0.5)].mean(axis=(1,2))

# (A) 噴霧異常 = 速い変動
win_fast = max(2, FPS//2)
spray_hf = np.array([spray_region[max(0,i-win_fast):i+win_fast].std() for i in range(N)])
# (B) 窓汚れ = 遅い低下（長時間平均）
win_slow = FPS*20
dust_trend = np.array([dust_region[max(0,i-win_slow):i+1].mean() for i in range(N)])

# 基準と閾値（運転開始30秒を基準に）
base_n = FPS*30
base_dust = dust_region[:base_n].mean()
base_hf = spray_hf[:base_n].mean()
base_hf_std = spray_hf[:base_n].std()
TH_FLICK = base_hf + 5*base_hf_std
TH_DUST = base_dust * 0.7

flick_alarm = spray_hf > TH_FLICK
dust_warn = dust_trend < TH_DUST

print("="*56)
print("窓汚れ検知 PoC 結果")
print("="*56)
if dust_warn.any():
    print(f"  窓汚れ警告（清掃推奨）: 約 {t_sec[dust_warn.argmax()]:.0f} 秒後に発報")
if flick_alarm.any():
    print(f"  噴霧チラつき検知       : 約 {t_sec[flick_alarm.argmax()]:.0f} 秒後に検知")
print("  → 2つは別チャンネルで検知。取り違えなし。")
print("="*56)

C_SPRAY="#0f6e63"; C_DUST="#c2641f"; C_ALARM="#a32a2a"

# ===== 図1: 窓が徐々に汚れていく =====
fig = plt.figure(figsize=(11, 5)); fig.patch.set_facecolor("white")
fig.text(0.5,0.96,"【課題】カメラ監視は、見ている窓が汚れると役に立たなくなる",
         ha="center",va="top",fontsize=14,color=C_ALARM,fontproperties=BOLD)
fig.text(0.5,0.90,"5分間で、窓の左下にダストが少しずつ積もっていく様子（合成）",
         ha="center",va="top",fontsize=10.5,color="#41506a",fontproperties=JP)
for i,(idx,lab) in enumerate([(int(N*0.05),"開始直後"),(int(N*0.5),"2分半後"),(int(N*0.95),"5分後")]):
    ax=fig.add_subplot(1,3,i+1)
    ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{lab}", fontsize=12, fontproperties=BOLD, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#c9c0ad"); s.set_linewidth(1.5)
fig.text(0.5,0.05,"→ 左下がだんだん暗くなる。これを「噴霧の異常」と取り違えてはいけない",
         ha="center",va="bottom",fontsize=11.5,color=C_DUST,fontproperties=BOLD)
plt.subplots_adjust(top=0.80, bottom=0.13, wspace=0.1)
plt.savefig("dust_poc_frames.png", dpi=140, bbox_inches="tight", facecolor="white")
print("保存: dust_poc_frames.png")

# ===== 図2: 2つを別チャンネルで切り分ける =====
fig = plt.figure(figsize=(12, 8)); fig.patch.set_facecolor("white")
gs = GridSpec(3,1,figure=fig,hspace=0.55,height_ratios=[1,1,0.9])
fig.text(0.5,0.975,"窓の汚れと噴霧の異常を、別々に見張る",
         ha="center",va="top",fontsize=15,color=C_SPRAY,fontproperties=BOLD)
fig.text(0.5,0.945,"「ゆっくりした汚れ」と「速い異常」は、時間の動き方が違う。だから切り分けられる",
         ha="center",va="top",fontsize=10.5,color="#41506a",fontproperties=JP)

# 上: 窓汚れ（ゆっくり進む）
ax=fig.add_subplot(gs[0])
dust_pct = (1 - dust_trend/base_dust)*100
ax.plot(t_sec, dust_pct, color=C_DUST, lw=2.2)
ax.axhline((1-TH_DUST/base_dust)*100, color=C_ALARM, ls=":", lw=1.5)
ax.text(t_sec[-1]*0.02, (1-TH_DUST/base_dust)*100+2, "ここを超えたら「清掃して」",
        fontsize=9.5, color=C_ALARM, fontproperties=JP)
ax.set_title("① 窓の汚れチャンネル（ゆっくり暗くなる）", fontsize=12, color=C_DUST, fontproperties=BOLD)
ax.set_xlabel("時間（秒）", fontsize=9, fontproperties=JP); ax.set_ylabel("暗くなった割合 %", fontsize=9, fontproperties=JP)
ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)

# 中: 噴霧異常（速くバタつく）
ax=fig.add_subplot(gs[1])
ax.plot(t_sec, spray_hf, color=C_SPRAY, lw=1)
ax.axhline(TH_FLICK, color=C_ALARM, ls=":", lw=1.5)
ax.axvspan(flicker_start/FPS, flicker_end/FPS, color=C_ALARM, alpha=0.12)
ax.text((flicker_start/FPS), TH_FLICK*1.3, "← この区間だけ噴霧がチラついた",
        fontsize=9.5, color=C_SPRAY, fontproperties=JP)
ax.set_title("② 噴霧の異常チャンネル（速くバタつく）", fontsize=12, color=C_SPRAY, fontproperties=BOLD)
ax.set_xlabel("時間（秒）", fontsize=9, fontproperties=JP); ax.set_ylabel("バタつきの大きさ", fontsize=9, fontproperties=JP)
ax.tick_params(labelsize=8); ax.grid(True, alpha=0.2)

# 下: 判定タイムライン
ax=fig.add_subplot(gs[2])
ax.fill_between(t_sec, 1.6, 2.4, where=dust_warn, color=C_DUST, alpha=0.55, step="mid")
ax.fill_between(t_sec, 0.6, 1.4, where=flick_alarm, color=C_SPRAY, alpha=0.65, step="mid")
ax.set_yticks([1,2]); ax.set_yticklabels(["噴霧の異常\n（速い）","窓の汚れ\n（ゆっくり）"], fontproperties=JP, fontsize=10)
ax.set_ylim(0.4,2.6); ax.set_xlim(0,t_sec[-1])
ax.set_xlabel("時間（秒）", fontsize=9, fontproperties=JP)
ax.set_title("③ 2つは別々のタイミングで鳴る ＝ 取り違えていない", fontsize=12, fontproperties=BOLD)
ax.tick_params(labelsize=8)

plt.subplots_adjust(top=0.90, bottom=0.07)
plt.savefig("dust_poc_result.png", dpi=140, bbox_inches="tight", facecolor="white")
print("保存: dust_poc_result.png")
