import sys
import glob
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import soundFunctions as s

args = [a for a in sys.argv[1:] if not a.startswith("--")]
dataPath = args[0] if args else "sound/data/"
plotsPath = "sound/plots/"
os.makedirs(plotsPath, exist_ok=True)
plt.rcParams["font.size"] = 13

L = 1.158
dL = 0.002
REF = "test6"
NBEST = 5
MIN_AMP = 50
MIN_GOOD = 5
MAX_SIGMA = 3
STARTS = [0, 25, 65, 135, 235]
LENGTHS = [345, 445, 645]
MAIN = (0, 445)
CO2_AIR = 0.04
DT_ROOM = 0.4
DRH = 2
T_BREATH = 25.0
I_CO2_REAL = 2 / (1.29 - 1)
RQ = 0.8
STUDENT95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}

t, rh = s.readConditions(dataPath + "conditions.txt")


def load(name):
    return np.loadtxt(dataPath + name + "_0.txt"), np.loadtxt(dataPath + name + "_1.txt")


def order(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    return (name.startswith("breath"), int(digits) if digits else 0)


names = sorted({os.path.basename(f)[:-len("_0.txt")] for f in glob.glob(dataPath + "*_0.txt")}, key=order)
REF = next((a[len("--ref="):] for a in sys.argv if a.startswith("--ref=")), REF)
if REF not in names:
    REF = names[0]

mic0, mic1 = load(REF)
Nref, iLag, iLead, refA, refB = s.referenceDelay(mic0, mic1)
cRef = L / (s.delayFromShift(Nref) / 1000)

records = {}
noise = []
for name in names:
    mic0, mic1 = load(name)
    amp = np.max(np.abs(mic0 - np.mean(mic0[:200])))
    if amp < MIN_AMP:
        noise.append((name, amp))
        continue
    w0 = s.leadWindowStart(mic0, mic1, Nref)
    N, sigma, good, rMean, rows = s.robustDelay(mic0, mic1, Nref, w0, STARTS, LENGTHS)
    Nmain, tauMain, rMain, shifts, err = s.findDelay(mic0, mic1, Nref, w0 + MAIN[0], w0 + MAIN[1])
    lead = mic1 if Nref > 0 else mic0
    records[name] = dict(N=N, tau=s.delayFromShift(N) * 1000 if good else np.nan, sigma=sigma, good=good, r=rMean,
                         fc=s.centroid(lead[w0:w0 + 1024]), valid=good >= MIN_GOOD and sigma <= MAX_SIGMA,
                         breath=name.startswith("breath"), w0=w0, shifts=shifts, err=err, rMain=rMain, rows=rows)


def select(breath):
    group = [n for n, d in records.items() if d["breath"] == breath and d["valid"]]
    return sorted(group, key=lambda n: -records[n]["r"])[:NBEST]


airNames = select(False)
breathNames = select(True)


def stats(group):
    taus = np.array([records[n]["tau"] for n in group])
    n = len(taus)
    tau = taus.mean()
    sd = taus.std(ddof=1)
    se = sd / np.sqrt(n)
    method = np.mean([records[n]["sigma"] for n in group]) * s.DT * 1000
    c = L / (tau * 1e-6)
    dcStat = c * se / tau
    dcMethod = c * method / tau
    dcL = c * dL / L
    return dict(n=n, taus=taus, tau=tau, sd=sd, se=se, t95=STUDENT95[n] * se, method=method, c=c,
                dcStat=dcStat, dcMethod=dcMethod, dcL=dcL, dc=np.sqrt(dcStat ** 2 + dcMethod ** 2 + dcL ** 2),
                dc95=c * STUDENT95[n] * se / tau)


air = stats(airNames)
breath = stats(breathNames)

pool = np.concatenate([air["taus"], breath["taus"]])
obs = breath["tau"] - air["tau"]
diffs = []
for idx in itertools.combinations(range(len(pool)), breath["n"]):
    mask = np.zeros(len(pool), bool)
    mask[list(idx)] = True
    diffs.append(pool[mask].mean() - pool[~mask].mean())
diffs = np.array(diffs)
pPerm = np.mean(diffs >= obs - 1e-9)
dTau = obs
dTauSe = np.hypot(air["se"], breath["se"])


def lines(temp, hum, iCO2=s.I_CO2):
    co2, room = s.speedOfSound(temp, s.h2oFraction(temp, hum), 8, iCO2)
    co2, wet = s.speedOfSound(temp, s.h2oFraction(temp, 100), 8, iCO2)
    return co2, room, wet, np.polyfit(room, co2, 1), np.polyfit(wet, co2, 1)


co2Grid, speedRoom, speedWet, fitRoom, fitWet = lines(t, rh)
cAirTheory = np.interp(CO2_AIR, co2Grid, speedRoom)


def co2Diff(tauAir, tauBr, temp=t, hum=rh, iCO2=s.I_CO2, tBr=None):
    grid, room, wet, fr, fw = lines(temp, hum, iCO2)
    cth = np.interp(CO2_AIR, grid, room)
    cStar = cth * tauAir / tauBr
    if tBr is not None:
        grid, room, wet, fr, fw = lines(tBr, hum, iCO2)
    return np.polyval(fw, cStar), cStar


co2AirA = np.polyval(fitRoom, air["c"])
co2BrA = np.polyval(fitWet, breath["c"])
co2B, cStarB = co2Diff(air["tau"], breath["tau"])
slopeWet = abs(fitWet[0])

bStat = slopeWet * cStarB * np.hypot(air["se"] / air["tau"], breath["se"] / breath["tau"])
nuB = (air["se"] ** 2 + breath["se"] ** 2) ** 2 / (air["se"] ** 4 / (air["n"] - 1) + breath["se"] ** 4 / (breath["n"] - 1))
tB = STUDENT95[int(nuB) + 1]
bLo, bHi = co2B - tB * bStat, co2B + tB * bStat
bMethod = slopeWet * cStarB * np.hypot(air["method"] / air["tau"], breath["method"] / breath["tau"])
bT = max(abs(co2Diff(air["tau"], breath["tau"], temp=t + DT_ROOM)[0] - co2B),
         abs(co2Diff(air["tau"], breath["tau"], temp=t - DT_ROOM)[0] - co2B))
bRH = max(abs(co2Diff(air["tau"], breath["tau"], hum=rh + DRH)[0] - co2B),
          abs(co2Diff(air["tau"], breath["tau"], hum=rh - DRH)[0] - co2B))
bGamma = abs(co2Diff(air["tau"], breath["tau"], iCO2=I_CO2_REAL)[0] - co2B)
bTotal = np.sqrt(bStat ** 2 + bMethod ** 2 + bT ** 2 + bRH ** 2 + bGamma ** 2)
bBreathT = co2Diff(air["tau"], breath["tau"], tBr=T_BREATH)[0] - co2B

dispNames = [n for n, d in records.items() if not d["breath"] and d["valid"]]
xAir = np.array([1 / np.sqrt(records[n]["fc"]) for n in dispNames])
yAir = np.array([records[n]["tau"] for n in dispNames])
xBr = np.array([1 / np.sqrt(records[n]["fc"]) for n in breathNames])
yBr = breath["taus"]


kDisp, tau0 = np.polyfit(xAir, yAir, 1)
nA = len(xAir)
sRes = np.sqrt(np.sum((yAir - kDisp * xAir - tau0) ** 2) / (nA - 2))
sxx = np.sum((xAir - xAir.mean()) ** 2)
kSe = sRes / np.sqrt(sxx)
tau0Se = sRes * np.sqrt(1 / nA + xAir.mean() ** 2 / sxx)
yBr0 = yBr - kDisp * xBr
tauBr0 = yBr0.mean()
tauBr0Se = np.sqrt(yBr0.var(ddof=1) / len(yBr) + (xBr.mean() * kSe) ** 2)
co2C = np.polyval(fitWet, L / (tauBr0 * 1e-6))
co2AirC = np.polyval(fitRoom, L / (tau0 * 1e-6))
cStat = slopeWet * (L / (tauBr0 * 1e-6)) * tauBr0Se / tauBr0
rDisp = np.corrcoef(xAir, yAir)[0, 1]
perm = np.array([np.corrcoef(xAir, pm)[0, 1] for pm in itertools.permutations(yAir)])
pDisp = np.mean(np.abs(perm) >= abs(rDisp) - 1e-12)
cInf = L / (tau0 * 1e-6)
cInfSe = cInf * tau0Se / tau0
co2AirCSe = abs(fitRoom[0]) * cInfSe

eta = lambda co2: (co2 / 100 - 0.0003) / 0.2095
etaB, etaBe = eta(co2B), bTotal / 100 / 0.2095
etaC, etaCe = eta(co2C), cStat / 100 / 0.2095
uptakeB = (co2B / 100 - 0.0003) / RQ / 0.2095

sens = None
if "breath3" in records and "breath3" not in breathNames:
    rows = records["breath3"]["rows"]
    ok = rows[(rows[:, 3] >= 0.8) & ~np.isnan(rows[:, 2])]
    if len(ok):
        tb3 = s.delayFromShift(np.median(ok[:, 2])) * 1000
        sens = (tb3, co2Diff(air["tau"], np.mean(np.append(breath["taus"], tb3)))[0])

out = []
add = out.append
add("Условия: t = %.1f ± %.1f C, rh = %.0f ± %d %%, L = %.3f ± %.3f м" % (t, DT_ROOM, rh, DRH, L, dL))
add("Опорная запись %s: пики %d и %d, N0 = %d, c = %.1f м/с" % (REF, iLead, iLag, Nref, cRef))
add("Шум вместо хлопка: " + ", ".join("%s (%d кодов)" % (n, a) for n, a in noise))
add("")
add("запись     N        sigmaN  окон  r      f_c, Гц  tau, мкс   годна")
for n, d in records.items():
    add("%-9s %8.2f %7.2f %4d  %5.3f %7.0f %9.1f   %s" % (n, d["N"], d["sigma"], d["good"], d["r"], d["fc"], d["tau"],
                                                   "да" if d["valid"] else "нет"))
add("")
for label, g, grp in [("Воздух", air, airNames), ("Выдох", breath, breathNames)]:
    add("%s (%d): %s" % (label, g["n"], ", ".join(grp)))
    add("  tau = %.1f мкс, sd = %.1f, se = %.1f, 95%% = ± %.1f, метод = %.1f мкс" % (g["tau"], g["sd"], g["se"], g["t95"], g["method"]))
    add("  c = %.2f ± %.2f м/с (случ. %.2f, метод %.2f, L %.2f); 95%% случ.: ± %.2f" %
        (g["c"], g["dc"], g["dcStat"], g["dcMethod"], g["dcL"], g["dc95"]))
add("Разность задержек выдох - воздух: %.1f ± %.1f мкс, перестановочный тест p = %.3f" % (dTau, dTauSe, pPerm))
add("")
add("Прямые: воздух %s, выдох %s; расчётная скорость в воздухе %.2f м/с" %
    (np.array2string(fitRoom, precision=4), np.array2string(fitWet, precision=4), cAirTheory))
add("A. По прямым без поправок: воздух %.1f %%, выдох %.1f %%" % (co2AirA, co2BrA))
add("B. Разностный метод: CO2 = %.2f %%, c* = %.2f м/с" % (co2B, cStarB))
add("   случайная %.2f, 95%% интервал (Стьюдент, nu = %.2f, t = %.2f) [%.2f; %.2f]; метода %.2f; t %.2f; rh %.2f; gamma(CO2) %.2f; итого ± %.2f %%" %
    (bStat, nuB, tB, bLo, bHi, bMethod, bT, bRH, bGamma, bTotal))
add("   если газ выдоха при %.0f C: %+.2f %%" % (T_BREATH, bBreathT))
add("C. Дисперсия: %d записей воздуха, r = %.3f, p = %.3f; tau0 = %.1f ± %.1f мкс, c(f->inf) = %.2f ± %.2f м/с" %
    (len(dispNames), rDisp, pDisp, tau0, tau0Se, cInf, cInfSe))
add("   CO2 воздуха после поправки %.2f ± %.2f %%, CO2 выдоха после поправки %.2f ± %.2f %%" % (co2AirC, co2AirCSe, co2C, cStat))
add("Доля усвоенного O2 (по CO2): B %.3f ± %.3f, C %.3f ± %.3f; с дыхательным коэффициентом %.1f: %.3f" %
    (etaB, etaBe, etaC, etaCe, RQ, uptakeB))
if sens:
    add("Чувствительность: breath3 по единственному годному окну tau = %.1f мкс; с ним CO2 (B) = %.2f %%" % sens)
print("\n".join(out))
with open(plotsPath + "results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

AIRC, BRC, INK = "#2A9D8F", "#E76F51", "#1B1F3B"

fig, ax = plt.subplots(figsize=(10, 4.5))
time0 = np.arange(len(refA)) * s.DT
ax.plot(time0, refA, color=INK, label="микрофон 0")
ax.plot(time0 + s.SKEW, refB, color=AIRC, label="микрофон 1")
ax.plot(time0[iLag], refA[iLag], "o", color=INK)
ax.plot(time0[iLead] + s.SKEW, refB[iLead], "o", color=AIRC)
ax.annotate("", xy=(time0[iLag], 1.08), xytext=(time0[iLead], 1.08), arrowprops=dict(arrowstyle="<->"))
ax.text((time0[iLag] + time0[iLead]) / 2, 1.13, "N₀ = %d отсчётов" % Nref, ha="center")
ax.set_xlabel("Время, мс")
ax.set_ylabel("Нормированный сигнал")
ax.set_xlim(0, 8)
ax.set_ylim(-1.2, 1.3)
ax.grid()
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(plotsPath + "reference.png", dpi=150)

d = records[REF]
mic0, mic1 = load(REF)
w0, w1 = d["w0"] + MAIN[0], d["w0"] + MAIN[1]
n = int(round(d["N"]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw=dict(width_ratios=[1, 1.4]))
ax1.plot(d["shifts"], d["err"] / (w1 - w0), ".-", color=INK)
ax1.axvline(d["N"], color=BRC, ls="--")
ax1.set_xlabel("Сдвиг N, отсчёты")
ax1.set_ylabel("S(N) / число точек")
ax1.set_title("Минимум при N = %.1f" % d["N"])
ax1.grid()
tt = np.arange(w0, w1) * s.DT
ax2.plot(tt, s.normalize(mic1[w0:w1]), color=AIRC, label="микрофон 1")
ax2.plot(tt, s.normalize(mic0[w0 + n:w1 + n]), color=INK, ls="--", label="микрофон 0, сдвинут на N")
ax2.set_xlabel("Время по микрофону 1, мс")
ax2.set_ylabel("Нормированный сигнал")
ax2.set_title("Участок с первым пиком, r = %.3f" % d["rMain"])
ax2.grid()
ax2.legend()
fig.tight_layout()
fig.savefig(plotsPath + "least-squares.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tt, mic1[w0:w1], color=AIRC, label="микрофон 1, среднее %.0f" % np.mean(mic1[w0:w1]))
ax.plot(tt, mic0[w0 + n:w1 + n], color=INK, ls="--", label="микрофон 0, сдвинут на N, среднее %.0f" % np.mean(mic0[w0 + n:w1 + n]))
ax.axhline(np.mean(mic1[w0:w1]), color=AIRC, lw=0.8)
ax.axhline(np.mean(mic0[w0 + n:w1 + n]), color=INK, lw=0.8)
ax.set_xlabel("Время по микрофону 1, мс")
ax.set_ylabel("Код АЦП")
ax.set_title("Тот же участок до нормировки")
ax.grid()
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(plotsPath + "offsets.png", dpi=150)

i = int(np.argmin(d["err"]))
S = d["err"] / (w1 - w0)
xs, ys = d["shifts"][i - 1:i + 2], S[i - 1:i + 2]
A = (ys[0] - 2 * ys[1] + ys[2]) / 2
B = (ys[2] - ys[0]) / 2
xf = np.linspace(-2.2, 2.2, 200)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(d["shifts"][i - 4:i + 5], S[i - 4:i + 5], "o", color="#9AA0B4", label="S(N) в целых точках")
ax.plot(xs, ys, "o", color=INK, ms=9, label="три точки у минимума")
ax.plot(xs[1] + xf, A * xf ** 2 + B * xf + ys[1], color=BRC, label="парабола через них")
dv = -B / (2 * A)
ax.plot(xs[1] + dv, ys[1] - B ** 2 / (4 * A), "*", color=BRC, ms=16, label="вершина: N = %.2f" % (xs[1] + dv))
for k, (x, y) in enumerate(zip(xs, ys)):
    ax.annotate("%.4f" % y, (x, y), textcoords="offset points", xytext=(-10 if k == 0 else 10, 10),
                ha="right" if k == 0 else "left", fontsize=10)
ax.set_xlabel("Сдвиг N, отсчёты")
ax.set_ylabel("S(N) / число точек")
ax.set_title("Минимум между отсчётами: δ = %.2f" % dv)
ax.set_ylim(min(ys) - 0.4 * (max(ys) - min(ys)), None)
ax.grid()
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(plotsPath + "parabola.png", dpi=150)

chosen = airNames + breathNames
cols = 5
rows_ = int(np.ceil(len(chosen) / cols))
fig, axs = plt.subplots(rows_, cols, figsize=(16, 3.2 * rows_), squeeze=False)
for ax, name in zip(axs.flat, chosen):
    d = records[name]
    mic0, mic1 = load(name)
    w0, w1 = d["w0"] + MAIN[0], d["w0"] + MAIN[1]
    n = int(round(d["N"]))
    tt = np.arange(w0, w1) * s.DT
    ax.plot(tt, s.normalize(mic1[w0:w1]), color=BRC if d["breath"] else AIRC, label="микрофон 1")
    ax.plot(tt, s.normalize(mic0[w0 + n:w1 + n]), ls="--", color=INK, label="микрофон 0, сдвинут")
    ax.set_title("%s: τ = %.1f мкс" % (name, d["tau"]), fontsize=12)
    ax.grid()
axs.flat[0].legend(fontsize=10)
for ax in axs.flat[len(chosen):]:
    ax.axis("off")
fig.supxlabel("Время, мс")
fig.tight_layout()
fig.savefig(plotsPath + "aligned.png", dpi=130)

fig, ax = plt.subplots(figsize=(10, 5.2))
pos, ticks, labels = 0, [], []
for g, group, color, title in [(air, airNames, AIRC, "Воздух"), (breath, breathNames, BRC, "Выдох")]:
    order = sorted(range(len(group)), key=lambda k: (len(group[k]), group[k]))
    names = [group[k] for k in order]
    taus = np.array(g["taus"])[order]
    xs = pos + np.arange(len(names))
    a0, a1 = xs[0] - 0.45, xs[-1] + 0.45
    ax.fill_between([a0, a1], g["tau"] - g["t95"], g["tau"] + g["t95"], color=color, alpha=0.13, lw=0)
    ax.hlines(g["tau"], a0, a1, color=color, ls="--", lw=1.8)
    ax.errorbar(xs, taus, yerr=[records[nm]["sigma"] * 4 for nm in names], fmt="o", color=color, ms=9, capsize=4)
    ax.text((xs[0] + xs[-1]) / 2, 0.97, "%s: %.1f ± %.1f мкс (95 %%)" % (title, g["tau"], g["t95"]), transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=color, fontsize=12, fontweight="bold")
    ticks += list(xs)
    labels += names
    pos += len(names) + 1
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_xlim(-0.8, pos - 1.2)
lo = min(air["tau"] - air["t95"], breath["tau"] - breath["t95"])
hi = max(air["tau"] + air["t95"], breath["tau"] + breath["t95"])
ax.set_ylim(lo - 8, hi + 10)
ax.set_xlabel("Запись (одна точка — один хлопок)")
ax.set_ylabel("Задержка τ, мкс")
ax.grid(axis="y")
ax.legend(handles=[
    plt.Line2D([], [], marker="o", ls="", color="#6C7A89", ms=8, label="задержка записи, усы — погрешность выбора участка"),
    plt.Line2D([], [], ls="--", color="#6C7A89", label="среднее по серии"),
    plt.matplotlib.patches.Patch(color="#6C7A89", alpha=0.2, label="95 % доверительный интервал среднего")],
    loc="lower right", fontsize=9, framealpha=0.95)
ax.set_title("Выдох медленнее на %.0f ± %.0f мкс (p = %.3f)" % (dTau, dTauSe, pPerm))
fig.tight_layout()
fig.savefig(plotsPath + "delays.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 5.2))
xs = np.linspace(0, max(xAir.max(), xBr.max()) * 1.05, 50)
ax.plot(xAir * 1000, yAir, "o", color=AIRC, ms=9, label="воздух, годные записи")
ax.plot(xBr * 1000, yBr, "o", color=BRC, ms=9, label="выдох")
ax.plot(xs * 1000, tau0 + kDisp * xs, color=AIRC, lw=1.5, label="τ = τ₀ + k/√f (воздух)")
ax.errorbar(0, tau0, yerr=tau0Se, fmt="*", color=INK, ms=14, capsize=5, label="τ₀ = %.0f ± %.0f мкс → c = %.1f ± %.1f м/с" % (tau0, tau0Se, cInf, cInfSe))
ax.set_xlabel("1/√f_c, 10⁻³ Гц^(−1/2)   (f_c — средняя частота сигнала)")
ax.set_ylabel("Задержка τ, мкс")
ax.set_title("Чем ниже частота хлопка, тем медленнее звук: r = %.2f, p = %.3f" % (rDisp, pDisp))
ax.grid()
ax.legend(fontsize=10, loc="upper left")
fig.tight_layout()
fig.savefig(plotsPath + "dispersion.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(co2Grid, speedRoom, color=AIRC, label="модель, влажность %d %%" % rh)
ax.plot(co2Grid, speedWet, color=BRC, label="модель, влажность 100 %")
ax.plot(co2AirA, air["c"], "o", mfc="none", mec=AIRC, ms=12, mew=2, label="воздух, измерено: %.1f м/с" % air["c"])
ax.plot(co2BrA, breath["c"], "o", mfc="none", mec=BRC, ms=12, mew=2, label="выдох, измерено: %.1f м/с" % breath["c"])
ax.errorbar(CO2_AIR, cAirTheory, fmt="*", color=AIRC, ms=16, label="воздух — калибровка: 0,04 %")
ax.errorbar(co2B, cStarB, xerr=bTotal, fmt="*", color=BRC, ms=16, capsize=5,
            label="выдох после калибровки: %.1f ± %.1f %%" % (co2B, bTotal))
ax.annotate("", xy=(co2B, cStarB), xytext=(co2BrA, breath["c"]), arrowprops=dict(arrowstyle="->", color="gray"))
ax.annotate("", xy=(CO2_AIR, cAirTheory), xytext=(co2AirA, air["c"]), arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("Доля CO₂ в сухом воздухе, %")
ax.set_ylabel("Скорость звука, м/с")
ax.set_xlim(-0.5, 8)
ax.grid()
ax.legend(fontsize=10)
ax.set_title("t = %.1f °C, влажность в аудитории %d %%" % (t, rh))
fig.tight_layout()
fig.savefig(plotsPath + "speed-of-sound.png", dpi=150)

cols = 4
rows_ = int(np.ceil(len(names) / cols))
fig, axs = plt.subplots(rows_, cols, figsize=(16, 2.4 * rows_), squeeze=False)
for ax, name in zip(axs.flat, names):
    mic0, mic1 = load(name)
    tt = np.arange(len(mic0)) * s.DT
    ax.plot(tt, mic0 - np.mean(mic0[:200]), lw=0.8, color=INK)
    ax.plot(tt, mic1 - np.mean(mic1[:15]), lw=0.8, color=AIRC)
    if name in chosen:
        status, color = "отобрана", "green"
    elif name in records:
        dd = records[name]
        status, color = "окон %d/15, σN = %.1f" % (dd["good"], dd["sigma"]) if dd["good"] else "форма не совпала", "gray"
    else:
        status, color = "шум, не хлопок", "red"
    ax.set_title("%s: %s" % (name, status), fontsize=11, color=color)
    ax.tick_params(labelsize=9)
    ax.grid(lw=0.4)
for ax in axs.flat[len(names):]:
    ax.axis("off")
fig.supxlabel("Время, мс")
fig.supylabel("Код АЦП минус начальный уровень")
fig.tight_layout()
fig.savefig(plotsPath + "records.png", dpi=110)

mic0, mic1 = load(REF)
fig, ax = plt.subplots(figsize=(10, 4))
tt = np.arange(len(mic0)) * s.DT
ax.plot(tt, mic0, color=INK, label="микрофон 0 (GP26)")
ax.plot(tt + s.SKEW, mic1, color=AIRC, label="микрофон 1 (GP27)")
ax.set_xlabel("Время, мс")
ax.set_ylabel("Код АЦП")
ax.set_title("Запись %s: сигнал микрофона 1 опережает микрофон 0 на ≈3,4 мс" % REF, fontsize=12)
ax.grid()
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(plotsPath + "raw.png", dpi=150)

if "--show" in sys.argv:
    plt.show()
