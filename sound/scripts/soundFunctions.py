import numpy as np

R = 8.314
PA = 101325

TEMPS = [20, 22, 24, 26, 28, 30, 32, 34, 36]
PW0 = [2336.8, 2642.4, 2982.4, 3360.0, 3778.5, 4241.7, 4759.6, 5325.1, 5947.9]

M_N2, M_O2, M_AR, M_CO2, M_H2O = 0.02801, 0.03200, 0.03995, 0.04401, 0.01802
I_N2, I_O2, I_AR, I_CO2, I_H2O = 5, 5, 3, 6, 6

X_N2, X_O2, X_AR = 0.7809, 0.2098, 0.0093


def h2oFraction(temperature, rh):
    pw0 = np.interp(temperature, TEMPS, PW0)
    return rh / 100 * pw0 / PA


def speedOfSound(temperature, h2oX, co2Max, iCO2=I_CO2):
    T = temperature + 273.15
    co2 = np.linspace(0, co2Max / 100, 51)

    k = 1 - h2oX
    xN2 = X_N2 * k
    xO2 = (X_O2 - co2) * k
    xAr = X_AR * k
    xCO2 = co2 * k

    M = M_N2 * xN2 + M_O2 * xO2 + M_AR * xAr + M_CO2 * xCO2 + M_H2O * h2oX
    Cv = (I_N2 * xN2 + I_O2 * xO2 + I_AR * xAr + iCO2 * xCO2 + I_H2O * h2oX) / 2 * R
    Cp = Cv + R

    soundSpeed = np.sqrt(Cp / Cv * R * T / M)
    return co2 * 100, soundSpeed


DT = 0.004
SKEW = 0.002


def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)


def mic1Leads(mic0, mic1, count=200):
    return np.std(mic1[:count]) > 3 * np.std(mic0[:count])


def delayFromShift(N):
    return abs(N * DT - SKEW)


def referenceDelay(mic0, mic1, nMin=640, nMax=1160, search=600, edge=15):
    a = mic0 - np.mean(mic0[:200])
    a = a / np.max(np.abs(a))
    if mic1Leads(mic0, mic1):
        b = mic1 - np.mean(mic1[:edge])
        b = b / np.max(np.abs(b[:nMax]))
        iLead = edge + int(np.argmax(np.abs(b[edge:search])))
        sign = np.sign(b[iLead])
        iLag = iLead + nMin + int(np.argmax(sign * a[iLead + nMin:iLead + nMax]))
        return iLag - iLead, iLag, iLead, a, b
    b = mic1 - np.mean(mic1[:200])
    b = b / np.max(np.abs(b))
    onset = int(np.argmax(np.abs(a) > 5 * np.std(a[:200])))
    iLead = onset + int(np.argmax(np.abs(a[onset:onset + search])))
    sign = np.sign(a[iLead])
    iLag = iLead + nMin + int(np.argmax(sign * b[iLead + nMin:iLead + nMax]))
    return -(iLag - iLead), iLead, iLag, a, b


def findDelay(mic0, mic1, Nref, w0, w1, half=30):
    if Nref >= 0:
        lag, lead, n0 = mic0, mic1, Nref
    else:
        lag, lead, n0 = mic1, mic0, -Nref
    B = normalize(lead[w0:w1])
    shifts = np.arange(n0 - half, n0 + half + 1)
    err = np.array([np.sum((normalize(lag[w0 + n:w1 + n]) - B) ** 2) for n in shifts])
    i = int(np.argmin(err))
    n = float(shifts[i])
    if 0 < i < len(err) - 1:
        n += (err[i - 1] - err[i + 1]) / (2 * (err[i - 1] - 2 * err[i] + err[i + 1]))
    r = 1 - err[i] / (2 * (w1 - w0))
    N = n if Nref >= 0 else -n
    return N, delayFromShift(N), r, shifts, err


def robustDelay(mic0, mic1, Nref, w0, starts, lengths, minR=0.8, half=30):
    rows = []
    for a in starts:
        for b in lengths:
            N, tau, r, shifts, err = findDelay(mic0, mic1, Nref, w0 + a, w0 + a + b, half)
            rows.append((a, b, N, r))
    rows = np.array(rows, dtype=float)
    good = rows[(rows[:, 3] >= minR) & ~np.isnan(rows[:, 2])]
    if len(good) == 0:
        return np.nan, np.nan, 0, np.nan, rows
    N = np.median(good[:, 2])
    sigma = 1.4826 * np.median(np.abs(good[:, 2] - N))
    return N, sigma, len(good), np.mean(good[:, 3]), rows


def centroid(x, fmin=100, fmax=20000):
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), DT / 1000)
    m = (f > fmin) & (f < fmax)
    return np.sum(f[m] * spec[m]) / np.sum(spec[m])


def leadWindowStart(mic0, mic1, Nref, pre=50, edge=15):
    if Nref > 0:
        return edge
    a = mic0 - np.mean(mic0[:200])
    onset = int(np.argmax(np.abs(a) > 5 * np.std(a[:200])))
    return max(onset - pre, edge)


def readConditions(path):
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=")
                values[key.strip()] = float(value.split()[0])
    return values["Temperature"], values["Relative humidity"]
