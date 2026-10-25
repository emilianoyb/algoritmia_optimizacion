"""Ejercicio 1: aceleración paralela de estimate_pi (1, 2, 4 y 8 procesos).
"""
import argparse
import json
import multiprocessing as mp
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS


def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


def medir():
    seeds = list(range(8))
    tiempos = {}
    for p in [1, 2, 4, 8]:
        mejor = float("inf")
        for _ in range(3):                       # mejor de 3 mediciones
            t0 = time.perf_counter()
            with mp.Pool(processes=p) as pool:
                pool.map(estimate_pi, seeds)
            mejor = min(mejor, time.perf_counter() - t0)
        tiempos[p] = mejor
        print(f"{p} procesos: {mejor:.3f} s")
    return {"cpus": mp.cpu_count(), "tiempos": {str(k): v for k, v in tiempos.items()}}


def graficar(datos, salida):
    P = [int(p) for p in datos["tiempos"]]
    T = [datos["tiempos"][str(p)] for p in P]
    sp = [T[0] / t for t in T]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(P, sp, "o-", label="Speedup medido")
    ax[0].plot(P, P, "--", color="gray", label="Speedup ideal (lineal)")
    ax[0].set(xlabel="Número de procesos", ylabel="Speedup", xticks=P,
              title="Aceleración (speedup) vs. procesos")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(P, T, "o-", label="Tiempo medido")
    ax[1].plot(P, [T[0] / p for p in P], "--", color="gray", label="Ideal (T1 / p)")
    ax[1].set(xlabel="Número de procesos", ylabel="Tiempo (s)", xticks=P,
              title="Tiempo vs. procesos")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(salida, dpi=130); plt.close(fig)
    for p, t, s in zip(P, T, sp):
        print(f"{p} procesos  t={t:.3f}s  speedup={s:.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usar-datos-guardados", action="store_true")
    a = ap.parse_args()
    if a.usar_datos_guardados:
        datos = json.load(open(RESULTS / "ej1_datos_medidos.json"))
    else:
        datos = medir()
        json.dump(datos, open(RESULTS / "ej1_medicion_local.json", "w"), indent=2)
    graficar(datos, RESULTS / "ej1_speedup.png")
