"""Ejercicio 4: recocido simulado (2-opt) con distintos esquemas de enfriamiento."""
import json
import random
import statistics as st
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import (RESULTS, dist_matrix, make_schedule, random_points,
                    simulated_annealing, tour_length, two_opt)

N_CIUDADES, SEMILLAS = 50, 10
PRESUPUESTOS = [10_000, 40_000, 160_000]
ESQUEMAS = ["lineal", "exponencial", "logarítmico", "exponencial rápido"]
T0, TF = 50.0, 0.5
COLORES = dict(zip(ESQUEMAS, ["tab:green", "tab:blue", "tab:purple", "tab:red"]))

if __name__ == "__main__":
    D = dist_matrix(random_points(N_CIUDADES, random.Random(11)))
    res, curvas = {}, {}
    for N in PRESUPUESTOS:
        for esq in ESQUEMAS:
            costos, tiempos = [], []
            for s in range(SEMILLAS):
                rng = random.Random(1000 + s)
                ini = list(range(N_CIUDADES)); rng.shuffle(ini)         # inicio aleatorio
                t0 = time.perf_counter()
                out = simulated_annealing(D, ini, make_schedule(esq, T0, TF, N), N, rng, paso=max(N // 400, 1))
                tiempos.append(time.perf_counter() - t0)
                costos.append(out[0])
                if N == 40_000:
                    curvas.setdefault(esq, []).append(out)
            res[(N, esq)] = (costos, tiempos)
            print(f"N={N:>7,}  {esq:20s} costo {st.mean(costos):7.1f} ± {st.pstdev(costos):5.1f}   "
                  f"tiempo {st.mean(tiempos):6.3f} s")

    # referencia: subir colinas 2-opt desde inicio aleatorio
    hc_c, hc_t = [], []
    for s in range(SEMILLAS):
        rng = random.Random(1000 + s)
        ini = list(range(N_CIUDADES)); rng.shuffle(ini)
        t0 = time.perf_counter(); t, _ = two_opt(D, ini); hc_t.append(time.perf_counter() - t0)
        hc_c.append(tour_length(D, t))
    print(f"2-opt puro (referencia)       costo {st.mean(hc_c):7.1f} ± {st.pstdev(hc_c):5.1f}   tiempo {st.mean(hc_t):6.3f} s")

    json.dump({"resultados": {f"{N}|{e}": dict(costo_media=st.mean(c), costo_desv=st.pstdev(c),
                                              tiempo_media=st.mean(t)) for (N, e), (c, t) in res.items()},
               "hill_climbing": dict(costo_media=st.mean(hc_c), costo_desv=st.pstdev(hc_c), tiempo_media=st.mean(hc_t))},
              open(RESULTS / "ej4_resumen.json", "w"), indent=2, ensure_ascii=False)

    # ---- Figura 1: curvas de convergencia (N = 40,000)
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for esq in ESQUEMAS:
        runs = curvas[esq]
        its = runs[0][2]
        medio = [st.mean(r[4][i] for r in runs) for i in range(len(its))]
        ax[0].plot(its, medio, color=COLORES[esq], label=esq)
        r0 = runs[0]
        ax[1].plot(r0[5], r0[3], color=COLORES[esq], lw=.7, label=esq)
        ax[2].plot(its, r0[5], color=COLORES[esq], label=esq)
    ax[0].set(xlabel="Iteración", ylabel="Mejor costo (media de 10 corridas)", title="Costo vs. iteraciones")
    ax[1].set(xlabel="Temperatura (log, el enfriamiento va hacia la derecha)", ylabel="Costo actual",
              title="Costo vs. temperatura (1 corrida)", xscale="log")
    ax[1].invert_xaxis()
    ax[2].set(xlabel="Iteración", ylabel="Temperatura", title="Esquemas de enfriamiento", yscale="log")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "ej4_convergencia.png", dpi=130); plt.close(fig)

    # ---- Figura 2: tiempo vs calidad
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    marcas = dict(zip(PRESUPUESTOS, ["o", "s", "^"]))
    for esq in ESQUEMAS:
        for N in PRESUPUESTOS:
            c, t = res[(N, esq)]
            ax.errorbar(st.mean(t), st.mean(c), yerr=st.pstdev(c), fmt=marcas[N], color=COLORES[esq], capsize=3, ms=7)
    ax.errorbar(st.mean(hc_t), st.mean(hc_c), yerr=st.pstdev(hc_c), fmt="*", color="k", ms=12, capsize=3)
    h = [Line2D([], [], color=COLORES[e], marker="o", ls="", label=e) for e in ESQUEMAS]
    h += [Line2D([], [], color="gray", marker=marcas[N], ls="", label=f"{N:,} iteraciones") for N in PRESUPUESTOS]
    h += [Line2D([], [], color="k", marker="*", ls="", label="2-opt puro")]
    ax.legend(handles=h, fontsize=7)
    ax.set(xlabel="Tiempo de ejecución (s)", ylabel="Costo final (media ± desv.)",
           title="Tiempo de cómputo vs. calidad de la solución", xscale="log")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "ej4_tiempo_vs_calidad.png", dpi=130); plt.close(fig)
