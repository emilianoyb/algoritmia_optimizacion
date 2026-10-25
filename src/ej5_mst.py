"""Ejercicio 5: razón de aproximación del MST-TSP (100 instancias de 9 ciudades)
y comparación con MST+2-opt, búsqueda local (2-opt) y recocido simulado."""
import json
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (RESULTS, dist_matrix, held_karp, make_schedule, mst_prim,
                    mst_weight, nearest_neighbour, preorder, random_points,
                    simulated_annealing, tour_length, two_opt)

if __name__ == "__main__":
    rng = random.Random(2026)
    M = ["MST puro", "MST + 2-opt", "2-opt (inicio aleatorio)", "2-opt (inicio NN)", "Recocido simulado"]
    costo = {m: [] for m in M}
    opt, wmst = [], []
    for _ in range(100):
        D = dist_matrix(random_points(9, rng, size=10.0))
        n = 9
        opt.append(held_karp(D))
        ar = mst_prim(D); wmst.append(mst_weight(D, ar))
        tour = preorder(ar)
        costo["MST puro"].append(tour_length(D, tour))
        costo["MST + 2-opt"].append(tour_length(D, two_opt(D, tour)[0]))
        ale = list(range(n)); rng.shuffle(ale)
        costo["2-opt (inicio aleatorio)"].append(tour_length(D, two_opt(D, ale)[0]))
        costo["2-opt (inicio NN)"].append(tour_length(D, two_opt(D, nearest_neighbour(D, rng.randrange(n)))[0]))
        ale = list(range(n)); rng.shuffle(ale) 
        costo["Recocido simulado"].append(
            simulated_annealing(D, ale, make_schedule("exponencial", 5.0, 0.01, 5000), 5000, rng, paso=5000)[0])

    razon = {m: [c / o for c, o in zip(costo[m], opt)] for m in M}
    resumen = {"n_instancias": 100}
    print(f"{'Método':26s} {'media':>6s} {'mediana':>8s} {'mín':>6s} {'máx':>6s}  óptimas")
    for m in M:
        r = razon[m]
        optimas = sum(x <= 1 + 1e-9 for x in r)
        resumen[m] = dict(media=st.mean(r), mediana=st.median(r), minima=min(r), maxima=max(r), optimas=optimas,
                          costo_medio=st.mean(costo[m]))
        print(f"{m:26s} {st.mean(r):6.3f} {st.median(r):8.3f} {min(r):6.3f} {max(r):6.3f}  {optimas}/100")
    resumen["costo_optimo_medio"] = st.mean(opt)
    resumen["w_mst_sobre_optimo_media"] = st.mean(w / o for w, o in zip(wmst, opt))
    print(f"Costo óptimo medio {st.mean(opt):.2f}; w(MST)/óptimo medio {resumen['w_mst_sobre_optimo_media']:.3f}")
    json.dump(resumen, open(RESULTS / "ej5_resumen.json", "w"), indent=2, ensure_ascii=False)

    # ---- Figura 1: razón de MST vs límite teórico
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(range(1, 101), razon["MST puro"], s=14, label="MST puro")
    ax[0].scatter(range(1, 101), razon["MST + 2-opt"], s=14, label="MST + 2-opt")
    ax[0].axhline(2, color="r", ls="--", label="Límite teórico ρ = 2")
    ax[0].axhline(1, color="g", ls=":", label="Óptimo")
    ax[0].set(xlabel="Instancia", ylabel="C obtenido / C óptimo", ylim=(0.95, 2.1), title="Razón de aproximación por instancia")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    bins = [1 + 0.05 * i for i in range(0, 22)]
    ax[1].hist(razon["MST puro"], bins=bins, alpha=.7, label="MST puro")
    ax[1].hist(razon["MST + 2-opt"], bins=bins, alpha=.7, label="MST + 2-opt")
    ax[1].axvline(2, color="r", ls="--", label="Límite teórico ρ = 2")
    ax[1].set(xlabel="C obtenido / C óptimo", ylabel="Frecuencia", xlim=(0.98, 2.1), title="Histograma de la razón")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "ej5_razon_mst.png", dpi=130); plt.close(fig)

    # ---- Figura 2: comparativa de costos
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    etq = ["MST\npuro", "MST\n+2-opt", "2-opt\naleatorio", "2-opt\nNN", "Recocido\nsimulado"]
    ax[0].boxplot([costo[m] for m in M] + [opt], tick_labels=etq + ["Óptimo"])
    ax[0].set(ylabel="Costo del tour", title="Costo obtenido por método")
    ax[1].boxplot([razon[m] for m in M], tick_labels=etq)
    ax[1].axhline(1, color="g", ls=":")
    ax[1].set(ylabel="C obtenido / C óptimo", title="Razón respecto al óptimo")
    for a in ax:
        a.grid(alpha=.3); a.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "ej5_comparativa.png", dpi=130); plt.close(fig)
