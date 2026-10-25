"""Ejercicio 3: 2-opt desde tour aleatorio vs. Vecino Más Cercano (nodo inicial aleatorio)."""
import json
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (RESULTS, dist_matrix, nearest_neighbour, random_points,
                    tour_length, two_opt)

N_CIUDADES, N_INSTANCIAS = 50, 100

if __name__ == "__main__":
    rng = random.Random(3)
    res = {k: {"mejoras": [], "inicial": [], "final": []} for k in ("aleatorio", "NN")}
    for _ in range(N_INSTANCIAS):
        D = dist_matrix(random_points(N_CIUDADES, rng))
        aleatorio = list(range(N_CIUDADES)); rng.shuffle(aleatorio)
        nn = nearest_neighbour(D, start=rng.randrange(N_CIUDADES))
        for k, ini in (("aleatorio", aleatorio), ("NN", nn)):
            t, m = two_opt(D, ini)
            res[k]["mejoras"].append(m)
            res[k]["inicial"].append(tour_length(D, ini))
            res[k]["final"].append(tour_length(D, t))

    resumen = {}
    for k, r in res.items():
        resumen[k] = {c: dict(media=st.mean(r[c]), desv=st.pstdev(r[c]), mediana=st.median(r[c]))
                      for c in r}
        print(f"{k:10s} mejoras {st.mean(r['mejoras']):7.2f} ± {st.pstdev(r['mejoras']):5.2f} | "
              f"L inicial {st.mean(r['inicial']):8.1f} | L final {st.mean(r['final']):7.1f} ± {st.pstdev(r['final']):5.1f}")
    menos = sum(a > b for a, b in zip(res["aleatorio"]["mejoras"], res["NN"]["mejoras"]))
    mejor = sum(b < a for a, b in zip(res["aleatorio"]["final"], res["NN"]["final"]))
    resumen["NN_menos_mejoras_en"] = menos
    resumen["NN_mejor_calidad_en"] = mejor
    resumen["n_instancias"] = N_INSTANCIAS
    print(f"NN necesitó menos mejoras en {menos}/{N_INSTANCIAS}; óptimo local más corto en {mejor}/{N_INSTANCIAS}")
    json.dump(resumen, open(RESULTS / "ej3_resumen.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    etq = ["Aleatorio", "Vecino\nmás cercano"]
    ax[0].boxplot([res["aleatorio"]["mejoras"], res["NN"]["mejoras"]], tick_labels=etq)
    ax[0].set(ylabel="Nº de mejoras 2-opt", title="Mejoras hasta el óptimo local")
    ax[1].boxplot([res["aleatorio"]["final"], res["NN"]["final"]], tick_labels=etq)
    ax[1].set(ylabel="Longitud del tour", title="Calidad del óptimo local")
    medias = [st.mean(res[k]["mejoras"]) for k in ("aleatorio", "NN")]
    ax[2].bar(etq, medias, yerr=[st.pstdev(res[k]["mejoras"]) for k in ("aleatorio", "NN")],
              color=["tab:orange", "tab:blue"], capsize=4)
    ax[2].set(ylabel="Promedio de mejoras", title="Promedio ± desv. estándar")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(f"2-opt: {N_INSTANCIAS} instancias de {N_CIUDADES} ciudades", y=1.02)
    fig.tight_layout(); fig.savefig(RESULTS / "ej3_2opt.png", dpi=130, bbox_inches="tight"); plt.close(fig)
