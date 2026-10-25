"""Ejercicio 2: greedy de la mochila, menor peso primero vs valor/peso."""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, greedy_knapsack, knapsack_dp

INSTANCIAS = {
    "A (menor peso falla)": dict(nombres=["A", "B", "C"], v=[1, 1, 30], w=[1, 2, 8], W=10),
    "B (valor/peso se atora)": dict(nombres=["X", "Y", "Z"], v=[10, 7, 7], w=[6, 5, 5], W=10),
}

if __name__ == "__main__":
    resumen = {}
    fig, ax = plt.subplots(len(INSTANCIAS), 2, figsize=(10, 6.2))
    for r, (nombre, ins) in enumerate(INSTANCIAS.items()):
        v, w, W, nom = ins["v"], ins["w"], ins["W"], ins["nombres"]
        metodos = {
            "Menor peso primero": greedy_knapsack(v, w, W, key=lambda i: -w[i]),
            "Mayor valor/peso": greedy_knapsack(v, w, W, key=lambda i: v[i] / w[i]),
            "Óptimo (DP)": knapsack_dp(v, w, W),
        }
        print(f"\nInstancia {nombre}  W={W}  items={dict(zip(nom, zip(v, w)))}")
        print(f"{'Método':22s} {'Ítems':10s} {'Valor':>6s} {'Peso usado':>11s}")
        resumen[nombre] = {}
        for m, (idx, val, peso) in metodos.items():
            print(f"{m:22s} {'+'.join(nom[i] for i in idx):10s} {val:6d} {peso:7d}/{W}")
            resumen[nombre][m] = dict(items=[nom[i] for i in idx], valor=val, peso=peso, W=W)
        etiquetas = [f"{m}\n({'+'.join(nom[i] for i in idx)})" for m, (idx, _, _) in metodos.items()]
        vals = [x[1] for x in metodos.values()]
        pesos = [x[2] for x in metodos.values()]
        cols = ["tab:red", "tab:blue", "tab:green"]
        ax[r][0].bar(etiquetas, vals, color=cols)
        ax[r][0].set(ylabel="Valor total", title=f"Instancia {nombre}: valor")
        for i, x in enumerate(vals):
            ax[r][0].text(i, x, str(x), ha="center", va="bottom")
        ax[r][1].bar(etiquetas, pesos, color=cols)
        ax[r][1].axhline(W, color="k", ls="--", label=f"Capacidad W={W}")
        ax[r][1].set(ylabel="Capacidad usada", title="Capacidad usada"); ax[r][1].legend()
        for i, x in enumerate(pesos):
            ax[r][1].text(i, x, str(x), ha="center", va="bottom")
        for a in ax[r]:
            a.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "ej2_mochila.png", dpi=130); plt.close(fig)

    # Búsqueda aleatoria: ¿cuántas veces falla cada regla?
    rng = random.Random(0)
    N, peor_peso, peor_ratio = 2000, 0, 0
    for _ in range(N):
        v = [rng.randint(1, 30) for _ in range(6)]
        w = [rng.randint(1, 15) for _ in range(6)]
        W = sum(w) // 2
        opt = knapsack_dp(v, w, W)[1]
        peor_peso += greedy_knapsack(v, w, W, key=lambda i: -w[i])[1] < opt
        peor_ratio += greedy_knapsack(v, w, W, key=lambda i: v[i] / w[i])[1] < opt
    resumen["aleatorias_2000"] = dict(no_optimo_menor_peso=peor_peso, no_optimo_valor_peso=peor_ratio)
    print(f"\n2000 instancias aleatorias (6 ítems): no óptimo -> menor peso {peor_peso}, valor/peso {peor_ratio}")
    json.dump(resumen, open(RESULTS / "ej2_resumen.json", "w"), indent=2, ensure_ascii=False)
