import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import math
from Cola import SimulacionCola
from statistics import mean


def _compute_edges(vals):
    vals = np.array(sorted(vals))
    if len(vals) == 1:
        d = 0.05
        return np.array([vals[0] - d, vals[0] + d])
    mids = (vals[:-1] + vals[1:]) / 2
    first = vals[0] - (mids[0] - vals[0])
    last = vals[-1] + (vals[-1] - mids[-1])
    edges = np.concatenate(([first], mids, [last]))
    return edges


def graficar_comparacion_equilibrios_1h(
    n,
    h,
    resultados_por_h,
    pronosticos_por_h,
    carpeta_salida=None,
    guardar=True,
    mostrar=True,
    dpi=300,
):
    """
    Draw (and optionally save) the comparison Simulation vs Theoretical
    for a given value of h.
    """

    # ----------------- Validations -----------------
    if h not in resultados_por_h:
        raise ValueError(f"h={h} not in resultados_por_h.")
    if h not in pronosticos_por_h:
        raise ValueError(f"h={h} not in pronosticos_por_h.")

    # ----------------- Output folder -----------------
    if carpeta_salida is None:
        carpeta_salida = f"Resultados eq. para largo {n}"
    os.makedirs(carpeta_salida, exist_ok=True)

    # s_c in {1, ..., n-1}
    s_c_posibles = list(range(1, n + 1))
    cmap = plt.cm.get_cmap("tab10", max(1, len(s_c_posibles)))
    color_by_sc = {sc: cmap(i) for i, sc in enumerate(s_c_posibles)}

    # Special colors
    color_no_existe = "gold"  # s_c = 0
    color_no_equilibrio = "black"  # s_c = -1

    # ----------------- Figure -----------------
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    fig.suptitle(
        f"Comparison Simulation vs Theoretical for h = {h}, n = {n}", fontsize=18
    )

    # =====================================================
    #  INFEASIBLE REGION  (smooth + red custom "hatching")
    # =====================================================

    # Dense grid in [0,1] x [0,1]
    ps_dense = np.linspace(0, 1, 300)
    gammas_dense = np.linspace(0, 1, 300)
    P_dense, G_dense = np.meshgrid(ps_dense, gammas_dense)

    v_l = 1.0
    v_c = G_dense
    # Continuous expression for theoretical s_c
    expr_cont = (
        n * P_dense * (v_l - v_c) / (P_dense * (v_l - v_c) + v_c) - (h - 1) * P_dense
    )
    s_c_dense = np.ceil(expr_cont)
    s_c_dense = np.maximum(1, s_c_dense)

    limite = n - 2 * h + 1
    infeasible = (s_c_dense > limite) & np.isfinite(s_c_dense)

    ny, nx = infeasible.shape

    # Base light-red region (semi-transparent)
    rgba_base = np.zeros((ny, nx, 4))
    rgba_base[infeasible] = [1.0, 0.8, 0.8, 0.7]  # R, G, B, alpha

    # Diagonal red stripes as a "hatching" pattern
    Y_idx, X_idx = np.indices(infeasible.shape)
    # Thin and smooth diagonal stripe pattern
    period = 4  # spacing between diagonal lines (bigger → smoother)
    width = 1  # thickness of the line (smaller → thinner)
    pattern = ((X_idx + Y_idx) % period) < width
    stripes = infeasible & pattern

    rgba_stripes = np.zeros((ny, nx, 4))
    rgba_stripes[stripes] = [1.0, 0.0, 0.0, 0.5]  # softer red, alpha lower

    # Draw base region + stripes on both axes
    for ax in axs:
        ax.imshow(rgba_base, extent=[0, 1, 0, 1], origin="lower", zorder=0)
        ax.imshow(rgba_stripes, extent=[0, 1, 0, 1], origin="lower", zorder=1)

    # ----------------- SIMULATION PANEL -----------------
    axs[0].set_title("Simulation")
    axs[0].set_xlabel(r"p")
    axs[0].set_ylabel(r"$\gamma=\frac{v_c}{v_\ell}$")

    puntos_sim = {sc: {"x": [], "y": []} for sc in s_c_posibles}
    sim_no_existe = {"x": [], "y": []}
    sim_no_equil = {"x": [], "y": []}

    for (gama, p), sc_list in resultados_por_h[h].items():
        for sc in sc_list:
            if sc in s_c_posibles:
                puntos_sim[sc]["x"].append(p)
                puntos_sim[sc]["y"].append(gama)
            elif sc == 0:
                sim_no_existe["x"].append(p)
                sim_no_existe["y"].append(gama)
            elif sc == -1:
                sim_no_equil["x"].append(p)
                sim_no_equil["y"].append(gama)

    for sc in s_c_posibles:
        xs, ys = puntos_sim[sc]["x"], puntos_sim[sc]["y"]
        if xs:
            axs[0].scatter(
                xs,
                ys,
                color=color_by_sc[sc],
                alpha=1,
                edgecolors="k",
                s=100,
                zorder=2,
            )

    if sim_no_existe["x"]:
        axs[0].scatter(
            sim_no_existe["x"],
            sim_no_existe["y"],
            color=color_no_existe,
            alpha=1,
            marker="o",
            edgecolors="k",
            s=100,
            zorder=2,
        )
    if sim_no_equil["x"]:
        axs[0].scatter(
            sim_no_equil["x"],
            sim_no_equil["y"],
            color=color_no_equilibrio,
            alpha=1,
            marker="o",
            edgecolors="k",
            s=100,
            zorder=2,
        )

    # ----------------- THEORETICAL PANEL -----------------
    axs[1].set_title("Theoretical")
    axs[1].set_xlabel("p")

    puntos_teo = {sc: {"x": [], "y": []} for sc in s_c_posibles}
    teo_no_existe = {"x": [], "y": []}
    teo_no_equil = {"x": [], "y": []}

    for (gama, p), sc in pronosticos_por_h[h].items():
        if sc in s_c_posibles:
            puntos_teo[sc]["x"].append(p)
            puntos_teo[sc]["y"].append(gama)
        elif sc == 0:
            teo_no_existe["x"].append(p)
            teo_no_existe["y"].append(gama)
        elif sc == -1:
            teo_no_equil["x"].append(p)
            teo_no_equil["y"].append(gama)

    for sc in s_c_posibles:
        xs, ys = puntos_teo[sc]["x"], puntos_teo[sc]["y"]
        if xs:
            axs[1].scatter(
                xs,
                ys,
                color=color_by_sc[sc],
                alpha=1,
                edgecolors="k",
                s=100,
                zorder=2,
            )

    if teo_no_existe["x"]:
        axs[1].scatter(
            teo_no_existe["x"],
            teo_no_existe["y"],
            color=color_no_existe,
            alpha=1,
            marker="o",
            edgecolors="k",
            s=100,
            zorder=2,
        )
    if teo_no_equil["x"]:
        axs[1].scatter(
            teo_no_equil["x"],
            teo_no_equil["y"],
            color=color_no_equilibrio,
            alpha=1,
            marker="o",
            edgecolors="k",
            s=100,
            zorder=2,
        )

    # ----------------- LEGEND -----------------
    sc_presentes = [
        sc for sc in s_c_posibles if puntos_sim[sc]["x"] or puntos_teo[sc]["x"]
    ]

    handles, labels = [], []
    for sc in sc_presentes:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_by_sc[sc],
                markeredgecolor="k",
                markersize=8,
            )
        )
        labels.append(f"s_c = {sc}")

    hay_no_existe = bool(sim_no_existe["x"] or teo_no_existe["x"])
    hay_no_equil = bool(sim_no_equil["x"] or teo_no_equil["x"])

    if hay_no_existe:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_no_existe,
                markeredgecolor="k",
                markersize=8,
            )
        )
        labels.append("No feasible value")

    if hay_no_equil:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_no_equilibrio,
                markeredgecolor="k",
                markersize=8,
            )
        )
        labels.append("Inequality not satisfied")

    # Legend patch (without hatch, because we are doing custom stripes)
    zone_patch = Patch(facecolor="#ffcccc", edgecolor="red", label="Infeasible region")
    handles.append(zone_patch)
    labels.append("Infeasible region")

    axs[1].legend(
        handles,
        labels,
        title="s_c",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    # ----------------- Format and save -----------------
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 0.86, 1])

    ruta = None
    if guardar:
        ruta = os.path.join(carpeta_salida, f"comparacion_sim_teo_h{h}n{n}.png")
        plt.savefig(ruta, dpi=dpi, bbox_inches="tight")
        print(f"✅ Figure saved in: {ruta}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return ruta


def calculo_info(s_c, n, p, v, h, sc_dominantes, theta, i, cant_sim, r, gama, pronostico):
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RESET = "\033[0m"  # Resetea el color y el estilo a los valores por defecto
    perfil = [(1, s_c)] * n

    ############################################################ s_c-1 ############################################################
    if s_c != 1:
        if h == 1:
            E_desv_del_teo = (v[0] * p**2 + v[1] * (1 - p**2)) / (n + (1 - p) * (1 - s_c))
        else:
            E_desv_del_teo = (v[0] * p**2 + v[1] * (1 - p**2)) / (
                n + (1 - p) * (1 + p**2 - h * p - s_c)
            )

        perfil_desv_del = [(1, s_c)] * (n - 1) + [(1, s_c - 1)]
        sim_desv_dels = {"E": []}
        for _ in range(cant_sim):
            sim_desv_del = SimulacionCola(
                n=n, viajes=v, p=p, r=r, h=h, perfil=perfil_desv_del, theta=theta, i=i
            )
            sim_desv_del.correr()
            sim_desv_dels["E"].append(sim_desv_del.E)
        E_desv_del = mean(sim_desv_dels["E"])
    else:
        E_desv_del = 0
        E_desv_del_teo = 0

    ############################################################ otra dominancia ############################################################
    E_desv_otra_dom_teo = v[1] / (n - h * p + p**h - s_c + 1 / (1 - p))

    perfil_desv_otra_dom = [(1, s_c)] * (n - 1) + [(s_c + 1, 1)]
    sim_desv_otra_doms = {"E": []}
    for _ in range(cant_sim):
        sim_desv_otra_dom = SimulacionCola(
            n=n, viajes=v, p=p, r=r, h=h, perfil=perfil_desv_otra_dom, theta=theta, i=i
        )
        sim_desv_otra_dom.correr()
        sim_desv_otra_doms["E"].append(sim_desv_otra_dom.E)
    E_desv_otra_dom = mean(sim_desv_otra_doms["E"])

    ############################################################ >s_c ############################################################
    if s_c != n:
        perfil_desv_det = [(1, s_c)] * (n - 1) + [(1, s_c + 1)]
        sim_desv_dets = {"E": []}
        for _ in range(cant_sim):
            sim_desv_det = SimulacionCola(
                n=n, viajes=v, p=p, r=r, h=h, perfil=perfil_desv_det, theta=theta, i=i
            )
            sim_desv_det.correr()
            sim_desv_dets["E"].append(sim_desv_det.E)

        E_desv_det = mean(sim_desv_dets["E"])

        E_desv_det_teo = p * v[0] / ((s_c + h * p - p) * (1 - p) + n * p)

    else:
        E_desv_det = 0
        E_desv_det_teo = 0

    sims = {"E": []}
    for _ in range(cant_sim):
        sim = SimulacionCola(n=n, viajes=v, p=p, r=r, h=h, perfil=perfil, theta=theta, i=i)
        sim.correr()
        sims["E"].append(sim.E)

    E_sim = mean(sims["E"])
    E_sim_teo = (v[0] * p + v[1] * (1 - p)) / n

    dif = abs(E_desv_del - E_desv_del_teo)
    dif_2 = abs(E_desv_otra_dom - E_desv_otra_dom_teo)

    color1 = GREEN
    color2 = RESET

    if s_c == pronostico:
        if E_sim + 0.05 >= E_desv_det and E_sim + 0.05 >= E_desv_del and E_sim + 0.05 >= E_desv_otra_dom:
            print(
                f"{BOLD}{color1}s_c = {s_c}: Desvío hacia atrás = (s:{E_desv_det:.2f}, t:{E_desv_det_teo:.2f}), "
                f"Desvío hacía adelante = (s:{E_desv_del:.2f},t:{E_desv_del_teo:.2f}, dif: {dif:.2f}), "
                f"simétrico = (s:{E_sim:.2f}, t:{E_sim_teo:.2f}), "
                f"otra dom = (s:{E_desv_otra_dom:.2f}, t:{E_desv_otra_dom_teo:.2f}, dif: {dif_2:.2f})+0.05{RESET}"
            )
            sc_dominantes.append(s_c)
            eq = True
        else:
            print(
                f"{BOLD}{color2}s_c = {s_c}: Desvío hacia atrás = (s:{E_desv_det:.2f}, t:{E_desv_det_teo:.2f}), "
                f"Desvío hacía adelante = (s:{E_desv_del:.2f},t:{E_desv_del_teo:.2f}, dif: {dif:.2f}), "
                f"simétrico = (s:{E_sim:.2f}, t:{E_sim_teo:.2f}), "
                f"otra dom = (s:{E_desv_otra_dom:.2f}, t:{E_desv_otra_dom_teo:.2f}, dif: {dif_2:.2f})+0.05{RESET}"
            )
            eq = False
    else:
        if E_sim - 0.05 >= E_desv_det and E_sim - 0.05 >= E_desv_del and E_sim - 0.05 >= E_desv_otra_dom:
            print(
                f"{BOLD}{color1}s_c = {s_c}: Desvío hacia atrás = (s:{E_desv_det:.2f}, t:{E_desv_det_teo:.2f}), "
                f"Desvío hacía adelante = (s:{E_desv_del:.2f},t:{E_desv_del_teo:.2f}, dif: {dif:.2f}), "
                f"simétrico = (s:{E_sim:.2f}, t:{E_sim_teo:.2f}), "
                f"otra dom = (s:{E_desv_otra_dom:.2f}, t:{E_desv_otra_dom_teo:.2f}, dif: {dif_2:.2f})-0.05{RESET}"
            )
            sc_dominantes.append(s_c)
            eq = True
        else:
            print(
                f"{BOLD}{color2}s_c = {s_c}: Desvío hacia atrás = (s:{E_desv_det:.2f}, t:{E_desv_det_teo:.2f}), "
                f"Desvío hacía adelante = (s:{E_desv_del:.2f},t:{E_desv_del_teo:.2f}, dif: {dif:.2f}), "
                f"simétrico = (s:{E_sim:.2f}, t:{E_sim_teo:.2f}), "
                f"otra dom = (s:{E_desv_otra_dom:.2f}, t:{E_desv_otra_dom_teo:.2f}, dif: {dif_2:.2f})-0.05{RESET}"
            )
            eq = False

    return sc_dominantes, eq


if __name__ == "__main__":
    print(">>> Arrancó el script")

    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    PURPLE = "\033[35m"
    RESET = "\033[0m"

    # Parámetros base
    n = 8
    i = n
    r = 100000
    theta = 1
    c = 20
    P = [x / c for x in range(1, c)]
    H = list(range(1, int(n / 2) + 1))
    gam = 20
    Gama = [x / gam for x in range(1, gam)]
    cant_sim = 1

    # Diccionario externo: resultados_por_h[h][(gama, p)] = [s_c,...]
    resultados_por_h = {h: {} for h in H}
    # Inicializar diccionario para pronósticos teóricos
    pronosticos_por_h = {h: {} for h in H}

    for h in H:
        for p_idx, p in enumerate(P):
            print(f"p = {p}")
            for gama in Gama:
                print(f"gama = {gama}")
                v = (100, gama * 100)
                sc_dominantes = []

                if h == 1:
                    expr_sup = (n * p * (v[0] - v[1]) / (p * v[0] + (1 - p) * v[1]) + 1)
                else:
                    expr_sup = (
                        n * p * (v[0] - v[1]) / (p * v[0] + (1 - p) * v[1])
                        - h * p
                        + 1
                        + p**2
                    )
                lim_sup = math.floor(expr_sup)

                expr_inf = (n * p * (v[0] - v[1]) / (p * v[0] + (1 - p) * v[1]) - h * p + p)
                lim_inf = math.ceil(expr_inf)

                print(f"h={h}, s_c \\in [{expr_inf:.2f}, {expr_sup:.2f}]")

                if lim_inf <= n - 2 * h + 1:
                    s_min = 1
                    s_max = n - 2 * h + 1
                    if lim_inf == lim_sup or lim_inf < 1 or lim_sup < 1:
                        if lim_inf <= 1:
                            expr_sup = (
                                n * p * (v[0] - v[1]) / (p * v[0] + (1 - p) * v[1])
                                - h * p
                                + p**h
                                + 1 / (1 - p)
                            )
                            if expr_sup >= 1:
                                s_c_eq = 1
                                print(f"{BOLD}{GREEN}Existe un único equilibrio simétrico y es s_c = {s_c_eq}{RESET}")
                                pronostico = s_c_eq
                            else:
                                print(f"{BOLD}{RED}No existe equilibrio simétrico{RESET}")
                                pronostico = -1
                        else:
                            s_c_eq = lim_inf
                            print(f"{BOLD}{GREEN}Existe un único equilibrio simétrico y es s_c = {s_c_eq}{RESET}")
                            pronostico = s_c_eq
                    elif lim_inf > lim_sup:
                        print(f"{BOLD}{RED}No existe equilibrio simétrico{RESET}")
                        pronostico = -1
                    else:
                        s_c_eq = lim_inf
                        print(f"Existe más de un equilibrio simétrico s_c = {s_c_eq}, {min(n - h, lim_sup)}")
                        pronostico = s_c_eq
                else:
                    s_min = n - 2 * h + 2
                    s_max = n
                    print(
                        f"{BOLD}{YELLOW}No existe un equilibrio simétrico que cumpla con s_c<= n-2h+1, se usa {lim_inf}{RESET}"
                    )
                    pronostico = lim_inf

                # Guardar pronóstico
                pronosticos_por_h[h][(gama, p)] = pronostico

                if pronostico >= 1:
                    s_c = pronostico
                    sc_dominantes, eq = calculo_info(
                        s_c, n, p, v, h, sc_dominantes, theta, i, cant_sim, r, gama, pronostico
                    )
                    if eq is False:
                        for s_c in range(s_min, s_max + 1):
                            sc_dominantes, eq = calculo_info(
                                s_c, n, p, v, h, sc_dominantes, theta, i, cant_sim, r, gama, pronostico
                            )
                            if eq is True:
                                break
                else:
                    for s_c in range(s_min, s_max + 1):
                        sc_dominantes, eq = calculo_info(
                            s_c, n, p, v, h, sc_dominantes, theta, i, cant_sim, r, gama, pronostico
                        )
                        if eq is True:
                            break

                # Si no se encontró ningún s_c dominante
                if not sc_dominantes:
                    sc_dominantes.append(-1)

                print()

                # Guardar resultados de este (gama, p) para h actual
                if pronostico == 0:
                    resultados_por_h[h][(gama, p)] = [0]
                else:
                    resultados_por_h[h][(gama, p)] = sc_dominantes

        ruta = graficar_comparacion_equilibrios_1h(
            n=n,
            h=h,
            resultados_por_h=resultados_por_h,
            pronosticos_por_h=pronosticos_por_h,
            carpeta_salida=None,
            guardar=True,
            mostrar=False,
            dpi=300,
        )
