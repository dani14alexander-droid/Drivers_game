from Conductores import chofer
import random

class SimulacionCola:
    """
    Simulación optimizada de la cola (θ = 1).
    Guarda solo métricas del chofer i:
      - X: lista de entre-llegadas X_k^i
      - g: lista de ganancias R_k^i
      - E: promedio de largo plazo sum(g)/sum(X)
    """

    def __init__(self, n: int, viajes: tuple, p: float, r: int, h: int,
                 perfil: list, theta: float, i: int):
        self.n = n
        self.viajes = viajes            # (v_largo, v_corto)
        self.p = p
        self.r = r
        self.h = h
        self.perfil = perfil            # [(s_l, s_c)]*n
        self.theta = theta              # ignorado (θ=1), se mantiene por compatibilidad
        self.i = i

        # métricas SOLO del chofer i
        self.X = []                     # entre-llegadas
        self.g = []                     # ganancias por aceptación
        self.E = 0.0                    # sum(g)/sum(X) al final

    def correr(self, info: bool = False):
        n, h, p = self.n, self.h, self.p
        v_l, v_c = self.viajes
        i = self.i

        # Cola efectiva y referencia directa al chofer i
        cola = [chofer(pos=k, nom=f"{k}", s=self.perfil[k-1]) for k in range(1, n+1)]
        ch_i = cola[i-1]  # referencia estable

        # Agenda de reingresos: t -> [chofer, ...]
        reingresos = {}

        # Acumuladores para E sin recorrer listas al final
        sum_X = 0.0
        sum_g = 0.0

        for t in range(self.r):
            # 1) Reingresos programados para t
            if t in reingresos:
                reintegrados = reingresos.pop(t)

                # θ = 1: si hay l y c, los 'l' entran antes; entre iguales se respeta el orden
                if len(reintegrados) >= 2:
                    reintegrados.sort(key=lambda ch: 0 if getattr(ch, "ult", None) == "l" else 1)

                # anexar en orden
                cola.extend(reintegrados)

            # 2) Actualizar posiciones actuales
            # (enumerar directo es más barato que múltiples len/assign)
            for idx, ch in enumerate(cola,  start=1):
                ch.pos = idx

            # 3) Sorteo del viaje de la ronda (sin construir V)
            v_actual = v_l if random.random() < p else v_c

            # 4) Recorremos la cola hasta que alguien acepte
            #    (agregar_viaje debe setear: rechazo, ult ('l'/'c') y tvolver si acepta)
            acepto_idx = -1
            for idx, ch in enumerate(cola):
                ch.agregar_viaje(x=t, h=h, v=v_actual, viajes=(v_l, v_c))
                if ch.rechazo == 0:         # aceptó
                    acepto_idx = idx
                    break

            if acepto_idx >= 0:
                ch = cola.pop(acepto_idx)    # sale del sistema
                tv = getattr(ch, "tvolver", None)
                if tv is not None:
                    reingresos.setdefault(tv, []).append(ch)  # agenda reingreso

            # 5) Métricas SOLO del chofer i (se actualizan internamente en agregar_viaje)
            #    Aquí solo acumulamos para E
            #    (las listas X y g del chofer i ya están en el objeto ch_i)
            #    Nota: si aún no aceptó nunca, len(ch_i.X) puede ser 0 y no sumamos nada
            if ch_i.X:
                # sumar solo lo nuevo desde la última ronda
                # (guardamos contadores para no re-sumar; más barato)
                pass  # vemos abajo

        # --- Recolectar métricas del chofer i ---
        # Copiamos listas completas (número de aceptaciones típicamente ~ r/n)
        self.X = list(ch_i.X)
        self.g = list(ch_i.g)
        sum_X = float(sum(self.X))
        sum_g = float(sum(self.g))
        self.E = (sum_g / sum_X) if sum_X > 0 else 0.0
