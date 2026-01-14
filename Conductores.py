class chofer:
    def __init__(self, pos, nom, s):
        self.g = []  # Lista de ganancias por viaje
        self.X = []  # Lista de tiempos entre llegadas de viajes
        self.pos = pos  # Posición en la fila
        self.nom = nom  # Etiqueta del chofer
        self.s = s  # Estrategia de aceptación (s_l,s_c)
        self.rechazo = 0
        self.tvolver = 0 #Tiempo en el que volverá a la cola
        self.l = None  # Último tiempo en que fue asignado un viaje
        self.ult = None

    def agregar_viaje(self, x, h, v, viajes):
        if v == viajes[0] and self.s[0] <= self.pos:
            self.rechazo = 0
            self.g.append(v)
            self.tvolver = x + h + 1
            if self.l is not None:
                self.X.append(x - self.l)
            self.l = x 
            self.ult = "l"

        elif v == viajes[1] and self.s[1] <= self.pos:
            self.rechazo = 0
            self.g.append(v)
            self.tvolver = x + 1
            if self.l is not None:
                self.X.append(x - self.l)
            self.l = x
            self.ult = "c"


        else:
            self.rechazo=1

    def apos(self, pos):
        self.pos = pos

    def mostrar_informacion(self):
        print(f"Chofer: {self.nom}")
        print(f"Posición en fila: {self.pos}")
        print(f"Ganancias: {self.g}")
        print(f"Ganancias totales: {sum(self.g)}")
        print(f"Tiempos entre llegadas: {self.X}")
        print(f"Tiempo final: {sum(self.X)}")
        print(f"Estrategia de aceptación: {self.s}")
        if len(self.X) != 0:
            print(f"Ganancias por periodo {sum(self.g) / sum(self.X)}")
