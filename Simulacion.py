#LIBRERIAS NECESARIAS
import numpy as np              # generacion de numeros aleatorios y operaciones vectorizadas
import networkx as nx           # construccion de los grafos
import matplotlib.pyplot as plt # graficas
import matplotlib.ticker as mtick   # para formatear el eje Y como porcentaje
import math                     # para elegir "pasos bonitos" en los ejes Y
import time                     # medir tiempo de ejecucion

#nuestra semilla aleatoria para reproducibilidad
np.random.seed(12082026)

#PARAMETROS GLOBALES
N_NODOS = 25      # numero de nodos en cada grafo
T_PASOS = 10000    # numero de pasos que camina cada caminante
N_SIM = 1000      # numero de caminantes (simulaciones) independientes por grafo
EPSILON = 0.05    # umbral para decidir que la caminata ya "se mezclo"
EPSILON2 = 0.25   # segundo umbral (el que usa el libro), solo para COMPARAR visualmente
                  # en las graficas de d(t); no se usa para calcular ningun t_mezcla


#FUNCION PARA GENERAR LOS 5 GRAFOS (n = 20 nodos, etiquetados 0..19)
def generar_grafos(n=N_NODOS):
    grafos = {}
    grafos["Camino"] = nx.path_graph(n)          # nodos conectados en linea: 0-1-2-...-19
    grafos["Ciclo"] = nx.cycle_graph(n)          # como el camino, pero el nodo 19 se conecta con el 0
    grafos["Estrella"] = nx.star_graph(n - 1)    # 1 nodo central conectado a los otros 19 (star_graph(k) crea k+1 nodos)
    grafos["Completo"] = nx.complete_graph(n)    # todos los nodos conectados entre si

    G_malla = nx.grid_2d_graph(5, 5)             # rejilla 5x5 = 25 nodos, etiquetados como (fila, col)
    G_malla = nx.convert_node_labels_to_integers(G_malla, ordering="sorted")  # renombramos a 0..24
    grafos["Malla"] = G_malla
    return grafos


#FUNCION QUE CONSTRUYE LA MATRIZ DE TRANSICION P DEL GRAFO
def construir_matriz_transicion(G, n=N_NODOS):
    # P[i][j] = 1/deg(i) si existe arista entre i y j, 0 en otro caso
    A = nx.to_numpy_array(G, nodelist=range(n))   # si hay arista entre i y j, si no 0.MATRIZ DE ADYACENCIA
    grados = A.sum(axis=1)                        # grado de cada nodo (suma de su fila para obtener el grado de cada nodo)
    P = A / grados[:, None]                        # dividimos cada fila entre su grado -> probabilidades (1/grado(i))
    return P                                        # GENERA LA MATRIZ DE TRANSICION P (lo de arriba)


#FUNCION QUE SIMULA N_SIM CAMINATAS USANDO DIRECTAMENTE LA MATRIZ P
def simular_caminata_con_P(P, n=N_NODOS, n_sim=N_SIM, t_pasos=T_PASOS):
    cumP = np.cumsum(P, axis=1)   # acumulado de probabilidades por fila, para muestrear por transformada inversa

    # vertice inicial: cada caminante arranca en un nodo elegido uniforme al azar (importante)
    #posiciones = np.random.randint(0, n, size=n_sim)

    # vertice inicial: TODOS los caminantes arrancan fijos en el nodo 0
    nodo_inicial = 0
    posiciones = np.full(n_sim, nodo_inicial, dtype=int)

    H = np.zeros((t_pasos + 1, n), dtype=int)     # H[t][i] = cuantos caminantes estan en el nodo i en el paso t
    H[0] = np.bincount(posiciones, minlength=n)    # cuenta cuantos caminantes iniciaron en cada uno de los n nodos en el instante t=0
                                                   # guarda esa foto inicial en la primera fila de la matriz H

    for t in range(1, t_pasos + 1):
        filas_prob = cumP[posiciones]                            # probabilidades acumuladas del nodo actual de cada caminante
        r = np.random.uniform(0, 1, n_sim)                        # un uniforme por caminante
        posiciones = np.argmax(filas_prob >= r[:, None], axis=1)  # primer nodo cuya prob. acumulada supera r -> siguiente nodo
        H[t] = np.bincount(posiciones, minlength=n)

    return H


#FUNCION: DISTRIBUCION ESTACIONARIA TEORICA
def pi_teorica(G, n=N_NODOS):
    grados = np.array([G.degree(i) for i in range(n)], dtype=float)  # grado de cada nodo
    return grados / grados.sum()   # pi(i) = deg(i) / (2|E|), ya que sum(deg) = 2|E|


#FUNCION: DISTANCIA DE VARIACION TOTAL d(t) Y TIEMPO DE MEZCLA
def calcular_dtv_y_tmezcla(H, pi, n_sim=N_SIM, epsilon=EPSILON):
    mu_t = H / n_sim                               # distribucion empirica en cada paso (normalizamos por num. de caminantes)
    d_t = 0.5 * np.sum(np.abs(mu_t - pi), axis=1)  # d(t) = 1/2 * suma |mu_t(i) - pi(i)| para cada paso t

    bajo_umbral = np.where(d_t < epsilon)[0]        # indices (pasos) donde d(t) ya esta por debajo de epsilon
    t_mezcla = int(bajo_umbral[0]) if bajo_umbral.size > 0 else None  # el primero de esos pasos = tiempo de mezcla
    return d_t, t_mezcla


#FUNCION: TIEMPO DE MEZCLA DE CESARO (formula 6.18 del libro)
#   nu_x^t := (1/t) * sum_{s=1}^{t} P^s(x, .)
# Es decir: en vez de comparar contra pi la distribucion de UN SOLO paso t (mu_t),
# comparamos contra pi el PROMEDIO de las distribuciones de los pasos 1, 2, ..., t.
# Esto "suaviza" la oscilacion de los grafos periodicos (bipartitos), porque promedia
# pasos pares e impares juntos, en vez de mirar un paso aislado.
def calcular_cesaro_dtv_y_tmezcla(H, pi, n_sim=N_SIM, epsilon=EPSILON):
    mu_t = H / n_sim   # mu_t[t] = distribucion empirica en el paso t (t = 0, 1, ..., T_PASOS)

    # suma_acumulada[k] = mu_t[1] + mu_t[2] + ... + mu_t[k+1]   (suma acumulada empezando en s=1, no en s=0)
    suma_acumulada = np.cumsum(mu_t[1:], axis=0)

    # dividimos cada suma acumulada entre el numero de terminos que lleva sumados (t = 1, 2, ..., T_PASOS)
    t_valores = np.arange(1, mu_t.shape[0]).reshape(-1, 1)
    nu_t = suma_acumulada / t_valores   # nu_t[k] = promedio de Cesaro para t = k+1 (formula 6.18)

    # misma formula de TVD de siempre, pero comparando nu_t (el promedio) contra pi, en vez de mu_t
    d_cesaro = 0.5 * np.sum(np.abs(nu_t - pi), axis=1)

    bajo_umbral = np.where(d_cesaro < epsilon)[0]
    # bajo_umbral son indices dentro de d_cesaro, que empieza en t=1 (no en t=0) -> sumamos 1 para el t real
    t_mezcla_cesaro = int(bajo_umbral[0]) + 1 if bajo_umbral.size > 0 else None

    return d_cesaro, t_mezcla_cesaro


#FUNCIONES PARA GRAFICAR 

#FUNCION: elige un "paso bonito" para las marcas del eje Y (1, 2 o 5 x 10^k),
# apuntando a mostrar aproximadamente n_marcas marcas en total.
# Esto evita que dos graficos con valores parecidos terminen con distinta
# cantidad de marcas solo por como matplotlib redondea automaticamente.
def elegir_paso_bonito(valor_max, n_marcas=5):
    if valor_max <= 0:
        return 0.01
    paso_crudo = valor_max / n_marcas
    exponente = math.floor(math.log10(paso_crudo))
    base = paso_crudo / (10 ** exponente)
    if base <= 1.5:
        paso_bonito = 1
    elif base <= 3.5:
        paso_bonito = 2
    elif base <= 7.5:
        paso_bonito = 5
    else:
        paso_bonito = 10
    return paso_bonito * (10 ** exponente)


# formateador manual: convierte 0.05 -> "5%", pero 0.0375 -> "3.75%"
# (mira cada valor individual y quita los ceros sobrantes solo cuando de verdad no hacen falta)
def formato_porcentaje(x, pos):
    valor = x * 100
    texto = f"{valor:.2f}".rstrip("0").rstrip(".")   # quita ceros y el punto si sobran
    if texto == "" or texto == "-":                   # caso especial: x=0 -> "0.00" -> "" tras strip
        texto = "0"
    return texto + "%"


#====================================================================
# EJECUCION PRINCIPAL
# Flujo: 1) construir grafo -> 2) obtener matriz P -> 3) simular con P
#        -> 4) graficar frecuencia de visitas -> 5) estimar pi
#        -> 6) calcular y graficar d(t) y tiempo de mezcla (TVD y Cesaro)
#====================================================================
print("Simulando caminatas aleatorias sobre 5 topologias de grafo (n=20 nodos).")
print(f"Cada grafo corre {N_SIM} caminantes independientes de {T_PASOS} pasos.\n")

grafos = generar_grafos()
resultados = {}   # tipo -> (d_t, t_mezcla, pi, G, H, d_cesaro, t_mezcla_cesaro)
tiempos = {}       # tipo -> tiempo de ejecucion individual (segundos)

inicio = time.time()

for tipo, G in grafos.items():
    t0 = time.time()                                # tiempo inicial de este grafo

    P = construir_matriz_transicion(G)             # 2) matriz de transicion del grafo
    H = simular_caminata_con_P(P)                  # 3) simulamos la caminata usando P
    pi = pi_teorica(G)                              # 5) estimamos (calculamos) pi
    d_t, t_mezcla = calcular_dtv_y_tmezcla(H, pi)   # 6) d(t) y tiempo de mezcla (TVD normal)
    d_cesaro, t_mezcla_cesaro = calcular_cesaro_dtv_y_tmezcla(H, pi)   # 7) d_cesaro(t) y tiempo de mezcla de Cesaro
    resultados[tipo] = (d_t, t_mezcla, pi, G, H, d_cesaro, t_mezcla_cesaro)

    t1 = time.time()                                # tiempo final de este grafo
    tiempos[tipo] = t1 - t0                          # guardamos cuanto tardo este grafo

    t_mezcla_str = str(t_mezcla) if t_mezcla is not None else f"> {T_PASOS} (no alcanzado)"
    t_mezcla_ces_str = str(t_mezcla_cesaro) if t_mezcla_cesaro is not None else f"> {T_PASOS} (no alcanzado)"
    print(f"  {tipo:10s} | |E| = {G.number_of_edges():3d} | t_mezcla(TVD) = {t_mezcla_str:18s} | t_mezcla(Cesaro) = {t_mezcla_ces_str:18s} | tiempo = {tiempos[tipo]:.2f} s")

fin = time.time()
print(f"\nTiempo total de simulacion: {fin - inicio:.2f} s")


#--------------------------------------
# VENTANA 1: frecuencia de visitas por nodo (datos crudos de la simulacion)
# Cuenta, para cada grafo, que proporcion del tiempo total paso cada nodo
# ocupado por algun caminante.
#--------------------------------------

# calculamos primero todas las frecuencias, para poder decidir una escala compartida
frecuencias = {}
for tipo, (_, _, _, _, H, _, _) in resultados.items():
    frecuencias[tipo] = H.sum(axis=0) / H.sum()

grafos_escala_chica = [t for t in frecuencias if t != "Estrella"]
max_chico = max(frecuencias[t].max() for t in grafos_escala_chica)
paso_chico = elegir_paso_bonito(max_chico)

paso_estrella = elegir_paso_bonito(frecuencias["Estrella"].max())

fig = plt.figure(figsize=(20, 9))
gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.6)

ax_camino   = fig.add_subplot(gs[0, 0:2])
ax_ciclo    = fig.add_subplot(gs[0, 2:4])
ax_estrella = fig.add_subplot(gs[0, 4:6])
ax_completo = fig.add_subplot(gs[1, 1:3])   # fila de abajo, centrada (1 columna de margen a cada lado)
ax_malla    = fig.add_subplot(gs[1, 3:5])

ejes_ordenados   = [ax_camino, ax_ciclo, ax_estrella, ax_completo, ax_malla]
tipos_ordenados  = ["Camino", "Ciclo", "Estrella", "Completo", "Malla"]

for ax, tipo in zip(ejes_ordenados, tipos_ordenados):
    frec = frecuencias[tipo]
    ax.bar(range(N_NODOS), frec, color="lightcoral", edgecolor="black")
    ax.set_title(tipo)
    ax.set_xlabel("Nodo")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(formato_porcentaje))
    ax.tick_params(axis="y", labelsize=8)

    if tipo == "Estrella":
        ax.yaxis.set_major_locator(mtick.MultipleLocator(paso_estrella))
    else:
        ax.yaxis.set_major_locator(mtick.MultipleLocator(paso_chico))
        ax.set_ylim(0, max_chico * 1.1)   # mismo rango para los 4 grafos de escala chica

ax_camino.set_ylabel("Frecuencia")
ax_completo.set_ylabel("Frecuencia")

fig.suptitle("Frecuencia de visitas por nodo (datos de la simulación)")
plt.show()


#--------------------------------------
# VENTANA 2 (a): d(t) vs pasos para las 5 topologias, sobrepuestas (DVT normal)
#--------------------------------------
plt.figure(figsize=(10, 6))
for tipo, (d_t, t_mezcla, _, _, _, _, _) in resultados.items():
    plt.plot(d_t, label=tipo, linewidth=1.5)

# Lineas de umbral SIN entrada en la leyenda (label=None / sin label)
plt.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1, label=f"epsilon = {EPSILON}")
plt.axhline(y=EPSILON2, color="gray", linestyle="--", linewidth=1, label=f"epsilon 2 = 0.25")

plt.xlabel("Paso (t)")
plt.ylabel("Distancia de variación total d(t)")
plt.title("d(t) vs Pasos")
plt.grid(alpha=0.3)
plt.xscale("log")

ax = plt.gca()

# Etiquetas epsilon junto a los numeros del eje Y (como tick extra)
# transform=ax.get_yaxis_transform() -> x en coords de los ejes (0 a 1), y en coords de datos
ax.text(-0.01, EPSILON, r"$\epsilon$", transform=ax.get_yaxis_transform(),
         ha="right", va="center", fontsize=10, color="gray", clip_on=False)
ax.text(-0.01, EPSILON2, r"$\epsilon_2$", transform=ax.get_yaxis_transform(),
         ha="right", va="center", fontsize=10, color="gray", clip_on=False)

# Leyenda fuera del area de la grafica (a la derecha), para no dejar hueco vacio adentro
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()


#--------------------------------------
# VENTANA 2b (NUEVA): d_cesaro(t) vs pasos para las 5 topologias, sobrepuestas
#--------------------------------------
plt.figure(figsize=(10, 6))
for tipo, (_, _, _, _, _, d_cesaro, t_mezcla_cesaro) in resultados.items():
    plt.plot(range(1, T_PASOS + 1), d_cesaro, label=tipo, linewidth=1.5)

plt.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1, label=f"epsilon = {EPSILON}")
plt.axhline(y=EPSILON2, color="gray", linestyle="--", linewidth=1, label=f"epsilon 2 = 0.25")

plt.xlabel("Paso (t)")
plt.ylabel("Distancia de Cesaro  d_cesaro(t)")
plt.title("d_cesaro(t) vs Pasos")
plt.grid(alpha=0.3)
plt.xscale("log")

ax = plt.gca()

ax.text(-0.01, EPSILON, r"$\epsilon$", transform=ax.get_yaxis_transform(),
         ha="right", va="center", fontsize=10, color="gray", clip_on=False)
ax.text(-0.01, EPSILON2, r"$\epsilon_2$", transform=ax.get_yaxis_transform(),
         ha="right", va="center", fontsize=10, color="gray", clip_on=False)

ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()


#--------------------------------------
# VENTANA 3: comparacion en barras agrupadas -> t_mezcla (TVD) vs t_mezcla (Cesaro)
# Un grafico de barras lado a lado, por tipo de grafo, para comparar directamente
# los dos tiempos de mezcla (el normal y el de Cesaro).
#--------------------------------------
tipos = list(resultados.keys())

valores_tvd = [resultados[t][1] for t in tipos]   # puede tener None si no convergio
valores_ces = [resultados[t][6] for t in tipos]   # puede tener None si no convergio

alturas_tvd = [v if v is not None else 0 for v in valores_tvd]   # 0 = no se dibuja la barra
alturas_ces = [v if v is not None else 0 for v in valores_ces]

x = np.arange(len(tipos))
ancho = 0.35

plt.figure(figsize=(10, 6))
barras_tvd = plt.bar(x - ancho/2, alturas_tvd, ancho, label="t_mezcla (DVT)", color="#C44E52")
barras_ces = plt.bar(x + ancho/2, alturas_ces, ancho, label="t_mezcla (Cesaro)", color="#55A868")

plt.xlabel("Tipo de grafo")
plt.ylabel("Pasos para llegar al tiempo de mezcla")
plt.title(f"Comparación: Tiempo de mezcla DVT vs Cesaro (epsilon = {EPSILON})")
plt.xticks(x, tipos)
plt.legend()
plt.grid(axis="y", alpha=0.3)

for barra, valor in zip(barras_tvd, valores_tvd):
    etiqueta = str(valor) if valor is not None else "no\nalcanzado"
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
              etiqueta, ha="center", va="bottom", fontsize=6)
for barra, valor in zip(barras_ces, valores_ces):
    etiqueta = str(valor) if valor is not None else "no\nalcanzado"
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
              etiqueta, ha="center", va="bottom", fontsize=6)
plt.show()
