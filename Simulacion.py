#LIBRERIAS NECESARIAS
import numpy as np              # generacion de numeros aleatorios y operaciones vectorizadas
import networkx as nx           # construccion de los grafos
import matplotlib.pyplot as plt # graficas
import time                     # medir tiempo de ejecucion

#nuestra semilla aleatoria para reproducibilidad
np.random.seed(12082026)

#PARAMETROS GLOBALES
N_NODOS = 20      # numero de nodos en cada grafo
T_PASOS = 1000    # numero de pasos que camina cada caminante
N_SIM = 10000      # numero de caminantes (simulaciones) independientes por grafo
EPSILON = 0.05    # umbral para decidir que la caminata ya "se mezclo"


#FUNCION PARA GENERAR LOS 5 GRAFOS (n = 20 nodos, etiquetados 0..19)
def generar_grafos(n=N_NODOS):
    grafos = {}
    grafos["Camino"] = nx.path_graph(n)          # nodos conectados en linea: 0-1-2-...-19
    grafos["Ciclo"] = nx.cycle_graph(n)          # como el camino, pero el nodo 19 se conecta con el 0
    grafos["Estrella"] = nx.star_graph(n - 1)    # 1 nodo central conectado a los otros 19 (star_graph(k) crea k+1 nodos)
    grafos["Completo"] = nx.complete_graph(n)    # todos los nodos conectados entre si

    G_malla = nx.grid_2d_graph(4, 5)             # rejilla 4x5 = 20 nodos, etiquetados como (fila, col)
    G_malla = nx.convert_node_labels_to_integers(G_malla, ordering="sorted")  # renombramos a 0..19
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
    nodo_inicial = 14
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


#====================================================================
# EJECUCION PRINCIPAL
# Flujo: 1) construir grafo -> 2) obtener matriz P -> 3) simular con P
#        -> 4) graficar frecuencia de visitas -> 5) estimar pi
#        -> 6) calcular y graficar d(t) y tiempo de mezcla
#====================================================================
print("Simulando caminatas aleatorias sobre 5 topologias de grafo (n=20 nodos).")
print(f"Cada grafo corre {N_SIM} caminantes independientes de {T_PASOS} pasos.\n")

grafos = generar_grafos()
resultados = {}   # tipo -> (d_t, t_mezcla, pi, G, H)
tiempos = {}       # tipo -> tiempo de ejecucion individual (segundos)

inicio = time.time()

for tipo, G in grafos.items():
    t0 = time.time()                                # tiempo inicial de este grafo

    P = construir_matriz_transicion(G)             # 2) matriz de transicion del grafo
    H = simular_caminata_con_P(P)                  # 3) simulamos la caminata usando P
    pi = pi_teorica(G)                              # 5) estimamos (calculamos) pi
    d_t, t_mezcla = calcular_dtv_y_tmezcla(H, pi)   # 6) d(t) y tiempo de mezcla
    resultados[tipo] = (d_t, t_mezcla, pi, G, H)

    t1 = time.time()                                # tiempo final de este grafo
    tiempos[tipo] = t1 - t0                          # guardamos cuanto tardo este grafo

    t_mezcla_str = str(t_mezcla) if t_mezcla is not None else f"> {T_PASOS} (no alcanzado)"
    print(f"  {tipo:10s} | |E| = {G.number_of_edges():3d} | t_mezcla = {t_mezcla_str:18s} | tiempo = {tiempos[tipo]:.2f} s")

fin = time.time()
print(f"\nTiempo total de simulacion: {fin - inicio:.2f} s")


#--------------------------------------
# IMPRIMIR MATRIZ DE TRANSICION DEL GRAFO ESTRELLA
#--------------------------------------
np.set_printoptions(precision=3, suppress=True, linewidth=200)  # formato mas legible (3 decimales, sin notacion cientifica)
P_estrella = construir_matriz_transicion(grafos["Estrella"])
print("\nMatriz de transicion P del grafo Estrella:")
print(P_estrella)



#--------------------------------------
# VENTANA 1: frecuencia de visitas por nodo (datos crudos de la simulacion)
# Cuenta, para cada grafo, que proporcion del tiempo total paso cada nodo
# ocupado por algun caminante. Es la "foto" empirica de la simulacion.
#--------------------------------------


fig, ejes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (tipo, (_, _, _, _, H)) in zip(ejes, resultados.items()):
    visitas_promedio = H.sum(axis=0) / N_SIM    # promedio de veces que un caminante paso por ese nodo
    ax.bar(range(N_NODOS), visitas_promedio, color="lightcoral", edgecolor="black")
    ax.set_title(tipo)
    ax.set_xlabel("Nodo")
fig.suptitle("Visitas promedio por nodo (datos de la simulacion)")
ejes[0].set_ylabel("Visitas promedio por caminante")
plt.tight_layout()
plt.show()

#--------------------------------------
# VENTANA 2 (a): d(t) vs pasos para las 5 topologias, sobrepuestas
#--------------------------------------
plt.figure(figsize=(10, 6))
for tipo, (d_t, t_mezcla, _, _, _) in resultados.items():
    plt.plot(d_t, label=tipo, linewidth=1.5)

plt.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1, label=f"epsilon = {EPSILON}")
plt.xlabel("Paso (t)")
plt.ylabel("Distancia de Variacion Total  d(t)")
plt.title("d(t) vs Pasos, para todos los grafos")
plt.legend()
plt.grid(alpha=0.3)
plt.xscale("log")

plt.show()


#--------------------------------------
# VENTANA 3 (b): barras, pasos para llegar al tiempo de mezcla vs tipo de grafo
#--------------------------------------
tipos = list(resultados.keys())
t_mezclas_graf = [resultados[t][1] if resultados[t][1] is not None else T_PASOS for t in tipos]

plt.figure(figsize=(8, 6))
barras = plt.bar(tipos, t_mezclas_graf, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
plt.xlabel("Tipo de grafo")
plt.ylabel("Pasos para llegar al tiempo de mezcla")
plt.title(f"Tiempo de mezcla vs Tipo de grafo (epsilon = {EPSILON})")
plt.grid(axis="y", alpha=0.3)

for barra, valor in zip(barras, [resultados[t][1] for t in tipos]):
    etiqueta = str(valor) if valor is not None else "no alcanzado"
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
              etiqueta, ha="center", va="bottom", fontsize=9)
plt.show()



#--------------------------------------
# VENTANA 4 (al final, aparte): dibujo de las 5 topologias
#--------------------------------------
fig, ejes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (tipo, (_, _, _, G, _)) in zip(ejes, resultados.items()):
    if tipo == "Malla":
        pos = {i: (i % 5, i // 5) for i in range(N_NODOS)}
    else:
        pos = nx.spring_layout(G, seed=23052026)
    nx.draw(G, pos, ax=ax, node_size=60, node_color="lightcoral", edge_color="gray", width=0.8)
    ax.set_title(tipo)
fig.suptitle("Topologias simuladas (n = 20 nodos)")
plt.tight_layout()
plt.show()

print("\nNota: Camino, Ciclo, Estrella y Malla son grafos bipartitos, por lo que la")
print("caminata aleatoria simple es periodica y d(t) nunca converge del todo a 0;")
print("solo oscila cerca del umbral. En Estrella esa oscilacion es muy grande porque")
print("su pi esta muy desbalanceada entre el centro y las hojas.")