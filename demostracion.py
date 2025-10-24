import numpy as np
import tensorflow as tf
from keras.models import load_model
from entrenamiento import SimuladorTrafico
from entrenamiento import AgenteDQN as AgenteDemo
import random
import time
from collections import deque # Necesario si importas la clase Agente

# --- IMPORTANTE: REPETIR LAS CLASES O IMPORTARLAS ---
# Para que este archivo funcione, debes copiar las clases
# SimuladorTrafico y AgenteDQN (o al menos su esqueleto) aquí,
# o ponerlas en un archivo 'comun.py' e importarlas.
# Aquí las duplicamos por simplicidad:

# CONFIGURACIÓN
STATE_SIZE = 5
ACTION_SIZE = 2
MODEL_NAME = "modelo_semaforo.keras"

# BUCLE PRINCIPAL DE DEMOSTRACIÓN

print("Iniciando simulación de demostración...")
env = SimuladorTrafico()
agente = AgenteDemo(STATE_SIZE, ACTION_SIZE, MODEL_NAME)

estado = env.reset()
done = False

# Códigos de color para la terminal
VERDE = "\033[92m"
ROJO = "\033[91m"
RESET = "\033[0m"

pasos_visualizacion = 50 # Actualizar la pantalla cada 50 pasos (segundos)

while not done:
    # 1. Agente toma la MEJOR decisión (sin exploración)
    accion = agente.actuar(estado)
    
    # 2. Entorno avanza (damos un "salto" para la demo)
    for _ in range(pasos_visualizacion):
        if not done:
            estado, recompensa, done = env.step(accion)
    
    # 3. Extraer datos para visualización
    carros_A = int(estado[0])
    carros_B = int(estado[1])
    luz_idx = int(estado[2])
    tiempo_luz = int(estado[3])
    hora = int(estado[4])
    
    # 4. Preparar strings de visualización
    if luz_idx == 0: # Vía A en Verde
        luz_A_str = f"{VERDE}VERDE ({tiempo_luz}s){RESET}"
        luz_B_str = f"{ROJO}ROJO{RESET}"
    else: # Vía B en Verde
        luz_A_str = f"{ROJO}ROJO{RESET}"
        luz_B_str = f"{VERDE}VERDE ({tiempo_luz}s){RESET}"
        
    # 5. Imprimir la visualización (se actualiza en la misma línea)
    print(f"\r--- HORA: {hora:02d}:00 --- "
          f"VÍA A [🚗 {carros_A:02d}] {luz_A_str:25} | "
          f"VÍA B [🚗 {carros_B:02d}] {luz_B_str:25}", end="")

    # Pausa para visualización (acelerada)
    # time.sleep(0.01) # Ajusta esto para la velocidad deseada

print("\n\nSimulación de 1 día completada con el agente entrenado.")