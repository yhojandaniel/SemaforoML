import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from collections import deque
import random
import time

# CONFIGURACIÓN DEL MODELO
STATE_SIZE = 5  # [carros_A, carros_B, luz_actual, tiempo_en_luz, hora_del_dia]
ACTION_SIZE = 2 # [0: Mantener luz, 1: Cambiar luz]
MODEL_NAME = "modelo_semaforo.keras" # Usamos la nueva extensión

# LA SIMULACIÓN
class SimuladorTrafico:
    def __init__(self):
        self.tiempo_amarillo = 5
        self.tiempo_min_verde = 10 
        self.reset()

    def reset(self):
        self.carros_A = 0
        self.carros_B = 0
        self.luz_actual = 0  
        self.tiempo_en_luz = 0
        self.tiempo_total_segundos = 0 
        return self._get_estado()

    def _get_estado(self):
        hora_del_dia = (self.tiempo_total_segundos // 3600) % 24
        return np.array([
            self.carros_A, 
            self.carros_B, 
            self.luz_actual, 
            self.tiempo_en_luz, 
            hora_del_dia
        ])

    def step(self, accion):
        if accion == 1 and self.tiempo_en_luz > self.tiempo_min_verde: 
            recompensa_amarillo = 0
            for _ in range(self.tiempo_amarillo):
                self._actualizar_carros(en_amarillo=True) 
                self.tiempo_total_segundos += 1
                recompensa_amarillo -= (self.carros_A + self.carros_B)
                
            self.luz_actual = 1 - self.luz_actual 
            self.tiempo_en_luz = 0
            recompensa_paso = recompensa_amarillo
        else:
            self.tiempo_en_luz += 1
            self.tiempo_total_segundos += 1
            self._actualizar_carros(en_amarillo=False)
            recompensa_paso = - (self.carros_A + self.carros_B)

        nuevo_estado = self._get_estado()
        done = self.tiempo_total_segundos >= 86400 
        
        return nuevo_estado, recompensa_paso, done

    def _actualizar_carros(self, en_amarillo=False):
        hora_del_dia = (self.tiempo_total_segundos // 3600) % 24
        prob_llegada_A, prob_llegada_B = 0, 0

        if 1 <= hora_del_dia <= 5: (prob_llegada_A, prob_llegada_B) = (0.001, 0.001)
        elif 7 <= hora_del_dia <= 9: (prob_llegada_A, prob_llegada_B) = (0.25, 0.1)
        elif 17 <= hora_del_dia <= 19: (prob_llegada_A, prob_llegada_B) = (0.1, 0.22)
        else: (prob_llegada_A, prob_llegada_B) = (0.08, 0.05)

        if random.random() < prob_llegada_A: self.carros_A += 1
        if random.random() < prob_llegada_B: self.carros_B += 1

        if not en_amarillo:
            tasa_flujo = 3 
            if self.luz_actual == 0 and self.carros_A > 0 and self.tiempo_en_luz % tasa_flujo == 0:
                self.carros_A -= 1
            elif self.luz_actual == 1 and self.carros_B > 0 and self.tiempo_en_luz % tasa_flujo == 0:
                self.carros_B -= 1
            
        self.carros_A = max(0, self.carros_A)
        self.carros_B = max(0, self.carros_B)

# EL CEREBRO IA
class AgenteDQN:
    def __init__(self, estado_size, accion_size):
        self.estado_size = estado_size
        self.accion_size = accion_size
        self.memoria = deque(maxlen=200000) 
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995 
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.actualizar_target_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.estado_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.accion_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') 
        return model

    def actualizar_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def guardar_experiencia(self, estado, accion, recompensa, sig_estado, done):
        self.memoria.append((estado, accion, recompensa, sig_estado, done))

    def actuar(self, estado):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.accion_size)
        
        estado_tensor = tf.convert_to_tensor([estado], dtype=tf.float32)
        q_values = self.model(estado_tensor)
        return np.argmax(q_values[0]) 

    def entrenar(self, batch_size):
        if len(self.memoria) < batch_size:
            return 

        minibatch = random.sample(self.memoria, batch_size)
        estados = np.array([e[0] for e in minibatch])
        sig_estados = np.array([e[3] for e in minibatch])
        
        q_actuales = self.model.predict(estados, verbose=0)
        q_futuros = self.target_model.predict(sig_estados, verbose=0)

        targets_q_values = []
        for i, (estado, accion, recompensa, sig_estado, done) in enumerate(minibatch):
            
            if done:
                Q_objetivo = recompensa
            else:
                max_Q_futuro = np.max(q_futuros[i])
                Q_objetivo = recompensa + self.gamma * max_Q_futuro

            target_q = q_actuales[i]
            target_q[accion] = Q_objetivo 
            targets_q_values.append(target_q)
            
        self.model.fit(estados, 
                       np.array(targets_q_values), 
                       epochs=1, 
                       verbose=0,
                       batch_size=batch_size) 

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# BUCLE PRINCIPAL DE ENTRENAMIENTO
EPISODIOS = 100
BATCH_SIZE = 64
PASOS_ENTRENAMIENTO = 4      
PASOS_ACTUALIZAR_TARGET = 50

env = SimuladorTrafico()
agente = AgenteDQN(STATE_SIZE, ACTION_SIZE)

print(f"Iniciando entrenamiento... Simulará {EPISODIOS} días.")
inicio_entrenamiento = time.time()

for e in range(EPISODIOS):
    estado = env.reset()
    score_total = 0 
    done = False
    paso = 0
    
    while not done: 
        paso += 1
        accion = agente.actuar(estado)
        sig_estado, recompensa, done = env.step(accion)
        score_total += recompensa
        agente.guardar_experiencia(estado, accion, recompensa, sig_estado, done)
        estado = sig_estado
        
        if paso % PASOS_ENTRENAMIENTO == 0:
            agente.entrenar(BATCH_SIZE)
        
        if paso % PASOS_ACTUALIZAR_TARGET == 0:
            agente.actualizar_target_model()
        
    print(f"Episodio: {e+1}/{EPISODIOS}, Score: {score_total:.0f}, Epsilon: {agente.epsilon:.3f}")

fin_entrenamiento = time.time()
print("\n¡Entrenamiento completado!")
print(f"Duración total: {fin_entrenamiento - inicio_entrenamiento:.2f} segundos.")
agente.model.save(MODEL_NAME)
print(f"Modelo guardado como '{MODEL_NAME}'")