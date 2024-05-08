import chess
import random
import numpy as np
import time
import tensorflow as tf

PIECE_VALUE = {chess.PAWN: 10, chess.KNIGHT: 30, chess.BISHOP: 30, chess.ROOK: 50, chess.QUEEN: 100}

def get_time(func):
    def inner_func(*args, **kwargs):
        init_time = time.time()
        result = func(*args, **kwargs)
        print(f"Función: {func.__name__} Tiempo: {time.time() - init_time}")
        return result
    return inner_func


# Crea el entorno de ajedrez usando python-chess
class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_state()
    
    def step(self, action):
        source = action // 64
        target = action % 64
        reward = 0

        if (self.board.piece_at(source) is not None and
            self.board.piece_at(source).piece_type == chess.PAWN and
            chess.square_rank(target) in (0, 7)):  
            move = chess.Move(source, target, promotion=chess.QUEEN)
        else:
            move = chess.Move(source, target)

        # Recompensa por captura de piezas
        if self.board.is_capture(move):
            piece_captured = self.board.piece_at(target)
            if piece_captured and piece_captured.piece_type != chess.KING:
                reward += PIECE_VALUE.get(piece_captured.piece_type, 0)
        
        self.board.push(move)
        done = int(self.board.is_game_over())

        # Recompensa por poner en jaque
        if self.board.is_check():
            reward += 5

        if move.promotion:
            reward += 100.0  # Recompensa adicional por la promoción

        if self.board.is_checkmate():
            reward += 1000.0  # Recompensa si el agente ha ganado
            if not self.board.turn:
                print("Gana Agent1")
            else:
                print("Gana Agent2")
        elif (self.board.is_stalemate() or self.board.is_insufficient_material()
                or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition()):
            print("Empate")
            reward -= 250
            done = 2
        
        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((8, 8, 12))

        for i in range(64):
            piece = self.board.piece_at(i)

            if piece:
                color = int(piece.color)  # 0 para blanco, 1 para negro
                piece_type = piece.piece_type - 1  # Restamos 1 porque los tipos de piezas van de 1 a 6 en python-chess

                layer = color * 6 + piece_type  # Calculamos la capa correcta para cada tipo de pieza y color

                row = i // 8
                col = i % 8

                state[row, col, layer] = 1

        return state


# Crea el agente de ajedrez
class ChessAgent:
    def __init__(self, model, color):
        self.model = model
        self.color = color
        self.memory = []  # Para almacenar las experiencias pasadas
        self.gamma = 0.95  # Factor de descuento para las recompensas futuras
        self.epsilon = 1.0  # Valor inicial de ε para la política ε-greedy
        self.epsilon_decay = 0.995  # Factor de decaimiento para ε
        self.min_epsilon = 0.01

    @tf.function
    def fast_predict(self, state):
        return self.model(state)

    def get_action(self, state, board):
        # Predicciones del modelo para el estado actual
        action_probs = self.fast_predict(state[None, :, :, :]).numpy()

        # Obtener movimientos legales del juego de ajedrez actual
        legal_moves = list(board.legal_moves)
        legal_actions = [move.from_square * 64 + move.to_square for move in legal_moves]

        legal_action_probs = action_probs[0, legal_actions]

        if np.random.rand() < self.epsilon:
            # Exploración: elige una acción aleatoria de las legales
            action = np.random.choice(legal_actions)
        else:
            # Explotación: elige la acción con la probabilidad más alta de las legales
            action = legal_actions[np.argmax(legal_action_probs)]
        
        # Decrementa epsilon para reducir la exploración a medida que el modelo aprende
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return action

    # def get_action(self, state, board):
    #     # Usa la red neuronal para seleccionar el próximo movimiento
    #     action_probs = self.fast_predict(state[None, :, :, :]).numpy()

    #     legal_moves = list(board.legal_moves)
    #     legal_actions = [move.from_square * 64 + move.to_square for move in legal_moves]

    #     # Filtra las probabilidades de acción para que solo contengan las acciones legales
    #     legal_action_probs = action_probs[0, legal_actions]

    #     if np.random.rand() < self.epsilon:
    #         # Elige una acción aleatoria
    #         action = np.random.choice(legal_actions)
    #     else:
    #         # Elige la acción que maximiza la recompensa
    #         action = legal_actions[np.argmax(legal_action_probs)]

    #     # Decrementa ε después de cada acción
    #     if self.epsilon > 0.01:  # Límite mínimo para ε
    #         self.epsilon *= self.epsilon_decay

    #     return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=128):
        if len(self.memory) < batch_size:
            return  # No entrenar si no hay suficientes experiencias

        minibatch = random.sample(self.memory, batch_size)  # Entrenar con muestras aleatorias

        # # Convertir las experiencias almacenadas en matrices numpy
        # states = np.array([exp[0] for exp in self.memory])
        # actions = np.array([exp[1] for exp in self.memory])
        # rewards = np.array([exp[2] for exp in self.memory])
        # next_states = np.array([exp[3] for exp in self.memory])
        # dones = np.array([exp[4] for exp in self.memory])

        # Desempaquetar las experiencias
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Obtiene las predicciones actuales y futuras
        current_qs = self.model.predict(states)
        next_qs = self.model.predict(next_states)
        
        # Aquí actualizamos los targets para el entrenamiento
        target_qs = np.copy(current_qs)
        max_future_qs = np.max(next_qs, axis=1)

        for i in range(batch_size):
            if dones[i]:
                target_qs[i, actions[i]] = rewards[i]
            else:
                target_qs[i, actions[i]] = rewards[i] + self.gamma * max_future_qs[i]

        # Usa one-hot encoding para convertir los índices de acción en vectores de clasificación
        one_hot_actions = tf.keras.utils.to_categorical(actions, num_classes=4096)

        # Entrena el modelo en el minibatch. Aquí utilizamos 'categorical_crossentropy'
        history = self.model.fit(states, one_hot_actions * target_qs, epochs=5, verbose=0)
        
        # Reduce epsilon
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

        return history

        # # Calcular los targets Q para todas las experiencias
        # targets = self.fast_predict(states).numpy()

        # Q_future = np.amax(self.fast_predict(next_states).numpy(), axis=1)
        # targets[np.arange(len(actions)), actions] = rewards + Q_future * self.gamma * (1 - dones)

        # Entrenar el modelo con el conjunto completo de datos
        # history = self.model.fit(states, targets, epochs=15)

        # return history

    # def old_train(self, batch_size):
    #     # Tomamos una muestra aleatoria de las experiencias pasadas
    #     minibatch = random.sample(self.memory, batch_size)

    #     # Creamos arrays numpy para las experiencias
    #     states = np.array([experience[0] for experience in minibatch])
    #     actions = np.array([experience[1] for experience in minibatch])
    #     rewards = np.array([experience[2] for experience in minibatch])
    #     next_states = np.array([experience[3] for experience in minibatch])
    #     dones = np.array([experience[4] for experience in minibatch])

    #     # Calculamos los valores Q actuales y futuros
    #     current_qs = self.model.predict(states, verbose=0)
    #     future_qs = self.model.predict(next_states, verbose=0)

    #     # Actualizamos los valores Q para las acciones tomadas
    #     for i in range(batch_size):
    #         if dones[i]:
    #             current_qs[i][actions[i]] = rewards[i]
    #         else:
    #             current_qs[i][actions[i]] = rewards[i] + self.gamma * np.max(future_qs[i])

    #     # Entrenamos el modelo en el minibatch
    #     history = self.model.fit(states, current_qs, epochs=15)

    #     return history
