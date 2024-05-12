import chess
import tensorflow as tf

from level3.ia_level import ChessEnvironment, ChessAgent


EPISODES = 3


# def create_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 12)),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(4096, activation='softmax')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
#     return model

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', input_shape=(1, 8, 8)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4096, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')  # Using categorical_crossentropy for a better gradient performance
    return model


def main():
    env = ChessEnvironment()

    # Cargamos o creamos los modelos de los agentes
    try:
        model1 = tf.keras.models.load_model("model_WHITE.h5")
        model1.compile(optimizer='adam', loss='categorical_crossentropy')
        model2 = tf.keras.models.load_model("model_BLACK.h5")
        model2.compile(optimizer='adam', loss='categorical_crossentropy')
        print("Modelos cargados correctamente")
    except:
        print("Creando nuevos modelos...")
        model1 = create_model()
        model2 = create_model()

    # Creamos los agentes
    agent1 = ChessAgent(model1, chess.WHITE)
    agent2 = ChessAgent(model2, chess.BLACK)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward1 = 0
        total_reward2 = 0

        for time in range(500):
            # Agent 1 plays
            action = agent1.predict(state, env.board)
            next_state, reward1, done = env.step(action)
            total_reward1 += reward1
            agent1.remember(state, action, reward1, next_state, done)
            state = next_state

            if done:
                if done == 2:
                    total_reward2 += reward1
                    reward2 = reward1
                elif reward1 > 0:
                    total_reward2 -= reward1
                    reward2 -= reward1
                agent2.remember(state, action, reward2, next_state, done)
                break

            # Agent 2 plays
            action = agent2.predict(state, env.board)
            next_state, reward2, done = env.step(action)
            total_reward2 += reward2
            agent2.remember(state, action, reward2, next_state, done)
            state = next_state
            
            if done:
                if done == 2:
                    total_reward1 += reward2
                    reward1 = reward2
                elif reward2 > 0:
                    total_reward1 -= reward2
                    reward1 -= reward2
                agent1.remember(state, action, reward1, next_state, done)

        print(f"Episodio: {episode+1}/{EPISODES}, Turno: {time+1}, Recompensa total agente1: {total_reward1}, Recompensa total agente2: {total_reward2}")
        #print(f"Episodio: {episode+1}/{EPISODES}, Recompensa total agente1: {total_reward1}, Recompensa total agente2: {total_reward2}")


    # Cada EPISODES partidas guardamos los modelos de los agentes
    breakpoint()
    if episode % 2 == 0 and episode != 0:
        # Entrenamos a los agentes con la memoria acumulada
        print("Entrenando modelos...")
        agent1.train()
        agent2.train()
        
        agent1.model.save(f'model_{episode}_WHITE.h5')
        agent2.model.save(f'model_{episode}_BLACK.h5')


if __name__ == "__main__":
    main()
