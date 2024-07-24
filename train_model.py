import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym

# Create the environment
env = gym.make("CartPole-v1")

# Build the model
model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(4,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(env.action_space.n, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Training function
def train_model(episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 4])
        for time in range(500):
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            target = reward
            if not done:
                target += 0.95 * np.amax(model.predict(next_state))
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)
            state = next_state
            if done:
                print(f"Episode: {episode+1}/{episodes}, Score: {time}")
                break

    # Save the trained model
    model.save("cartpole_model.h5")

train_model()
