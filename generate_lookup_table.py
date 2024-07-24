import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("cartpole_model.h5")

states = []
actions = []
for _ in range(10000):
    state = np.random.uniform(low=-1.0, high=1.0, size=(4,))
    action = np.argmax(model.predict(state.reshape(1, -1)))
    states.append(state)
    actions.append(action)

np.save('states.npy', np.array(states))
np.save('actions.npy', np.array(actions))

states = np.load('states.npy')
actions = np.load('actions.npy')

with open('lookup_table.h', 'w') as f:
    f.write('#define N_STATES {}\n\n'.format(states.shape[0]))
    f.write('float states[N_STATES][4] = {\n')
    for state in states:
        f.write('    {' + ', '.join(map(str, state)) + '},\n')
    f.write('};\n\n')

    f.write('int actions[N_STATES] = {\n')
    for action in actions:
        f.write('    {},\n'.format(action))
    f.write('};\n')
