import retro
import numpy as np
import tensorflow as tf

import os
import time

scenario = 'reward.json'
checkpoint_path = "./checkpoint/my_dqn.ckpt"
logdir_path = "./logs/1st_train/"
recordings_path = './recordings'

if not os.path.exists(recordings_path):
    os.makedirs(recordings_path)
start_time = time.time()


from queue import Queue
frame_diff_length = 5
diff_queue = Queue(frame_diff_length)
def preprocess_obs(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    diff_queue.put(gray)
    if diff_queue.qsize() >= frame_diff_length:
        old_gray = diff_queue.get()
        gray = 2/3*gray+1/3*old_gray
    gray = (gray  - 128).astype(np.int8)
    return gray[::2,::2].reshape((112, 160, 1))


input_height = 112
input_width = 160
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 20 * 14  # conv3 has 64 maps of 20x14 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = 9  # 9 discrete actions are available
initializer = tf.variance_scaling_initializer()

def q_network(X_state, name):
    prev_layer = X_state / 128.0 # scale pixel intensities to the [-1.0, 1.0] range.
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name


X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)


learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keepdims=True)
    loss = tf.reduce_mean(tf.square(y - q_value)) 

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)
    
# Variables to track progress
with tf.variable_scope("Episode_summary"):
    tf_average_reward = tf.placeholder(dtype = tf.float32, shape = (), name='tf_average_reward')
    tf_games_finished = tf.placeholder(dtype = tf.float32, shape = (), name='tf_games_finished')
    tf_endposition = tf.placeholder(dtype = tf.float32, shape = (), name='tf_endposition')
    tf_game_length = tf.placeholder(dtype = tf.float32, shape = (), name='tf_game_length')
    summary_average_reward = tf.summary.scalar('average_episode_reward', tf_average_reward)
    summary_games_finished = tf.summary.scalar('games_finished', tf_games_finished)
    summary_endposition = tf.summary.scalar('average_final_xposition', tf_endposition)
    summary_game_length = tf.summary.scalar('average_game_length', tf_game_length)

summary = tf.summary.merge([summary_average_reward , summary_games_finished, summary_endposition, summary_game_length ])
file_writer = tf.summary.FileWriter(logdir_path, tf.get_default_graph())
init = tf.global_variables_initializer()
saver = tf.train.Saver()

class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0
        
    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
    
    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size) # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

replay_memory_size = 100000
replay_memory = ReplayMemory(replay_memory_size)

def sample_memories(batch_size):
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

eps_min = 0.1
eps_max = 1
eps_decay_steps = 100000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs), epsilon # random action
    else:
        return np.argmax(q_values), epsilon # optimal action
    
def action2vec(action):
     actions = np.zeros(12, dtype='int8')
     if action == 0:
         actions[0] = 1 # jump
     elif action == 1:
         actions[7] = 1 # right
     elif action == 2:
         actions[6] = 1 # left
     elif action == 3:
         actions[5] = 1 # down
     elif action == 4:
         actions[0] = 1
         actions[7] = 1
     elif action == 5:
         actions[0] = 1
         actions[6] = 1
     elif action == 6:
         actions[0] = 1
         actions[5] = 1
     elif action == 7:
         actions[5] = 1
         actions[7] = 1
     elif action == 8:
         actions[5] = 1
         actions[6] = 1    
     return actions         
         
n_training_step = 800000 # total number of training steps
n_action_reapeats = 20 # reapeat action for a number of frames
training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
save_steps = 1000  # save the model every 1,000 training steps
copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.99
batch_size = 50
max_episode_duration = 90 # stop episode after number of seconds

# Avarege over episodes for graphing 
n_episode_average = 100

# Variables to track progress
episodes_played = 0
iteration = 0
game_frames = 0
total_game_frames = 0
reward = 0
total_episode_reward = 0
cum_game_rewards = 0
cum_games_finished = 0
cum_game_endposition = 0
cum_game_length= 0 


env = retro.make(game='SonicTheHedgehog2-Genesis', state='EmeraldHillZone.Act1', scenario=scenario, record=False)
obs = env.reset()
state = preprocess_obs(obs)
done = False


# Training loop
with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
        
    
    training_step = global_step.eval()
    print('Started training at step: ', training_step)
    
    while True:
        if training_step >= n_training_step:
            break
        iteration += 1

        if done: # game over, start again
            episodes_played += 1

            cum_game_rewards += total_episode_reward
            cum_games_finished += info['screen_x']>=info['screen_x_end']
            cum_game_endposition += info['screen_x']/info['screen_x_end']
            cum_game_length += game_frames/60 # frames/fps
            

            if episodes_played % n_episode_average == 0:
                average_reward = cum_game_rewards/n_episode_average
                average_games_finished = cum_games_finished/n_episode_average
                average_game_endposition = cum_game_endposition/n_episode_average
                average_game_length = cum_game_length/n_episode_average
                tf_episode_summary= sess.run(summary, feed_dict={
                        tf_average_reward: average_reward,
                        tf_games_finished: average_games_finished,
                        tf_endposition: average_game_endposition,
                        tf_game_length: average_game_length})
                file_writer.add_summary(tf_episode_summary, episodes_played)
                
                cum_game_rewards = 0
                cum_games_finished = 0
                cum_game_endposition = 0
                cum_game_length = 0
                

            obs = env.reset()
            state = preprocess_obs(obs)
            game_frames = 0
            total_episode_reward = 0
            
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action, epsilon = epsilon_greedy(q_values, training_step)

        reward = 0
        for i in range(n_action_reapeats):
            obs, reward_notseen, done, info = env.step(action2vec(action))
            game_frames +=1
            total_game_frames +=1
            total_episode_reward += reward_notseen
            reward += reward_notseen/n_action_reapeats

            #env.render() # Watch while training
            
            if done or game_frames > 60*max_episode_duration: # (60 fps)*(x sec)
                done = True
                break
            
        next_state = preprocess_obs(obs)
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state
        
        if iteration%1000 == 0:
            print("\rIteration {},\tTraining step {}/{} ({:.1f})%,\tEpisodes played {:.0f},\tReplay memory size: {}, epsilon: {:.2f},\tTotal time {:.0f} min".format(
                        iteration, training_step,
                        n_training_step, training_step*100/n_training_step,
                        episodes_played,
                        replay_memory.length,
                        epsilon,
                        (time.time() - start_time)/60),
                        end="")

        if iteration < training_start or iteration % training_interval != 0:
            continue # Only if replay memory is sufficent and only every training_interval steps

        # Update network
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        _, training_step = sess.run([training_op, global_step], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})
    

        if training_step % copy_steps == 0:
            copy_online_to_target.run()

        if training_step % save_steps == 0:
            saver.save(sess, checkpoint_path)
            
stop_time = time.time()
total_time = stop_time - start_time
print('\nTotal training steps: {}, Number of episodes: {}, Total game frames: {} Total time: {}'.format(
        training_step, episodes_played, total_game_frames, total_time))
file_writer.close()
env.close()
# Done Training

# Test network
record_games = True
n_episodes = 1

record = recordings_path if record_games else False
env = retro.make(game='SonicTheHedgehog2-Genesis', state='EmeraldHillZone.Act1', scenario=scenario, record=record)
done = False
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            state = preprocess_obs(obs)
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            for i in range(n_action_reapeats):
                action, epsilon = epsilon_greedy(q_values, eps_decay_steps)
                obs, reward, done, info = env.step(action2vec(action))
                env.render()
                if done:
                    break
            
env.render(close=True)
env.close()
