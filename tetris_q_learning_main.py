import numpy as np
from tetris_env import TetrisEnv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

env = TetrisEnv(seed=0)

class LFAAgent:

    def __init__(self, seed):
        """
        initialize the coefficients theta and set hyper-parameters.
        """
        # The following are recommended hyper-parameters.

        # Initial learning rate: 0.01
        # Learning rate decay for each episode: 0.995
        # Minimum learning rate: 0.001
        # Initial epsilon for exploration: 0.5
        # Epsilon decay for each episode: 0.98
        
        self.gamma = 0.9  # Discount factor
        self.theta = np.zeros((640,))  # The weight vector to be learned
        self.learning_rate = 0.01  # Learning rate.
        self.learning_rate_decay = 0.995  # You may decay the learning rate as the training proceeds.
        self.min_learning_rate = 0.001
        self.epsilon = 0.5  # For the epsilon-greedy exploration.
        self.epsilon_decay = 0.98  # You may decay the epsilon as the training proceeds.
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        
        #### Mohamad added this #####
        self.pos_action_list = [0,1,2,3,4]
        self.rot_action_list = [0,1,2,3]
        self.possible_actions = [[e1,e2] for e1 in self.pos_action_list for e2 in self.rot_action_list]
        #############################
        
    def select_action(self, state_game, state_piece):
        """
        This function returns an action for the agent to take.
        Args:
            state_game: s_{gam}, the state of the game area in the current step
            state_piece: s_{pie}, the state of the next piece in the current step
        Returns:
            action_pos: a_{pos}, the position where the agent will put the next piece down
            action_rot: a_{rot}, clockwise rotation of the piece before putting the piece down
        """
        action_q_values = []
        
        for action in self.possible_actions:
            value = np.inner(self.theta, self.phi(state_game,state_piece,action[0],action[1]))
            action_q_values.append(value)
            
        q_values_row = action_q_values
        
        arg_max_actions = np.flatnonzero(q_values_row == np.max(q_values_row)) 
        sub_opt_actions = list(set(np.arange(len(q_values_row))) - set(arg_max_actions) )
        
        if self.rng.random() < self.epsilon:
            if sub_opt_actions:
                idx = np.random.choice(sub_opt_actions)
                action = self.possible_actions[idx]
            else:
                idx = np.random.choice(arg_max_actions)
                action = self.possible_actions[idx]
        else:
            idx = np.random.choice(arg_max_actions)
            action = self.possible_actions[idx]
        
        action_pos = action[0]
        action_rot = action[1]
        # Please complete codes for choosing an action given the current state  
        """
        Hint: You may use epsilon-greedy for exploration. 
        With probability self.epsilon, choose an action uniformly at random;
        Otherwise, choose a greedy action based on the approximated Q values.
        Recall that the Q values are aprroximated by the inner product of the weight vector (self.theta) and the feature vector (self.phi). 
        """
        # we need to choose an action which maximizes Q(state,v)
        # Q(state,u) = np.inner( self.theta, self.phi(state,action))
        return action_pos, action_rot
    
    def train(self, pre_state_game, pre_state_piece, pre_action_pos, pre_action_rot, pre_reward,
              cur_state_game, cur_state_piece, cur_action_pos, cur_action_rot, done):
        """
        This function is used for the update of the Q table
        Args:
            - pre_state_game: the state of the game area in the previous step
            - pre_state_piece: the state of the next piece in the previous step
            - pre_action_pos: the position where the agent puts the next piece down in the previous step
            - pre_action_rot: clockwise rotation of the piece before putting the piece down in the previous step
            - pre_reward: the reward received in the previous step.
            - cur_state_game: the state of the game area in the current step
            - cur_state_piece: the state of the next piece in the current step
            - cur_action_pos: the position where the agent puts the next piece down in the current step
            - cur_action_rot: clockwise rotation of the piece before putting the piece down in the current step
            - `done=0` means that the current episode does not terminate;
              `done=1` means that the current episode terminates.
              We set the length of each episode to be 100.
        """
        
        # Please complete codes for updating the weight vector self.theta
        """
        Hint: You may use the feature function self.phi
              You may use the discount factor self.gamma (=0.9)
        """
        action_q_values = []
        
        for action in self.possible_actions:
            value = np.inner(self.theta, self.phi(cur_state_game,cur_state_piece,action[0],action[1]))
            action_q_values.append(value)
            
        max_q_value = np.max(action_q_values)
        
        delta_k = pre_reward + self.gamma*(max_q_value) -             np.inner(self.theta, self.phi(pre_state_game,pre_state_piece,pre_action_pos,pre_action_rot))
        
        self.theta += self.learning_rate*delta_k*self.phi(pre_state_game,pre_state_piece,pre_action_pos,pre_action_rot)
        
        if done != 0:
            self.learning_rate = self.learning_rate * self.learning_rate_decay
            if self.learning_rate < self.min_learning_rate:
                self.learning_rate = self.min_learning_rate
            self.epsilon = self.epsilon * self.epsilon_decay
    
    @staticmethod
    def rotate(p, action_rot):
        """
        Rotate the piece `p` clockwise.
        Args:
            - p: the piece
            - action_rot: clockwise rotation of the piece. 
                          action_rot = 0, 1, 2, or 3.
                          0 means no rotation.
        Returns:
            - a new piece after the rotation
        """
        while action_rot > 0:
            q = p >> 2
            p = (2 if p & 1 != 0 else 0) + (2 << 2 if p & 2 != 0 else 0) + (
                1 << 2 if q & 2 != 0 else 0) + (1 if q & 1 != 0 else 0)
            action_rot = action_rot - 1
        if p % (1 << 2) == 0:
            p >>= 2
        return p
    
    # For your reference, the following function is an example of the feature vector \phi(s,a)
    # You can directly use this function as \phi(s,a), or you can design your own.
    def phi(self, state_game, state_piece, action_pos, action_rot):
        """
        Implement the feature function phi(s, a)
        Args:
            state_game: s_{gam}, the state of the game area in the current step
            state_piece: s_{pie}, the state of the next piece in the current step
            action_pos: a_{pos}, the position where the agent puts the next piece down in the current step
            action_rot: a_{rot}, clockwise rotation of the piece before putting the piece down in the current step
        Returns:
            feature_vec: feature vector
        """
        feature_vec = np.zeros((640,))
        feature_s_vec = np.zeros((8,))
        h_row = np.unpackbits(np.array([state_game >> 6], dtype=np.uint8))
        l_row = np.unpackbits(np.array([state_game & 63], dtype=np.uint8))
        heights = h_row.astype(int) * 2 + (l_row - h_row == 1).astype(int)
        feature_s_vec[0] = np.max(heights)  # the height of the highest column
        feature_s_vec[1] = np.sum(h_row.astype(int) - l_row.astype(int) == 1)  # holes
        wells = 0
        for i in range(2, 8):
            if (i == 2 or heights[i] - heights[i - 1] < 0) and (i == 7 or heights[i + 1] - heights[i] > 0):
                wells += 1
        feature_s_vec[2] = wells  # wells

        for i in range(3, 8):
            feature_s_vec[i] = heights[i] - heights[i - 1]  # differences in height between neighboring columns

        piece_rotated = self.rotate(state_piece, action_rot)

        action = action_pos * 16 + piece_rotated
        feature_vec[action * 8:(action + 1) * 8] = feature_s_vec

        return feature_vec
    
    @staticmethod
    def visualize(env,action_rot,action_pos):
        # borderlines of the plot
        xmin = 0
        xmax = 6
        ymin = 0
        ymax = 6
        
        # start plotting the game
        state_game, _ = env.get_state()
        
        upper_row = bin(state_game >> 6)
        upper_row = upper_row [2:].zfill(6)

        lower_row = bin(state_game & 63)
        lower_row = lower_row[2:].zfill(6)

        fig, ax = plt.subplots()

        y_lower_left_corner = 0
        x_lower_left_corner = 0

        for i in range(6):
            s = lower_row[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1,color="black"))
            x_lower_left_corner +=1

        y_lower_left_corner += 1
        x_lower_left_corner = 0
        for i in range(6):
            s = upper_row[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1,color="black"))
            x_lower_left_corner +=1

        # start plotting the piece    
        y_lower_left_corner += 1
        state_piece = env.rotate(env.piece,action_rot) << action_pos

        upper_row_piece = bin(state_piece >> 6)
        upper_row_piece = upper_row_piece [2:].zfill(6)

        lower_row_piece = bin(state_piece & 63)
        lower_row_piece = lower_row_piece [2:].zfill(6)

    
        x_lower_left_corner = 0
        for i in range(6):
            s = lower_row_piece[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1,color="red"))
            x_lower_left_corner +=1

        x_lower_left_corner = 0
        y_lower_left_corner += 1

        for i in range(6):
            s = upper_row_piece[i]
            if s == '1':
                ax.add_patch(Rectangle((x_lower_left_corner, y_lower_left_corner), 1, 1,color="red"))
            x_lower_left_corner +=1


        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        plt.xlabel("X - position")
        plt.ylabel("Height")
        plt.title("Tetris Grid")
        plt.grid()
        black_patch = mpatches.Patch(color='black', label='Game Area')
        red_patch = mpatches.Patch(color='red', label='Rotated Piece')
        plt.legend(handles=[red_patch, black_patch])
        plt.show()




if __name__ == '__main__':
    # The code in this cell is provided for debugging and is also a sample test.
    # The following code will train and run your agent in the Teris environment for 500 episodes.
    # The length of each episode is set to be 100.
    # We will print your agent's first 10 actions during the last episode and the total reward averaged over the last 100 episodes.
    # You can check whether your policy is good by this information.
    np.random.seed(0)
    agent = LFAAgent(seed=0)
    print("Your actions during the last episode:")
    total_reward = 0.0
    num_ep =5

    for ep in range(num_ep):
        state_game, state_piece = env.reset()
        pre_state_game = None
        pre_state_piece = None
        pre_action_pos = None
        pre_action_rot = None
        pre_reward = None
        for step in range(100):
            action_pos, action_rot = agent.select_action(state_game, state_piece)
            
            if ep == num_ep - 1 and step < 10:
                print('step %d: ' % (step + 1))
                print('state of the game area:')
                print(format(state_game >> 6, 'b').zfill(6))
                print(format(state_game & 63, 'b').zfill(6))
                print('next piece:')
                print(format(state_piece >> 2, 'b').zfill(2))
                print(format(state_piece & 3, 'b').zfill(2))
                print('actions:')
                print('position=', action_pos, 'rotation=', action_rot)
                agent.visualize(env,action_rot,action_pos)
            
            next_state_game, next_state_piece, reward = env.step(action_pos, action_rot)
            tst_game,test_piece = env.get_state()
            
            if 1 <= step < 99:
                agent.train(pre_state_game, pre_state_piece, pre_action_pos, pre_action_rot, pre_reward,
                            state_game, state_piece, action_pos, action_rot, 0)
            elif step == 99:
                agent.train(pre_state_game, pre_state_piece, pre_action_pos, pre_action_rot, pre_reward,
                            state_game, state_piece, action_pos, action_rot, 1)

            if num_ep - ep <= 100:
                total_reward = total_reward + reward

            pre_state_game = state_game
            pre_state_piece = state_piece
            pre_action_pos = action_pos
            pre_action_rot = action_rot
            pre_reward = reward
            state_game = next_state_game
            state_piece = next_state_piece

    total_reward = total_reward / 100
    print("")
    print("Your total reward averaged over the last %d episodes:\n%.3f" % (100, total_reward))

    # Sample test
    # Check if the total reward is larger than -15
    assert total_reward >= -15, "Sample test, average total reward is less than -15."

