# Pikachu-Escape
Built a custom environment using Python and Open AI’s Gymnasium library.
The environment consists of an Agent, Hell states and a Goal state. 
It also has all the necessary components like .reset(), .step(), .render(), .close() methods along with observation, actions, rewards, info and so on.
Q-learning has been implemented in the agent to find the optimal path to the goal state.
Seaborn is used to generate heatmap of the optimal path with Q values.
 main.py: Entry point to the project; handles the training and visualization of the Q-learning agent.
 q_learning.py: Contains the Q-learning algorithm implementation and functions for training and visualizing the Q-table.
 pikachu_escape.py: Defines the custom environment where the agent interacts and learns.
 q_table.npy: Stores the Q-values, representing the learned policy.
