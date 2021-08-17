from Version12_generalizedPlan_oneAgent import *
from helper_functions import *
from matplotlib import pyplot as plt

# Save plan to text file if save_plan is True
save_plan = False

# # =================================================================
### Root RRT ###

# Parameters
x0 = Node(np.array([0, 0]))  # Start point
Xi = [[-0.75, 3.75], [0, 17.5]]  # Constraint set
xy_cords = [[0, 1]]  # Indices of xy-coordinates
Delta = 0.5  # Incremental distance in RRT
Q = 0.5 * Delta * 1e3 * np.eye(2)  # Stage cost
QN = 1e4 * np.eye(2)  # Terminal cost
xg1 = np.array([3, 6.25])  # First partially observable goal state
xg2 = np.array([-0.5, 14.5])  # Second partially observable goal state
goal_states = [xg1, xg2]
gamma = 10000  # RRT* radius parameter
eta = 4 * Delta  # RRT* radius parameter
Theta1 = np.array([[1, 0], [0, 0]])  # Observation accuracy matrix
Theta2 = np.array([[0, 0], [0, 1]])  # Observation accuracy matrix
Omega = np.eye(2)  # Partially observable environment transition matrix
b0 = np.array([1 / 2, 1 / 2])  # Initial belief
obstacles = [[[0.75, 3.75], [0, 5]], [[-0.25, 3.75], [7.5, 18]]]  # Obstacles
observation_area1 = ObservationArea([[1.25, 3.75], [5, 7.5]], [Theta1, Theta2])  # First observation area
observation_area2 = ObservationArea([[-0.75, -0.25], [13, 16]], [Theta1, Theta2])  # Second observation area
observation_areas = [observation_area1, observation_area2]  # TODO: Add observation_area2 for experiment
N = 1000  # Number of nodes for final RRT
N_subtrees = 5  # Number of children of each RRT

# Create the root RRT
RRT_root = RRT(start=x0, Xi=Xi, Delta=Delta, Q=Q, QN=QN, goal_states=goal_states, Omega=Omega, v0=b0,
          star=True, gamma=gamma, eta=eta, obstacles=obstacles, observation_areas=observation_areas,
          N_subtrees=N_subtrees)

### Mixed Observable RRT model ###
model = Model(RRT_root, N)



# # =================================================================
# Uncomment to run and plot

plt.figure()
model, best_plan = run_MORRT(model)
plan_ends = flatten_list(best_plan)
plot_environment(model)
model.plot_plan(plan_ends, colors=['r', 'b'])
plt.legend()
plt.plot()



# x_cl_nlp_obs1 = np.loadtxt('observation1.txt')
# x_cl_nlp_obs2 = np.loadtxt('observation2.txt')

### Plot environment
plt.figure()
# Observation Areas
for observation_area in observation_areas:
    x_min, x_max = observation_area.region[0][0], observation_area.region[0][1]
    y_min, y_max = observation_area.region[1][0], observation_area.region[1][1]
    rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5, label='Observation Area')
    plt.gca().add_patch(rectangle)

# Obstacles
for obstacle in obstacles:
    x_min, x_max = obstacle[0][0], obstacle[0][1]
    y_min, y_max = obstacle[1][0], obstacle[1][1]
    rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k", label='Obstacle')
    plt.gca().add_patch(rectangle)

# Goal Regions
plt.plot(xg1[0], xg1[1], 'o', color='r', label = 'Goal Regions')
plt.plot(xg2[0], xg2[1], 'o', color='r')
# # plt.plot(0, 10, 'o', color='r', label='e=3')
plt.annotate('e=1', (xg1[0] + 0.1, xg1[1] + 0.1,))
plt.annotate('e=2', (xg2[0] + 0.1, xg1[1] + 0.1,))
plt.xlim(Xi[0])
plt.ylim(Xi[1])

# Qudaruped sim
# plt.plot(x_cl_nlp_obs1[:, 0], x_cl_nlp_obs1[:, 1], '-*b', label="Closed-loop trajectories")
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# # Qudaruped sim
# plt.plot(x_cl_nlp_obs2[:, 0], x_cl_nlp_obs2[:, 1], '-*r')



plt.legend()
plt.show()

## Save plans as numpy arrays
if save_plan:
    path_obs1 = return_nodes_to_follow(return_subpath(plan_ends[0], 0)) + return_nodes_to_follow(return_subpath(plan_ends[0], 1))
    path_obs2 = return_nodes_to_follow(return_subpath(plan_ends[1], 0)) + return_nodes_to_follow(return_subpath(plan_ends[1], 1))

    path_obs1 = path_to_array(path_obs1)
    path_obs2 = path_to_array(path_obs2)

    pathForObs1 = open("pathForObs1.txt", "w")
    for row in path_obs1:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs1, row, delimiter=',')
    pathForObs1.close()

    pathForObs2 = open("pathForObs2.txt", "w")
    for row in path_obs2:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs2, row, delimiter=',')
    pathForObs2.close()

