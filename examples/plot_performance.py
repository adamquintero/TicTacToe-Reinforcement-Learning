import os
import subprocess
import matplotlib.pyplot as plt
import json

# Set parameters
train_intervals = 100      # Episodes to train before each benchmark
total_episodes = 10000     # Total episodes for training
bench_episodes = 3000      # Episodes to run during each benchmark
model_file = 'agent.dat'  # Model file to save/load agent state
performance_log = []

# Plot setup
plt.ion()  # Interactive mode for live updates
fig, ax = plt.subplots()
ax.set_xlabel('Episodes')
ax.set_ylabel('Wins')
ax.set_title('Agent Performance During Training')
ax.set_ylim(0,3000) 

# Training and benchmarking loop
for episode in range(train_intervals, total_episodes + 1, train_intervals):
    # Train the agent
    subprocess.run([
        'python', 'v_agent.py', 'learn',
        '-p', str(train_intervals),
        '-f', model_file
    ])

    # Benchmark the agent
    result = subprocess.run([
        'python', 'v_agent.py', 'bench',
        '-p', str(bench_episodes),
        '-f', model_file
    ], capture_output=True, text=True)

    # Parse result and extract number of wins
    bench_result = json.loads(result.stdout.strip())
    wins = bench_result.get('td_win', 0)
    performance_log.append((episode, wins))

    # Plot the performance
    ax.plot(*zip(*performance_log), marker='.', color='b')
    plt.pause(0.01)  # Pause to update plot

# Final plot display
plt.ioff()
plt.show()
