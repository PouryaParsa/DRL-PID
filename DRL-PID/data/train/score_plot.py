import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, average_time=100):
    running_avg = np.convolve(scores, np.ones(average_time)/average_time, mode='valid')
    plt.plot(x[:len(running_avg)], running_avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Learning Curve')
    plt.savefig(figure_file)
    plt.show()

def plot_data(file_name, figure_file, average_time=100):
    scores = []
    with open(file_name) as f:
        file_content = f.readlines()
    
    for line in file_content:
        scores.append(list(map(float, line.strip('[]').split(", "))))
    
    scores = np.array(scores).flatten()

    x = np.arange(len(scores))
    plot_learning_curve(x, scores, figure_file, average_time)

if __name__ == "__main__":
    figure_file = 'score_plot.png'
    file_name = 'score_data.txt'
    plot_data(file_name, figure_file, 1)

