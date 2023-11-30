from const import *
import matplotlib.pyplot as plot

"""
This research is answering the following questions:
What is the overall performance of the tool implementing no feature selection with SVM as the classifier in rice leaf disease classification based on:
1.1. Accuracy
1.2. Precision
1.3. Recall
1.4. F-Score

What is the overall performance of the tool implementing feature selections: A.) PSO, B.) ABC, C.) ACO; with SVM as the classifier in rice leaf disease classification based on:
2.1. Accuracy
2.2. Precision
2.3. Recall
2.4. F-Score

What is the significant difference between the performance of the tool:
3.1. Without feature selector and with PSO as feature selector
3.2. Without feature selector and with ABC as feature selector
3.3. Without feature selector and with ACO as feature selector
3.4. Using PSO as feature selector and using ABC as feature selector
3.5. Using ABC as feature selector and using ACO as feature selector
3.6. Using ACO as feature selector and using PSO as feature selector

Among the swarm optimization techniques with the best performance for feature selection, what is the accuracy of the SVM in classifying:
4.1. Rice Leaf with Bacterial Leaf Blight
4.2. Rice Leaf with Blast
4.3. Rice Leaf with Sheath Blight

"""

def plotMetrics(model):
    categories = []
    values = []
    for c, v in model.metrics['total'].items():
        categories.append(c)
        values.append(v)

    colors = ['red', 'blue', 'green', 'orange']  # Define colors for each bar

    # Creating the bar graph with different colors for each bar
    print(categories, values)
    plot.figure(figsize=(6, 6))
    plot.bar(categories, values, color=colors)

    # Adding titles and labels
    plot.title(f"{model.model.name}")
    plot.xlabel('Categories')
    plot.ylabel('Values')
    plot.ylim = 0
    # Displaying the graph
    plot.tight_layout()
    plot.show()

def plotComparison(model):
    categories = []
    values = []
    colors = ['red', 'blue', 'green', 'orange']
    for c, v in model.metrics.items():
        categories.append(c)
        values.append(v)
    categories = categories[0:4]
    values = [v*100 for v in values[0:4]]

    # Creating the bar graph for set 1
    plot.figure(figsize=(5, 3))
    bar_width = 0.1  # Width of each bar
    index = np.arange(len(categories))  # The label locations
    plot.bar(index - bar_width * 1.5, values[0], bar_width, color=colors[0], label=model.encoder.classes_[0])
    plot.bar(index - bar_width * 0.5, values[1], bar_width, color=colors[1], label=model.encoder.classes_[1])
    plot.bar(index + bar_width * 0.5, values[2], bar_width, color=colors[2], label=model.encoder.classes_[2])
    plot.bar(index + bar_width * 1.5, values[3], bar_width, color=colors[3], label=model.encoder.classes_[3])

    # Adding titles and labels
    plot.title('Per class')
    plot.xlabel('Categories')
    plot.ylabel('Values')
    plot.xticks(index + bar_width / 2, categories)
    plot.legend()

    # Displaying the graph
    plot.tight_layout()
    plot.show()