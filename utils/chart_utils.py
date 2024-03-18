import json
import matplotlib.pyplot as plt

import numpy as np

def draw_chart(json_path, class_name='cat'):
    """draw chart of test accuracy and test class accuracy

    Args:
        json_path (str): json_path
        class_name (str, optional): Class Name. Defaults to 'cat'.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    epochs = len(data)
    x = [i for i in range(epochs)]
    y = [data[i]['test_acc'] for i in range(epochs)]
    y_cat = [data[i]['test_class_acc'][class_name]['correct_rate'] for i in range(epochs)]

    plt.plot(x, y, lw=2, label='Test Accuray')
    plt.plot(x, y_cat, lw=2, label='Test Cat Accuary')
    
    plt.xlabel('Number of epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(np.arange(0, epochs+1, 2))
    plt.title('Evolution of testSet accuary of all and specific class during training process', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid()
    plt.show()