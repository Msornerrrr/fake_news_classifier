import matplotlib.pyplot as plt
import pandas as pd

def plot_model_info(model_info):
    # Extracting training and validation performance
    train_dev_performance = model_info['performance']['train_dev']

    # Convert to DataFrame for easy plotting
    df_performance = pd.DataFrame(train_dev_performance)

    # Line plot for training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(df_performance['epoch'], df_performance['train_loss'], label='Train Loss')
    plt.plot(df_performance['epoch'], df_performance['dev_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Line plot for validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df_performance['epoch'], df_performance['dev_acc'], label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.show()

    # Bar chart for test performance
    test_performances = model_info['performance']['test']
    metrics = ['test_acc', 'test_prec', 'test_rec', 'test_f1']

    # Determine the number of subplots needed
    num_tests = len(test_performances)
    fig, axes = plt.subplots(nrows=num_tests, ncols=1, figsize=(10, 5 * num_tests))

    # Check if we have only one subplot (axes is not an array in this case)
    if num_tests == 1:
        axes = [axes]

    for i, test_performance in enumerate(test_performances):
        scores = [test_performance[metric] for metric in metrics]
        dataset = test_performance['test_dataset']
        timestamp = test_performance['test_datetime']

        axes[i].bar(metrics, scores, color='orange')
        axes[i].set_xlabel('Metric')
        axes[i].set_ylabel('Score')
        axes[i].set_title(f'Test Performance for Dataset {dataset} at {timestamp}')
        axes[i].set_ylim(0.9, 1.0)  # Adjust limits if necessary
        for j, v in enumerate(scores):
            axes[i].text(j, v + 0.01, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.show()


    # Display hyperparameters in a table
    hyperparameters = model_info.get('hyperparameters', {})
    print("Model Hyperparameters:")
    print(pd.DataFrame(hyperparameters.items(), columns=['Parameter', 'Value']))
