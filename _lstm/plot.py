import matplotlib.pyplot as plt
import pandas as pd

from util.json_io import load_json


def plot_model_info(model_id, model_info, save=True):
    num_epochs = model_info['hyperparameters']['num_epochs']

    # Extracting training and validation performance
    train_dev_performance = model_info['performance']['train_dev']

    # Convert to DataFrame for easy plotting
    df_performance = pd.DataFrame(train_dev_performance)

    # Line plot for training and validation loss
    plt.figure(figsize=(1.5 * num_epochs, 5))
    plt.plot(df_performance['epoch'], df_performance['train_loss'], label='Train Loss')
    plt.plot(df_performance['epoch'], df_performance['dev_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if save:
        plt.savefig(f'figures/{model_id}-lstm_train_dev_loss.png')
    plt.show()

    # Line plot for validation accuracy
    plt.figure(figsize=(1.5 * num_epochs, 5))
    plt.plot(df_performance['epoch'], df_performance['dev_acc'], label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    if save:
        plt.savefig(f'figures/{model_id}-lstm_dev_acc.png')
    plt.show()

    # Plot for test performance
    test_performances = model_info['performance']['test']
    num_test = len(test_performances)
    metrics = ['test_acc', 'test_prec', 'test_rec', 'test_f1']

    # Prepare data for plotting
    test_labels = [f"Dataset {perf['test_dataset']} at {perf['test_datetime']}" for perf in test_performances]
    test_data = {metric: [perf[metric] for perf in test_performances] for metric in metrics}

    # Create plot
    plt.figure(figsize=(3 * num_test, 7))
    for metric, values in test_data.items():
        plt.plot(test_labels, values, marker='o', label=metric)

    plt.xlabel('Test Case')
    plt.ylabel('Score')
    plt.title('Test Performance Over Different Cases')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.ylim(0.0, 1.0)  # Adjust limits if necessary
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'figures/{model_id}-test_performance.png')  # Save the plot
    plt.show()


    # Display hyperparameters in a table
    hyperparameters = model_info.get('hyperparameters', {})
    print("Model Hyperparameters:")
    print(pd.DataFrame(hyperparameters.items(), columns=['Parameter', 'Value']))


def plot_hyperparameter_tuning(
        model_info_path, 
        hyperparameter_name, 
        default_hyperparameters,
        default_dataset,
        metrics=['dev_acc', 'dev_f1'],
        save=True):
    # Load all model information
    all_model_info = load_json(model_info_path)

    # Filter models based on criteria
    filtered_models = {}
    for model_id, model_info in all_model_info.items():
        hyperparameters = model_info['hyperparameters']
        if hyperparameter_name in hyperparameters and model_info['dataset'] == default_dataset:
            # Check if other hyperparameters are consistent with the default
            if all(hyperparameters.get(h, default_hyperparameters[h]) == default_hyperparameters[h] for h in default_hyperparameters if h != hyperparameter_name):
                filtered_models[model_id] = model_info

    # Check if there are enough models to compare
    if len(filtered_models) < 2:
        print("Not enough models to compare for hyperparameter tuning.")
        return

    # Prepare data for plotting
    plot_data = {metric: [] for metric in metrics}
    hyperparameter_values = []

    for model_id, model_info in filtered_models.items():
        hyperparameter_values.append(model_info['hyperparameters'][hyperparameter_name])
        for metric in metrics:
            plot_data[metric].append(max([epoch[metric] for epoch in model_info['performance']['train_dev']]))

    # Create DataFrame for plotting
    df = pd.DataFrame({'Hyperparameter Value': hyperparameter_values})
    for metric in metrics:
        df[metric] = plot_data[metric]

    # Sort DataFrame based on hyperparameter value
    df = df.sort_values(by='Hyperparameter Value')

    # Plotting
    plt.figure(figsize=(10, 5))
    for metric in metrics:
        plt.plot(df['Hyperparameter Value'], df[metric], marker='o', label=metric)
        for x, y in zip(df['Hyperparameter Value'], df[metric]):
            plt.text(x, y, f'{y:.5f}', ha='center', va='bottom')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('Metric Value')
    plt.title(f'Effect of {hyperparameter_name} on Performance Metrics')
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(f'figures/LSTM_hyperparameter_tuning-{hyperparameter_name}.png')
    plt.show()