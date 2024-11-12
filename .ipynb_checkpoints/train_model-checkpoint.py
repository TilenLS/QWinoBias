import random, pickle, datetime, sys
import pandas as pd
from lambeq import NumpyModel, Dataset, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss
import numpy as np
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

SEED = random.randint(0, 400)
BATCH_SIZE = 15
EPOCHS = 200

def load_pkl(path: str):
    file = open(path, 'rb')
    data =  pickle.load(file)
    file.close()
    return data

train_data = load_pkl("wino/data/data_final/train_data.pkl")
validation_data = load_pkl("wino/data/data_final/val_data.pkl")
test_data = load_pkl("wino/data/data_final/test_data.pkl")

train_circuits, train_labels, train_diagrams, _ = zip(*train_data)
validation_circuits, validation_labels, validation_diagrams, _ = zip(*validation_data)
test_circuits, test_labels, test_diagrams, _ = zip(*test_data)

total_len = len(train_labels) + len(validation_labels) + len(test_labels)
print("===========(DATA SUMMARY)===========", file=sys.stderr)
print(f"Using batch size of [{BATCH_SIZE}]", file=sys.stderr)
print(f"Training size: {len(train_labels)} ({len(train_labels)/total_len})", file=sys.stderr)
print(f"Validation size: {len(validation_labels)} ({len(validation_labels)/total_len})", file=sys.stderr)
print(f"Test size: {len(test_labels)} ({len(test_labels)/total_len})")
print("=====================================", file=sys.stderr)

model = NumpyModel.from_diagrams(train_circuits + validation_circuits + test_circuits, use_jit=False)

loss = BinaryCrossEntropyLoss(use_jax=True) 
acc = lambda y_hat, y: np.sqrt(np.mean((np.array(y_hat)-np.array(y))**2)/2)

train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
validation_dataset = Dataset(validation_circuits, validation_labels, shuffle=True)
test_dataset = Dataset(test_circuits, test_labels)

def evaluate(parameters, EPOCHS):
    trainer = QuantumTrainer(model,
                             loss_function=loss,
                             optimizer=SPSAOptimizer,
                             epochs=EPOCHS,
                             optim_hyperparams={'a': parameters.get('a', 0.1), 
                                                'c': parameters.get('C', 0.06), 
                                                'A': parameters.get('A', 0.01) * EPOCHS},
                             evaluate_functions={"err": acc},
                             evaluate_on_train=True,
                             verbose='text', 
                             seed=parameters.get('a', 42))
    print("Learning parameters: "+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), file=sys.stderr)
    trainer.fit(train_dataset, validation_dataset, eval_interval=1, log_interval=1)
    test_acc = acc(model(test_dataset.data), test_dataset.targets)
    print(f"Test accuracy: {test_accc}", file=sys.stderr)

ax_client = AxClient()
ax_client.create_experiment(name='bayesianopt',
                            parameters=[{"name": "a", "type": "range", "bounds": [1e-4, 1e-1], "log_scale": True},
                                        {"name": "c", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
                                        {"name": "A", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
                                        {"name": "a", "type": "range", "bounds": [0, 500]}],
                                        objective_name='evaluate_mnist', minimize=False)

for _ in range(50):
    parameters, trial_index = ax_client.get_next_trial()
    print(f"Trial number: [{trial_index}] with parameters", file=sys.stderr)
    print(parameters, file=sys.stderr)
    print('\n')
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

best_parameters, metrics = ax_client.get_best_parameters()
print("The best performing parameters are: ", file=sys.stderr)
print(best_parameters, file=sys.stderr)

