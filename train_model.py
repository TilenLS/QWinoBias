import random, pickle, datetime 
import pandas as pd
from lambeq import NumpyModel, Dataset, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss
import numpy as np

SEED = random.randint(0, 400)
BATCH_SIZE = 15
EPOCHS = 200

print("===========(MODEL SUMMARY)===========")
print(f"Training for [{EPOCHS}] epochs with a batch size of [{BATCH_SIZE}] and initialise with seed [{SEED}]")
print(f"Using a numpy model with cross entropy loss and RMSE for accuracy")
print("=====================================")


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
print("===========(DATA SUMMARY)===========")
print(f"Training size: {len(train_labels)} ({len(train_labels)/total_len})")
print(f"Validation size: {len(validation_labels)} ({len(validation_labels)/total_len})")
print(f"Test size: {len(test_labels)} ({len(test_labels)/total_len})")
print("=====================================")

model = NumpyModel.from_diagrams(train_circuits + validation_circuits + test_circuits, use_jit=False)

loss = BinaryCrossEntropyLoss(use_jax=True) 
acc = lambda y_hat, y: np.sqrt(np.mean((np.array(y_hat)-np.array(y))**2)/2)

train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
validation_dataset = Dataset(validation_circuits, validation_labels, shuffle=True)
test_dataset = Dataset(test_circuits, test_labels)

trainer = QuantumTrainer(model,
                         loss_function=loss,
                         optimizer=SPSAOptimizer,
                         epochs=EPOCHS,
                         optim_hyperparams={'a': 0.1, 'c': 0.06, 'A': 0.01 * EPOCHS},
                         evaluate_functions={"err": acc},
                         evaluate_on_train=True,
                         verbose='text', 
                         seed=SEED)


print("Learning parameters: "+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
trainer.fit(train_dataset, validation_dataset, eval_interval=1, log_interval=1)
test_acc = acc(model(test_dataset.data), test_dataset.targets)
print('Test accuracy:', test_acc)
