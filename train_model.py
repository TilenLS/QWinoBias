import random 
import pandas as pd

SEED = random.randint(0, 400)
BATCH_SIZE = 15
EPOCHS = 200

printf("===========(MODEL SUMMARY)===========")
print(f"Training for [{EPOCHS}] epochs with a batch size of [{BATCH_SIZE}] and initialise with seed [{SEED}]")
print(f"Using a numpy model with cross entropy loss and RMSE for accuracy")
printf("=====================================")


def load_pkl(path: str):
    file = open(path, 'rb')
    data =  pickle.load(file)
    file.close()
    return data

train_data = load_pkl("wino/data/data_final/train_data_2024-09-24_121915.pkl")
validation_data = load_pkl("wino/data/data_final/train_data_2024-09-24_121915.pkl")
test_data = load_pkl("wino/data/data_final/train_data_2024-09-24_121915.pkl")

train_circuits, train_labels, train_diagrams = zip(*train_data)
validation_circuits, validation_labels, validation_diagrams = zip(*validation_data)
test_circuits, test_labels, test_diagrams = zip(*test_data)

total_len = len(train_labels) + len(val_labels) + len(test_labels)
printf("===========(DATA SUMMARY)===========")
print(f"Training size: {len(train_labels)} ({len(train_labels)/total_len})")
print(f"Validation size: {len(val_labels)} ({len(val_labels)/total_len})")
print(f"Test size: {len(test_labels)} ({len(test_labels)/total_len})")
printf("=====================================")

model = NumpyModel.from_diagrams(train_circuits + val_circuits + test_circuits, use_jit=False)

loss = BinaryCrossEntropyLoss(use_jax=True) 
acc = lambda y_hat, y: np.sqrt(np.mean((np.array(y_hat)-np.array(y))**2)/2)

train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels, shuffle=True)
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
trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=1)
test_acc = acc(model(test_dataset.data), test_dataset.targets)
print('Test accuracy:', test_acc)
