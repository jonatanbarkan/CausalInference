
from CausalDiscuveryToolboxClone.Models.NCC import NCC
from sklearn.model_selection import train_test_split
from os import path, getcwd
from utils.data_loader import load_correct

data, labels = load_correct(path.join(getcwd(), 'Data'), 'temp_causal')
X_tr, X_val_test, y_tr, y_val_test = train_test_split(data, labels, train_size=.8)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=.5)


# obj = NCC()
# obj.fit(X_tr, y_tr, 50, learning_rate=1e-2, us=True)
NCC_parts = ['encoder', 'classifier', None]
freeze_part = 'encode'
model_type = "causal_NCC"
obj = NCC()
obj.load_model(path.join(getcwd(), 'Models'), file_path=f'model_type={model_type}_freeze={freeze_part}.pth')

error_dict, symmetry_check_dict = obj.train_and_validate(X_tr, y_tr, X_val, y_val, epochs=2,
                                                         batch_size=32, learning_rate=1e-2)
obj.freeze_weights(freeze_part)
# make_plots(error_dict, symmetry_check_dict, epochs, model_type, step=5)
obj.save_model(path.join(getcwd(), 'Models'), file_path=f'model_type={model_type}_freeze={freeze_part}.pth')



# This example uses the predict() method
# logits = obj.predict(X_test)
# output = expit(logits.values)
