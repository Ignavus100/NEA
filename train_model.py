import numpy as np
import pickle
from DatabaseAccess import select
from NeuralNetwork import Network, Activation_ReLU, Activation_Softmax, normalizeData, cost_calculation # Import from NeuralNetwork.py

# Removed local definitions of Activation_ReLU, Activation_Softmax, Layer_Dense, Network

def load_training_data(batch_size=100):
    X = []
    y = []
    
    for j in range(batch_size):
        current_sample_features = []
        start_feature_id = 15 * j + 1 
        valid_sample = True
        for i in range(15):
            current_day_id = start_feature_id + i
            row_data = select("c, h, l, o, v", "AAPL", f"ID = {current_day_id}")
            if row_data and len(row_data[0]) == 5:
                current_sample_features.extend(row_data[0])
            else:
                valid_sample = False
                break
        
        if valid_sample and len(current_sample_features) == 75:
            id_current_close = start_feature_id + 14
            id_next_close = start_feature_id + 15
            id_first_day_close_for_threshold = start_feature_id

            data_current_close = select("c", "AAPL", f"ID = {id_current_close}")
            data_next_close = select("c", "AAPL", f"ID = {id_next_close}")
            data_first_day_close_for_threshold = select("c", "AAPL", f"ID = {id_first_day_close_for_threshold}")

            if (data_current_close and data_current_close[0] and 
                data_next_close and data_next_close[0] and 
                data_first_day_close_for_threshold and data_first_day_close_for_threshold[0]):
                
                current_close_price = float(data_current_close[0][0])
                next_close_price = float(data_next_close[0][0])
                first_day_close_price = float(data_first_day_close_for_threshold[0][0])
                
                threshold = 2 * abs(current_close_price - first_day_close_price)
                y.append(1 if next_close_price > current_close_price + threshold else 0)
                X.append(current_sample_features)

    if not X or not y:
        print("Warning: No valid training data loaded.")
        return np.array([]), np.array([])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_and_save_model(epochs=1000, learning_rate=0.01):
    X_train_raw, y_train = load_training_data()
    
    if X_train_raw.size == 0 or y_train.size == 0:
        print("Training cannot proceed: No data loaded.")
        return

    X_train_normalized = normalizeData(X_train_raw) # Use normalizeData from NeuralNetwork.py
    
    if X_train_normalized.size == 0:
        print("Training cannot proceed: X_train is empty after normalization.")
        return
    
    if len(X_train_normalized) != len(y_train):
        print(f"Mismatch in number of samples: X_train has {len(X_train_normalized)}, y_train has {len(y_train)}.")
        return

    # Initialize the Network from NeuralNetwork.py
    # Network(hidden_layers: int, inp_size: int, out_size: int, initial_values: list)
    # Assuming 2 hidden layers, input size 75, output size 2
    # The 'values' parameter in NeuralNetwork.Network constructor initializes the input layer.
    # We'll pass the first training sample for initialization.
    # The NeuralNetwork.py Network has 16 neurons per hidden layer by default.
    model = Network(hidden=2, inp_size=75, out_size=2, values=X_train_normalized[0].tolist())
    cost_func = cost_calculation()

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        
        for i in range(len(X_train_normalized)):
            x_sample = X_train_normalized[i].tolist() # Ensure it's a list for newInput
            y_sample_idx = y_train[i]
            
            # Convert y_sample_idx to one-hot encoding for NeuralNetwork.py's backward pass
            y_sample_one_hot = np.zeros(model.out_size)
            y_sample_one_hot[y_sample_idx] = 1
            
            model.newInput(x_sample)
            output = model.forward() # Returns probabilities from softmax
            
            # The backwards method in NeuralNetwork.py handles weight updates
            model.backwards(y_sample_one_hot, learning_rate=learning_rate)
            
            loss = cost_func.calculate(output, [y_sample_idx]) # cost_func.calculate expects y_true as a list of indices
            total_loss += loss
            
            if np.argmax(output) == y_sample_idx:
                correct_predictions += 1
        
        avg_loss = total_loss / len(X_train_normalized)
        accuracy = correct_predictions / len(X_train_normalized)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as trained_model.pkl")

if __name__ == "__main__":
    train_and_save_model()