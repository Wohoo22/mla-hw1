import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

visualize_tree = True
replace_feature_index_with_feature_name = True
calculate_information_gain = True

def load_data(fake_file='clean_fake.txt', real_file='clean_real.txt'):
    # Load the data from the text files
    with open(fake_file, 'r') as f:
        fake_news = f.readlines()

    with open(real_file, 'r') as f:
        real_news = f.readlines()

    # Add labels (0 for fake, 1 for real)
    fake_labels = [0] * len(fake_news)
    real_labels = [1] * len(real_news)

    # Combine the data and labels
    all_news = fake_news + real_news      
    all_labels = fake_labels + real_labels

    # Preprocess the data using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(all_news)

    # Split the data into training (70%), validation (15%), and test (15%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", len(y_train))
print("y_val shape:", len(y_val))
print("y_test shape:", len(y_test))

def select_model(X_train, y_train, X_val, y_val, X_test, y_test, vectorizer):
    # Define different values for max_depth and criteria
    max_depth_values = [3, 5, 10, 15, 20]  # 5 different values of max_depth
    criteria = ['gini', 'entropy', 'log_loss']  # 3 split criteria

    # Dictionary to store results
    results = {}

    # Variable to track the best model and its accuracy
    best_accuracy = 0
    best_params = None
    best_model = None

    # Loop through each combination of max_depth and criterion
    for depth in max_depth_values:
        for criterion in criteria:
            # Initialize the DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=42)

            # Train the model on the training set
            model.fit(X_train, y_train)

            # Predict on the validation set
            y_pred = model.predict(X_val)

            # Calculate accuracy on the validation set
            accuracy = accuracy_score(y_val, y_pred)

            # Store the accuracy in the results dictionary
            results[(depth, criterion)] = accuracy

            # Print the model's configuration and its accuracy
            print(f"max_depth: {depth}, criterion: {criterion}, validation accuracy: {accuracy:.4f}")

            # Check if this is the best model so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (depth, criterion)
                best_model = model

    # Once the best model is found, evaluate it on the test set
    if best_model is not None:
        y_test_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Print the test accuracy of the best model
        print("\nBest model hyperparameters:")
        print(f"max_depth: {best_params[0]}, criterion: {best_params[1]}, validation accuracy: {best_accuracy:.4f}")
        print(f"Test accuracy for best model: {test_accuracy:.4f}")

        if visualize_tree is True:
        # Visualize only the first two layers of the best decision tree
            plt.figure(figsize=(20, 10))  # Set the figure size

            if replace_feature_index_with_feature_name is True:
                # Get feature names from the vectorizer
                feature_names = vectorizer.get_feature_names_out()
                
                # Plot the tree with feature names
                plot_tree(best_model, max_depth=2, filled=True, feature_names=feature_names, class_names=['Fake', 'Real'], rounded=True)
            else:
                # Plot the tree with feature index instead of feature names
                plot_tree(best_model, max_depth=2, filled=True, feature_names=None, class_names=['Fake', 'Real'], rounded=True)
            
            plt.title(f"Decision Tree (max_depth={best_params[0]}, criterion={best_params[1]}) - First Two Layers")
            plt.show()

    return results

results = select_model(X_train, y_train, X_val, y_val, X_test, y_test, vectorizer)

def compute_entropy(y):
    """Compute the entropy of a label array."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)

def compute_information_gain(X, y, feature_index):
    """Compute the information gain of a split based on a feature."""
    # Convert X to a dense array if it's sparse
    if hasattr(X, "toarray"):  # Check if it's a sparse matrix
        X = X.toarray()

    # Ensure y is a NumPy array for boolean indexing
    y = np.array(y)
    
    # Compute the entropy of the entire dataset
    initial_entropy = compute_entropy(y)
    
    # Get the unique values of the feature
    feature_values = np.unique(X[:, feature_index])
    
    # Compute the weighted average of the entropy after the split
    weighted_entropy = 0
    for value in feature_values:
        subset = (X[:, feature_index] == value)
        subset_indices = np.where(subset)[0]  # Get the indices of the subset
        subset_entropy = compute_entropy(y[subset_indices])
        weighted_entropy += (len(subset_indices) / len(y)) * subset_entropy
    
    # Information Gain is the difference between the initial entropy and the weighted average entropy
    information_gain = initial_entropy - weighted_entropy
    return information_gain

def print_information_gains(X, y, feature_names):
    """Print information gains for the topmost split and several other features."""
    # Assuming X is a NumPy array and y is a NumPy array of labels
    # Example feature indices; you can adapt based on your specific data
    top_split_feature_index = 1290  # donald
    feature_indices = [top_split_feature_index] + [4793, 124, 4797]  # 4792 - trump, 124 - america, 4797 - trumps

    print("Information Gains:")
    for index in feature_indices:
        info_gain = compute_information_gain(X, y, index)
        print(f"Feature: {feature_names[index]}, Information Gain: {info_gain:.4f}")


if calculate_information_gain is True:
    feature_names = vectorizer.get_feature_names_out()
    print_information_gains(X_train, y_train, feature_names)