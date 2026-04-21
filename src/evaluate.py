from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(results):
    evaluations = {}
    for model_name, data in results.items():
        accuracy = accuracy_score(data["y_test"], data["predictions"])
        matrix   = confusion_matrix(data["y_test"], data["predictions"])
        evaluations[model_name] = {"accuracy": accuracy, "matrix": matrix}
    return evaluations