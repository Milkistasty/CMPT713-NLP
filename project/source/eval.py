from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Read the file
out_file = 'output.txt'
with open(out_file, 'r') as file:
    lines = file.readlines()

# Extract predictions and ground truths
predictions, ground_truths = zip(*[line.split() for line in lines])

# Convert to DataFrame for easier manipulation
df = pd.DataFrame({'Prediction': predictions, 'GroundTruth': ground_truths})

# Valid classes and mapping "down" to "flipped"
valid_classes = {"cropped", "blurred", "similar", "decolorized", "flipped"}
df['ValidPrediction'] = df['Prediction'].apply(
    lambda x: 'flipped' if x == 'down' else (x if x in valid_classes else 'wrong')
)

# Metrics calculation
accuracy = accuracy_score(df['GroundTruth'], df['ValidPrediction'])
conf_matrix = confusion_matrix(
    df['GroundTruth'], df['ValidPrediction'], labels=list(valid_classes) + ['wrong']
)
class_report = classification_report(
    df['GroundTruth'], df['ValidPrediction'], labels=list(valid_classes) + ['wrong'], zero_division=0
)

# Print results
print(f"Accuracy: {accuracy*100:.2f}%")
print()
print("Confusion Matrix: ")
print(conf_matrix)
print()
print("Full Class Report: ")
print(class_report)
