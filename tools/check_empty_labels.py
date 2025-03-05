import os
import argparse

def analyze_labels(path, label_type):
    """
    Analyze label files in the given dataset path.
    
    Args:
        path (str): Path to the labels directory
        label_type (str): Type of labels (e.g., 'train' or 'valid')
    
    Returns:
        dict: Analysis results including total and empty files
    """
    empty_files = []
    total_files = 0

    # Check if path exists
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
        return None

    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            total_files += 1
            file_path = os.path.join(path, filename)
            file_size = os.path.getsize(file_path)

            if file_size == 0:
                empty_files.append(filename)

    return {
        'total_files': total_files,
        'empty_files': empty_files,
        'empty_percentage': (len(empty_files) / total_files * 100) if total_files > 0 else 0
    }

def check_empty_label_files(base_path):
    """
    Check empty label files in train and valid directories.
    
    Args:
        base_path (str): Base path to the data directory
    """
    # Construct paths to labels directories
    train_labels_path = os.path.join(base_path, 'data', 'train', 'labels')
    valid_labels_path = os.path.join(base_path, 'data', 'valid', 'labels')

    # Analyze train labels
    train_analysis = analyze_labels(train_labels_path, 'train')
    if train_analysis:
        print("\nTrain Labels Analysis:")
        print(f"Total train label files: {train_analysis['total_files']}")
        print(f"Empty train label files: {len(train_analysis['empty_files'])}")
        print(f"Percentage of empty train label files: {train_analysis['empty_percentage']:.2f}%")
        
    # Analyze valid labels
    valid_analysis = analyze_labels(valid_labels_path, 'valid')
    if valid_analysis:
        print("\nValid Labels Analysis:")
        print(f"Total valid label files: {valid_analysis['total_files']}")
        print(f"Empty valid label files: {len(valid_analysis['empty_files'])}")
        print(f"Percentage of empty valid label files: {valid_analysis['empty_percentage']:.2f}%")
        
def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(description='Check for empty label files in train and valid directories.')
    parser.add_argument('base_path', type=str, help='Base path to the data directory')
    
    args = parser.parse_args()
    
    check_empty_label_files(args.base_path)

if __name__ == '__main__':
    main()