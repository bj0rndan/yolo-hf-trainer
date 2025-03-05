import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
import traceback
from typing import Set, List, Tuple
from huggingface_hub import dataset_info, HfApi, list_datasets
from huggingface_hub.repocard import RepoCard
import inquirer
from concurrent.futures import ThreadPoolExecutor

class DatasetProcessor:
    def __init__(self, validation_split: float = 0.2):
        """
        Inicializa el procesador con el porcentaje de validación
        """
        self.validation_split = validation_split
        self.unique_classes: Set[str] = set()
        self.train_obj = 0
        self.validation_obj = 0
        self.dataset = None
        self.dataset_name = None

    def get_classes(self, repo_id: str, version: str = "main"):
        card_data = dataset_info(repo_id, revision=version, expand=["cardData"]).card_data
        classes_raw = card_data.isr.get("classes", {})
        return {int(k): v for k, v in classes_raw.items()}
    
    def dataset_choice(self):
        dataset_names = []
        hf_api = HfApi(
            endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
            token=os.environ.get("HF_API_KEY"), # Token is not persisted on the machine.
        )

        for i in hf_api.list_datasets(author='isr-innovation'):
            dataset_names.append(i.id)

        try:
            questions = [
                inquirer.List(
                    'dataset',
                    message="Elige el dataset disponible",
                    choices=dataset_names,
                ),
            ]

            answers = inquirer.prompt(questions)

            if answers:
                selected_dataset = answers['dataset']
                print(f"\n✅ Dataset elegido: {selected_dataset}")
                
        except KeyboardInterrupt:
            print("\n❌ Dataset selection cancelled.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return selected_dataset

    def download_dataset(self) -> bool:
        """
        Descarga el dataset de Hugging Face
        """
        try:
            from huggingface_hub import login
            from datasets import load_dataset
        except ImportError:
            print("Error: Se requieren las librerías 'huggingface-hub' y 'datasets'. Instálalas con:")
            print("pip install huggingface-hub datasets")
            return False

        print("\n=== Descargador y Procesador de Datasets de Hugging Face ===\n")

        # Get API key
        api_key = os.environ.get("HF_API_KEY")
        if not api_key:
            print("Error: Se requiere una API key válida. El API key se coge de las variables de entorno de Conda. puedes consultar las variables de entorno actuales con conda env config vars list -n myenv")
            return False

        # Login to Hugging Face
        try:
            login(api_key)
        except Exception as e:
            print(f"Error al iniciar sesión en Hugging Face: {str(e)}")
            return False

        self.dataset_name = self.dataset_choice()
        if not self.dataset_name:
            print("Error: Se requiere un nombre de dataset válido.")
            return False

        # Download dataset
        print(f"\nDescargando dataset '{self.dataset_name}'...")
        try:
            self.dataset = load_dataset(self.dataset_name, data_dir='data')
            print(f"\n¡Dataset descargado con éxito!")
            return True
        
        except Exception as e:
            print(f"\nError al descargar el dataset:")
            print(f"{str(e)}")
            print("\nStack trace completo:")
            traceback.print_exc()
            return False
    
    def setup_directories(self) -> Tuple[str, dict]:
        """
        Crea la estructura de directorios necesaria
        """
        base_path = Path(self.dataset_name) / "data"
        paths = {
            'base': str(base_path),
            'train': str(base_path / "train"),
            'valid': str(base_path / "valid"),
            'train_images': str(base_path / "train" / "images"),
            'valid_images': str(base_path / "valid" / "images"),
            'train_labels': str(base_path / "train" / "labels"),
            'valid_labels': str(base_path / "valid" / "labels")
        }
        
        # Crear directorios
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            
        return str(base_path), paths
    
    def process_dataset(self) -> None:
        """
        Procesa el dataset dividiéndolo en conjuntos de entrenamiento y validación
        """
        if not self.dataset or 'train' not in self.dataset:
            print("Error: No se ha cargado un dataset válido.")
            return
            
        num_samples = int(self.validation_split * len(self.dataset['train']))
        test_samples = set(np.random.choice(range(len(self.dataset['train'])), num_samples, replace=False))
        self.class_dict = self.get_classes(self.dataset_name)
        print("Clases en el dataset: ", self.class_dict)
        
        # Crear directorios
        base_path, self.paths = self.setup_directories()
        
        print("\nProcesando imágenes y etiquetas...")

        def process_sample(i):
            try:
                sample = self.dataset['train'][i]
                image_ext = sample['oid'].split(".")[-1]
                image_fname = sample["id"]
                image = sample['image']
                labels = sample['labels']
                annotations = sample['annotations']
                annotation_id = sample['annotation_id']
                
                # Case 1: annotations == [] and annotation_id is not None
                # Create an empty .txt file (backgound)
                if annotations == [] and annotation_id is not None:
                    is_validation = i in test_samples
                    output_path = self.paths['valid_images'] if is_validation else self.paths['train_images']
                    label_path = (self.paths['valid_labels'] if is_validation else self.paths['train_labels']) 
                    
                    # Save image
                    image_path = os.path.join(output_path, f"{image_fname}.{image_ext}")
                    image.save(image_path)
                    
                    # Create empty label file
                    with open(os.path.join(label_path, f"{image_fname}.txt"), 'w') as f:
                        pass
                    
                    return True
                
                # Case 2: annotations != [] and annotation_id is not None
                # Process the image normally
                elif annotations and annotation_id is not None:
                    is_validation = i in test_samples
                    
                    return self._process_sample_helper(
                        image, f"{image_fname}.{image_ext}", labels, 
                        annotations, is_validation
                    )
                
                # Case 3: annotations == [] and annotation_id == None
                # Skip this sample (not annotated yet)
                else:
                    return None
                
                    
            except Exception as e:
                print(f"\nError procesando muestra {i}: {str(e)}")
                return None

        # Use ThreadPoolExecutor to process samples in parallel, considerable speedup
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_sample, range(len(self.dataset['train']))), total=len(self.dataset['train'])))
        
        self.create_yaml_config()
        self.print_statistics()
        
    def _process_sample_helper(self, image, image_fname: str, labels, 
                                annotations: List, is_validation: bool) -> None:
        """
        Helper method to dispatch sample processing to _process_sample
        """
        validation_labels = [] if is_validation else None
        train_labels = [] if not is_validation else None
        
        output_path = self.paths['valid_images'] if is_validation else self.paths['train_images']
        
        return self._process_sample(image, image_fname, labels, 
                                    validation_labels if is_validation else train_labels, 
                                    annotations, output_path, is_validation, 
                                    self.class_dict)
    
    def _process_sample(self, image, image_fname: str, labels, labels_list: List, annotations: List, 
                       output_path: str, is_validation: bool, class_dict: dict) -> None: 
        """
        Procesa una muestra individual del dataset con manejo robusto de anotaciones
        """
        # Save image with it's extension, conserve the 100 quality
        image_path = os.path.join(output_path, image_fname)
        image_ext = image_fname.split(".")[-1]
        label_fname = f"{image_fname.rstrip(f'.{image_ext}')}.txt"
        image.save(image_path, quality=100)
        
        # Process the labels
        if labels_list is not None:
            labels_list.append(labels)
        
        try:
            label_path = fr'{self.paths["valid_labels"] if is_validation else self.paths["train_labels"]}/{label_fname}'
            with open(label_path, "w") as file:
                for j in range(len(annotations)):
                    label = labels[j]
                    if label == "Unknown":
                        continue
                    
                    class_int = next((k for k, v in class_dict.items() if v == label), None)
                    if class_int is None:
                        print(f"Warning: No class found for label {label}")
                        continue
                    
                    # Robust annotation handling
                    ann = annotations[j]
                    
                    # Handle single coordinate tuple/list 
                    if (isinstance(ann, (tuple, list)) and len(ann) == 4):
                        # Assumes format [x, y, width, height]
                        x, y, w, h = ann
                        x_center, y_center = (x + w/2) / 100, (y + h/2) / 100
                        w, h = w / 100, h / 100
                        line = f"{class_int} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                    
                    # Handle polygon/multi-point annotations
                    elif (isinstance(ann, (list, tuple)) and 
                          all(isinstance(point, (list, tuple)) and len(point) == 2 for point in ann)):
                        # For polygon annotations
                        line = f"{class_int}"
                        for point in ann:
                            line += f" {point[0]/100:.6f} {point[1]/100:.6f}"
                        line += "\n"
                    
                    else:
                        print(f"Warning: Unhandled annotation format for {image_fname} at index {j}")
                        continue
                    
                    file.write(line)
                
                # Update object counts
                if is_validation:
                    self.validation_obj += len(labels)
                else:
                    self.train_obj += len(labels)
                    
                self.unique_classes.update(labels)
        
        except Exception as e:
            print(f"Error processing {image_fname}: {e}")
            traceback.print_exc()
    
    def create_yaml_config(self) -> None:
        """
        Crea el archivo de configuración YAML
        """
        unique_classes_list = list(self.unique_classes)
        file_content = f"""train: {'/home/jovyan/work/' + self.paths['train']}
val: {'/home/jovyan/work/' + self.paths['valid']}
nc: {len(list(self.class_dict.values()))}
names: {list(self.class_dict.values())}
"""
        file_path = os.path.join(self.paths['base'], 'data.yaml')
        with open(file_path, 'w') as file:
            file.write(file_content)
        print(f'\nConfig guardado en {file_path}')
    
    def print_statistics(self) -> None:
        """
        Imprime estadísticas del proceso de división
        """
        print("\nEstadísticas del dataset:")
        print(f"Imágenes de entrenamiento: {len(os.listdir(self.paths['train_images']))}")
        print(f"Imágenes de validación: {len(os.listdir(self.paths['valid_images']))}")
        print("--------------------------------------------")
        print(f"Objetos en conjunto de entrenamiento: {self.train_obj}")
        print(f"Objetos en conjunto de validación: {self.validation_obj}")
        print(f"Número total de clases únicas: {len(self.unique_classes)}")

def main():
    try:
        # Inicializar y ejecutar el procesador
        processor = DatasetProcessor(validation_split=0.2)
        
        # Descargar dataset
        if not processor.download_dataset():
            return
        
        # Procesar dataset
        processor.process_dataset()
        
        print("\nProceso completado!")
        
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
