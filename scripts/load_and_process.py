#!/usr/bin/env python3

import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
import traceback
from typing import Set, List, Tuple
from huggingface_hub import dataset_info
from huggingface_hub.repocard import RepoCard



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

        # Get dataset name
        self.dataset_name = input("\nIngresa la ruta del dataset de Hugging Face (ej. isr-innovation/isr-oit-smartbar-images): ").strip()
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
        test_samples = np.random.choice(range(len(self.dataset['train'])), num_samples, replace=False)
        self.class_dict = self.get_classes(self.dataset_name)
        print("Clases en el dataset: ", self.class_dict)
        
        # Crear directorios
        base_path, self.paths = self.setup_directories()
        
        print("\nProcesando imágenes y etiquetas...")
        validation_labels = []
        train_labels = []
        for i in tqdm(range(len(self.dataset['train']))):
            try:
                sample = self.dataset['train'][i]
                image_fname = sample['id']
                image = sample['image']
                labels = sample['labels']
                annotations = sample['annotations']
                
                if i in test_samples:
                    self._process_sample(image, image_fname, labels, validation_labels, annotations, 
                                      self.paths['valid_images'], is_validation=True, class_dict=self.class_dict)
                else:
                    self._process_sample(image, image_fname, labels, train_labels, annotations, 
                                      self.paths['train_images'], is_validation=False, class_dict=self.class_dict)
                    
            except Exception as e:
                print(f"\nError procesando muestra {i}: {str(e)}")
                continue
                    
        self.create_yaml_config()
        self.print_statistics()
        
    def _process_sample(self, image, image_fname: str, labels, labels_list: List, annotations: List, 
                       output_path: str, is_validation: bool, class_dict: dict) -> None: 
        """
        Procesa una muestra individual del dataset
        """
        # Guardar imagen
        image_path = os.path.join(output_path, f"{image_fname}.jpg")
        label_fname = f"{image_fname}.txt"
        image.save(image_path)
            
        # Procesar etiquetas
        labels_list.append(labels) #<- FIX ME, ADDING LABELS HERE!!!
        try:
            if is_validation:
                self.validation_obj += len(labels)
                with open(fr'{self.paths["valid_labels"]}/{label_fname}', "w") as file:
                    for j in range(len(annotations)):
                        label = labels[j]
                        x = annotations[j][0] / 100
                        y = annotations[j][1] / 100
                        h = annotations[j][3] / 100
                        w = annotations[j][2] / 100
                        x_center, y_center = x + (w/2), y+(h/2)
                        class_int = next((k for k, v in class_dict.items() if v == label), None)
                        line_output = (class_int, x_center, y_center, w, h)
                        line = f"{line_output[0]} {line_output[1]:.6f} {line_output[2]:.6f} {line_output[3]:.6f} {line_output[4]:.6f}\n"
                        file.write(line)
            else:
                self.train_obj += len(labels)
                with open(fr'{self.paths["train_labels"]}/{label_fname}', "w") as file:
                    for j in range(len(annotations)):
                        label = labels[j]
                        x = annotations[j][0] / 100
                        y = annotations[j][1] / 100
                        h = annotations[j][3] / 100
                        w = annotations[j][2] / 100
                        x_center, y_center = x + (w/2), y+(h/2)
                        class_int = next((k for k, v in class_dict.items() if v == label), None)
                        line_output = (class_int, x_center, y_center, w, h)
                        line = f"{line_output[0]} {line_output[1]:.6f} {line_output[2]:.6f} {line_output[3]:.6f} {line_output[4]:.6f}\n"
                        file.write(line)
                        
            self.unique_classes.update(labels)
        except TypeError:
            if is_validation:
                self.validation_obj += 0
            else:
                self.train_obj += 0
    
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