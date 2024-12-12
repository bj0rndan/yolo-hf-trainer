import os
import sys
import yaml
import argparse
import glob
from collections import defaultdict
import tempfile
from pathlib import Path
import traceback
from typing import Dict, Any
import comet_ml
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate

class YOLOTrainer:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el trainer de YOLO
        Args:
            config: Diccionario con la configuración o None para usar argumentos CLI
        """
        self.config = config
        self.model = None
        self.comet_api_key = os.environ.get("COMET_API_KEY")
        self.comet_project = None
        
    def setup_comet(self) -> bool:
        """
        Configura CometML con las credenciales proporcionadas
        """
        if not self.comet_api_key:
            self.comet_api_key = os.environ.get("COMET_API_KEY")
            if not self.comet_api_key:
                print("Error: El API KEY de CometML introducido como variable del entorno no es válido.")
                self.comet_api_key = input("Ingresa tu API key de CometML manualmente: ").strip()
                return False
                
        if not self.comet_project:
            self.comet_project = self.config.get("project", "yolo_project")
            if not self.comet_project:
                print("Error: El project de CometML introducido en el config no es válido.")
                self.comet_project = input("Ingresa el nombre del proyecto en CometML manualmente: ").strip()
                return False
        
        try:
            comet_ml.init(project_name=self.comet_project, api_key=self.comet_api_key)
            print("Conexión exitosa con CometML")
            return True
        except Exception as e:
            print(f"Error al conectar con CometML: {str(e)}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Carga el modelo YOLO
        """
        try:
            self.model = YOLO(model_path)
            print(f"Modelo cargado: {model_path}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return False
        
    def train(self, task: str = "detect") -> None:
        """
        Entrena el modelo con los parámetros configurados
        """
        def log_instances(data_dir: str, experiment: comet_ml.Experiment):

            with open(os.path.join(data_dir, "data.yaml"), "r") as f:
                id_class_mapping = yaml.safe_load(f)["names"]

            classes = defaultdict(lambda: defaultdict(int))
            splits = set()
            for file_path in glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
                split = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                splits.add(split)
                with open(file_path, "r") as f:
                    for line in f.readlines():
                        if line.split(" ")[0] != "None":
                            class_id = int(line.split(" ")[0])
                            classes[id_class_mapping[class_id]][split] += 1
                        else:
                            continue
                        
            splits = list(splits)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "instances.csv")
                with open(temp_file_path, "w") as f:
                    f.write(",".join(["defect", *splits, "total"]) + "\n")
                    for defect, distro in classes.items():
                        f.write(f"{defect}")
                        total = 0
                        for split in splits:
                            f.write(f",{distro[split]}")
                            total += distro[split]
                        f.write(f",{total}\n")

                experiment.log_table(temp_file_path, headers=True)

            return 

        if not self.model:
            print("Error: No se ha cargado ningún modelo")
            return
            
        try:
            # Extraer parámetros de entrenamiento
            train_args = {
                "task": task,
                "data": self.config.get("data", ""),
                "project": self.config.get("project", "yolo_project"),
                "batch": self.config.get("batch", 4),
                "save_json": self.config.get("save_json", True),
                "patience": self.config.get("patience", 20),
                "optimizer": self.config.get("optimizer", "SGD"),
                "augment": self.config.get("augment", False),
                "cos_lr": self.config.get("cos_lr", True),
                "iou": self.config.get("iou", 0.6),
                "lr0": self.config.get("lr0", 0.0001),
                "lrf": self.config.get("lrf", 0.00001),
                "verbose": self.config.get("verbose", True),
                "box": self.config.get("box", 5.5),
                "cls": self.config.get("cls", 5.5),
                "dfl": self.config.get("dfl", 1.5),
                "pretrained": self.config.get("pretrained", True),
                "single_cls": self.config.get("single_cls", False),
                "epochs": self.config.get("epochs", 300),
                "imgsz": self.config.get("imgsz", 640),
                "dropout": self.config.get("dropout", 0.2),
                "device": self.config.get("device", 0),
                "cache": self.config.get("cache", False),
            }
            
            # Agregar classes si está especificado
            if "classes" in self.config:
                train_args["classes"] = self.config["classes"]
            
            # Iniciar entrenamiento
            print("\nIniciando entrenamiento con los siguientes parámetros:")
            for key, value in train_args.items():
                print(f"{key}: {value}")
            
            experiment = comet_ml.Experiment(project_name="nombre-proyecto")
            log_instances(self.config.get("dataset_dir", ""), experiment)
            results = self.model.train(**train_args)
            print("\n¡Entrenamiento completado!")
            
        except Exception as e:
            print(f"\nError durante el entrenamiento: {str(e)}")
            traceback.print_exc()

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error al cargar el archivo de configuración: {str(e)}")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Entrenador de YOLO con integración CometML')
    
    # Argumentos principales
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración YAML')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                       help='Modo de operación (default: train)')
    
    # Argumentos de CometML
    parser.add_argument('--comet-key', type=str, help='API key de CometML')
    parser.add_argument('--comet-project', type=str, help='Nombre del proyecto en CometML')
    
    # Argumentos de entrenamiento
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Ruta al modelo pre-entrenado')
    parser.add_argument('--data', type=str, help='Ruta al archivo de datos')
    parser.add_argument('--epochs', type=int, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, help='Tamaño del batch')
    parser.add_argument('--imgsz', type=int, help='Tamaño de la imagen')
    parser.add_argument('--device', type=int, default=0, help='Dispositivo GPU')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    try:
        # Cargar configuración
        config = None
        if args.config:
            config = load_yaml_config(args.config)
        else:
            # Usar argumentos CLI
            config = {
                "model": args.model,
                "data": args.data,
                "epochs": args.epochs,
                "batch": args.batch_size,
                "imgsz": args.imgsz,
                "device": args.device
            }
            
        # Inicializar trainer
        trainer = YOLOTrainer(config)
        
        # Configurar CometML
        trainer.comet_api_key = args.comet_key
        trainer.comet_project = args.comet_project
        if not trainer.setup_comet():
            return
        
        # Cargar modelo
        if not trainer.load_model(config.get("model", "yolov8n.pt")):
            return
        
        # Ejecutar modo seleccionado
        if args.mode == 'train':
            trainer.train()
        elif args.mode == 'predict':
            image_path = input("Ingresa la ruta de la imagen a predecir: ")
            trainer.predict(image_path)
            
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()