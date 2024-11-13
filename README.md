# YOLO HF Trainer

Sistema automatizado para entrenamiento de modelos YOLO utilizando datasets de HuggingFace.

## 📋 Descripción
Esta herramienta permite entrenar modelos YOLO de manera sencilla utilizando datasets almacenados en HuggingFace, incluyendo datasets privados si se dispone de las credenciales necesarias. Diseñada específicamente para usuarios sin experiencia técnica en IA.

## 🚀 Inicio Rápido

### Prerrequisitos
- Python 3.9 o superior
- Cuenta en HuggingFace con API key
- Cuenta en CometML con API key
- Acceso al dataset que se desea utilizar

### Instalación

1. Clonar el repositorio:
```bash
git clone <URL-del-repositorio>
cd yolo-hf-trainer
```

2. Ejecutar instalación del entorno utilizando el fichero environment.yml
```bash
conda env create --name envname --file=environment.yml
```

2.1 (Si el punto 2 da problemas) Ejecutar instalación de requirements.txt en tu entorno conda utilizando PIP:
```bash
pip install -r requirements.txt
```

3. Configurar credenciales (❗WORK IN PROGRESS❗):
- Modificar `credentials.json` para cada usuario personalmente
- Añadir tu API key de HuggingFace en el archivo
- Añadir tu API key de CometML en el archivo

## 💻 Uso

### Carga del dataset
```bash
python load_and_process.py
```

### Entrenamiento Básico
```bash
python yolo_trainer.py --config yolo_config.yaml --comet-key your_api_key --comet-project your_cometml_project_name
```

## 🛠️ Funcionalidades Principales
- Carga automática de datasets desde HuggingFace
- Soporte para datasets privados
- Configuración simplificada del entrenamiento
- Visualización de resultados
- Exportación automática del modelo entrenado
- Monitorización de entrenamiento via CometML

## 📊 Visualización de Resultados
Los resultados del entrenamiento se guardarán en la carpeta `runs/train`, incluyendo:
- Gráficas de pérdida
- Métricas de evaluación
- Matrices de confusión
- Ejemplos de detección

## ❗ Solución de Problemas Comunes
Están configurados los mensajes para los errores más comunes que puedan ocurrir.

### Error de Acceso al Dataset
Si aparece un error de acceso al dataset, verificar:
1. API key correctamente configurada
2. Permisos de acceso al dataset en HuggingFace
3. Nombre de dataset introducido incorrecto

### Problemas de Memoria
Si aparecen errores de memoria:
1. Reducir el tamaño del batch
2. Reducir el tamaño de imagen
3. Reducir el tamaño de modelo
4. Volver a iniciar el servidor con "Stop Server" en JupyterHun

## 📫 Soporte
Para problemas o consultas:
1. Abrir un issue en el repositorio
2. Contactar al autor

## 🔄 Actualizaciones
Para obtener la última versión:
```bash
git pull origin main
```

## 📝 Notas Importantes
- Asegurarse de tener suficiente espacio en disco
- Recomendado ejecutar en un entorno con GPU, ya que tiempos de entrenamiento e inferenia pueden incrementarse con CPU
- Hacer backup de los modelos entrenados importantes
