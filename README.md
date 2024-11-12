# YOLO HF Trainer

Sistema automatizado para entrenamiento de modelos YOLO utilizando datasets de HuggingFace.

## 📋 Descripción
Esta herramienta permite entrenar modelos YOLO de manera sencilla utilizando datasets almacenados en HuggingFace, incluyendo datasets privados si se dispone de las credenciales necesarias. Diseñada específicamente para usuarios sin experiencia técnica en IA.

## 🚀 Inicio Rápido

### Prerrequisitos
- Python 3.8 o superior
- Cuenta en HuggingFace con API key
- Acceso al dataset que se desea utilizar

### Instalación

1. Clonar el repositorio:
```bash
git clone <URL-del-repositorio>
cd yolo-hf-trainer
```

2. Ejecutar script de instalación:
```bash
bash setup.sh
```

3. Configurar credenciales:
- Copiar el archivo `config/credentials_template.json` a `config/credentials.json`
- Añadir tu API key de HuggingFace en el archivo

## 💻 Uso

### Entrenamiento Básico
```bash
python scripts/train_model.py --dataset nombre_dataset
```

### Opciones de Configuración
- `--dataset`: Nombre del dataset en HuggingFace
- `--epochs`: Número de épocas de entrenamiento (por defecto: 100)
- `--batch_size`: Tamaño del batch (por defecto: 16)
- `--img_size`: Tamaño de imagen (por defecto: 640)

## 📁 Estructura del Proyecto
```
proyecto/
├── config/              # Archivos de configuración
├── scripts/             # Scripts de entrenamiento
├── src/                 # Código fuente
├── requirements.txt     # Dependencias
└── README.md           # Este archivo
```

## 🛠️ Funcionalidades Principales
- Carga automática de datasets desde HuggingFace
- Soporte para datasets privados
- Configuración simplificada del entrenamiento
- Visualización de resultados
- Exportación automática del modelo entrenado

## 📊 Visualización de Resultados
Los resultados del entrenamiento se guardarán en la carpeta `runs/train`, incluyendo:
- Gráficas de pérdida
- Métricas de evaluación
- Ejemplos de detección

## ❗ Solución de Problemas Comunes

### Error de Acceso al Dataset
Si aparece un error de acceso al dataset, verificar:
1. API key correctamente configurada
2. Permisos de acceso al dataset en HuggingFace
3. Conexión a internet estable

### Problemas de Memoria
Si aparecen errores de memoria:
1. Reducir el tamaño del batch
2. Reducir el tamaño de imagen
3. Utilizar un dataset más pequeño para pruebas

## 📫 Soporte
Para problemas o consultas:
1. Abrir un issue en el repositorio
2. Contactar al equipo de soporte

## 🔄 Actualizaciones
Para obtener la última versión:
```bash
git pull origin main
```

## 📝 Notas Importantes
- Asegurarse de tener suficiente espacio en disco
- Recomendado ejecutar en un entorno con GPU
- Hacer backup de los modelos entrenados importantes

## 📜 Licencia
Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.
