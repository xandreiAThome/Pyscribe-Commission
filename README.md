# Pyscribe---Commission

Pyscribe: A Python-TrOCR driven Prototype Device for Scanning Antibiotic Prescriptions

### This is the code for the Research Project Commission

## How to run

```bash
docker build -t my-python-app .
```

### For Windows

```bash
docker run --rm -it --device=/dev/video0 -v "%cd%:/app" my-python-app
```

### For Linux

```bash
docker run --rm -it --device=/dev/video0 -v "$(pwd):/app" my-python-app
```

## Put the trocr model folder in the app folder
