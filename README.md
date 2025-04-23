# Pyscribe---Commission

Pyscribe: A Python-TrOCR driven Prototype Device for Scanning Antibiotic Prescriptions

### This is the code for the Research Project Commission

## Setup

1. Create a .env file and input the API key

```
GEMINI_KEY=API_KEY_HERE
```

2. And install the dependencies

```bash
pip install -r requirements.txt
```

3. Change the constant in the app.py

```
TROCR_PATH = "Input trained model path here"
```

> [!NOTE]  
> You may need to install the tkinter python library in linux

```bash
sudo apt-get install python3-tk
```

## How to run

```bash
cd app
python app.py
```
