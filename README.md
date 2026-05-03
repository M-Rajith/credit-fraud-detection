# Credit Card Fraud Detection System
### Hybrid BiLSTM + Isolation Forest | Real Sparkov Transaction Data

## Results Summary
| Model | ROC-AUC | Recall | F1 | Accuracy |
|-------|---------|--------|-----|---------|
| BiLSTM | 0.9981 | 98.46% | 0.4895 | 98.16% |
| Isolation Forest | 0.5252 | 12.50% | 0.0173 | 87.24% |
| Hybrid | 0.9949 | 97.70% | 0.5336 | 98.47% |
| User-Level | - | 100.0% | 0.9972 | - |

## Dataset
Download sparkov.csv from: https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Setup
pip install -r requirements.txt
python main.py

## Web Interface
python app.py
Open http://localhost:5000

## License
MIT License
