# Pneumonia X-ray — XAI Dashboard (research demo)

- ResNet-50 classifier with calibration, TTA, ROC/ECE, bootstrap CIs
- XAI (Grad-CAM + IG) + lung-focus sanity check
- Operates at 90% specificity; PPV/NPV vs prevalence

**Not for clinical use.**

## Run locally
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

