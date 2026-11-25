# Smart Irrigation App - upgraded version

## How to install (drop-in replacement)
1. Keep your existing `model/irrigation_model.pkl` and `irrigation_prediction.csv.xlsx` in the project root.
2. Delete the old `app.py`, `pages/1_Prediction.py`, `pages/2_Analytics.py`.
3. Copy these into the project root: `Home.py`, `utils.py`, `test_utils.py`, `requirements.txt`, `.streamlit/config.toml`
   and the whole `pages/` folder (it now has 3 pages).
4. Install: `pip install -r requirements.txt`
5. Run: `streamlit run Home.py`
6. Test the logic: `python -m unittest test_utils -v`

## Folder layout
```
project/
  Home.py
  utils.py
  test_utils.py
  requirements.txt
  irrigation_prediction.csv.xlsx
  model/irrigation_model.pkl
  .streamlit/config.toml
  pages/1_Prediction.py
  pages/2_Analytics.py
  pages/3_Model_Performance.py
```
