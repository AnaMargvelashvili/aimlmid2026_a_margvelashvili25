# AI and ML for Cybersecurity — Midterm
Student: Ana Margvelashvili  
Repo: aimlmid2026_a_margvelashvili25

## Project structure
- `correlation/` — correlation task code
- `spam/` — spam classifier (train, features, app, plots)
- `data/` — dataset csv
- `images/` — saved plots/screenshots used in this README

---

## 1) Correlation task
Source: max.ge/aiml_midterm/19582_html

### Method
I collected (x, y) coordinates from the interactive plot (blue dots) and computed Pearson correlation coefficient.

### Result
- Pearson r = **TODO**

### Plot
![Correlation scatter](images/correlation_scatter.png)
<img width="1653" height="993" alt="correlation_plot" src="https://github.com/user-attachments/assets/570f04ae-30e9-4e2c-af61-74a323ac3456" />

Code: `correlation/correlation.py`

---

## 2) Spam email detection (Logistic Regression)
Dataset: `data/` (csv uploaded to this repo)

### Train/test split
70% train / 30% test

### Result
- Accuracy = **TODO**
- Confusion matrix = **TODO**

### Confusion Matrix
The model correctly classifies most spam and non-spam messages.  
True Positives and True Negatives dominate the matrix, resulting in high overall accuracy (~95%).


![Confusion Matrix](images/confusion_matrix.png)

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/2b902ecd-5e6c-47c3-a3d0-fe7d36a73419" />

### Visualizations
1) Class distribution  
![Class distribution](images/class_distribution.png)

<img width="640" height="480" alt="class_distribution" src="https://github.com/user-attachments/assets/9dac2ad8-b267-4cf2-8f38-b2b5b913cce8" />


3) Top coefficients  
![Top coefficients](images/top_coefficients.png)
<img width="640" height="480" alt="top_coefficients" src="https://github.com/user-attachments/assets/a08e2273-dbf7-4a94-be95-9ae66dd11430" />



Code: `spam/train.py`, `spam/plots.py`, `spam/features.py`, `spam/app.py`

---

## How to run
```bash
pip install -r requirements.txt
python correlation/correlation.py
python spam/train.py
python spam/plots.py
python spam/app.py
