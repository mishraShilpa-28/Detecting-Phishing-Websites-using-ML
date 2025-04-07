
# 🛡️ Phishytics - Intelligent Phishing Website Detection with Machine Learning

Phishytics is a machine learning-powered tool to detect phishing websites using a combination of **Byte Pair Encoding (BPE)** and **TF-IDF** features. The model uses a **Random Forest Classifier** to deliver high accuracy in phishing detection. Pre-trained models are available for quick deployment.

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)

---

## 🧠 Blog Reference
Based on the blog post [Phishytics – Machine Learning for Detecting Phishing Websites](https://faizanahmad.tech/blog/2020/02/phishytics-machine-learning-for-phishing-websites-detection/), this repository contains complete source code and pre-trained models.

---

## 📁 Project Structure

| Path | Description |
| :--- | :---------- |
| `phishytics-machine-learning-for-phishing` | Main folder |
| └── `tokenizer/` | Stores tokenizer output files |
| └── `saved_models/` | Folder to store your trained models |
| └── `pretrained_models/` | Contains pre-trained model files |
| └── `labeled_data/` | Contains phishing and legitimate HTML pages |
| &emsp;&emsp; ├── `phishing_htmls/` | Phishing web page HTMLs |
| &emsp;&emsp; └── `legitimate_htmls/` | Legitimate web page HTMLs |
| `create_data_for_tokenization.py` | Creates tokenized data using BPE |
| `train_phishing_detection_model.py` | Trains a phishing detection model |
| `test_model.py` | Tests a URL using a custom trained model |
| `test_pretrained_model.py` | Tests a URL using a pre-trained model |

---

## 🧰 Requirements

Install the following Python packages before training or testing:

- `scikit-learn`
- `numpy`
- `tokenizers`
- `langdetect`
- `joblib`
- `requests`

```bash
pip install scikit-learn numpy tokenizers langdetect joblib requests
```

---

## 🚀 Getting Started

### 1. Tokenize Data

Prepare the dataset using BPE:

```bash
python3 create_data_for_tokenization.py --labeled_data_folder labeled_data --vocab_size 300 --min_frequency 3
```

**Parameters**:
- `--labeled_data_folder`: Folder with phishing and legit HTMLs
- `--vocab_size`: Maximum vocabulary size
- `--min_frequency`: Ignore tokens with frequency below this

---

### 2. Train the Model

Train a phishing detection model using TF-IDF over tokenized data:

```bash
python3 train_phishing_detection_model.py --tokenizer_folder tokenizer/ --labeled_data_folder labeled_data/ --ignore_other_languages 1 --apply_different_thresholds 1 --save_model_dir saved_models
```

**Parameters**:
- `--tokenizer_folder`: Path to tokenizer files
- `--labeled_data_folder`: HTML data folder
- `--ignore_other_languages`: 1 = Only English
- `--apply_different_thresholds`: 1 = Evaluate with multiple thresholds
- `--save_model_dir`: Where to save trained models

---

### 3. Test a Website (Custom Model)

Test a URL using your trained model:

```bash
python3 test_model.py --tokenizer_folder tokenizer --threshold 0.5 --model_dir saved_models --website_to_test https://www.google.com
```

---

## 🧪 Using the Pre-trained Model

1. Unzip the file `document-frequency-dictionary.zip` inside the `pretrained_models` folder (do **not** move it).

2. Run the script:

```bash
python3 test_pretrained_model.py --tokenizer_folder pretrained_models --threshold 0.5 --model_dir pretrained_models --website_to_test https://www.google.com
```

---

## 📝 Notes

- Always include `http://` or `https://` in the test URL to avoid errors.
- Pre-trained model offers **99% test accuracy**.
- Ideal for companies, educators, and cybersecurity enthusiasts.

---

## 📌 License

MIT License © Shilpa Mishra

---

## 🙌 Acknowledgements

Thanks to the open-source community and Hugging Face for their tokenizer library.
