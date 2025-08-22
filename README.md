# IMDB Reviews Sentiment Analysis

A machine learning project that performs sentiment analysis on IMDB movie reviews to classify them as positive or negative. This project demonstrates the complete pipeline from data preprocessing to model deployment for natural language processing tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements sentiment analysis on the IMDB Movie Reviews dataset using machine learning techniques. The goal is to automatically classify movie reviews as positive or negative based on the text content. The project explores various approaches including traditional machine learning algorithms and deep learning models to achieve optimal performance.

**Key Objectives:**
- Build an accurate sentiment classifier for movie reviews
- Compare different machine learning approaches
- Provide a user-friendly interface for sentiment prediction
- Demonstrate end-to-end ML pipeline implementation

## 📊 Dataset

The project uses the **IMDB Movie Reviews Dataset**, which contains:
- **50,000** movie reviews (25,000 positive, 25,000 negative)
- **Binary classification** problem (positive/negative sentiment)
- **Balanced dataset** ensuring equal representation
- **Text preprocessing** required for optimal model performance

**Data Source:** The dataset is typically loaded from Keras datasets or can be downloaded from Stanford AI Lab.

## ✨ Features

- **Data Preprocessing:** Text cleaning, tokenization, and sequence padding
- **Multiple Models:** Implementation of various ML algorithms
- **Performance Evaluation:** Comprehensive metrics and visualization
- **Prediction Interface:** Easy-to-use function for new review classification
- **Model Comparison:** Side-by-side comparison of different approaches
- **Visualization:** Training history and performance plots

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Aadya105/IMDB-Reviews-Sentiment-Analysis.git
cd IMDB-Reviews-Sentiment-Analysis
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

### Required Libraries
```
tensorflow>=2.0.0
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
wordcloud
jupyter
```


### Key Insights
- **BiLSTM** achieved the highest accuracy at 89.2%
- **Deep learning models** outperformed traditional ML approaches
- **Text preprocessing** significantly improved model performance
- **Word embeddings** enhanced feature representation

## 🛠️ Technologies Used

- **Python** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning library
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Development environment


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Contact

If you have any questions, suggestions, or feedback, please feel free to reach out:

- **GitHub:** [@Aadya105](https://github.com/Aadya105)


---

**⭐ If you found this project helpful, please give it a star! ⭐**
