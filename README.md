# 🌸 Iris Species Predictor with Deep Learning and Streamlit

A lightweight deep learning application to classify Iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on their sepal and petal measurements. This project includes a trained neural network model and a deployable Streamlit app.

---

## 📊 Dataset

I used the classic [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris), consisting of:

- **150 samples**
  
- **4 features**:
  
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **3 target classes**:
  
  - *Iris Setosa*
  - *Iris Versicolor*
  - *Iris Virginica*

## 🧠 Deep Learning Model

Built a simple yet effective neural network using **TensorFlow/Keras**, featuring:

- **Input Layer**: 4 neurons
  
- **Hidden Layers**: 
  - Dense (10 neurons) + ReLU
  - Dense (8 neurons) + ReLU
    
- **Output Layer**: 3 neurons + Softmax

### ✅ Accuracy Achieved: `98%+` on test set


## 🚀 Streamlit App

An interactive web app built with **Streamlit**, where users can:

- Adjust sepal/petal dimensions using sliders
- Get real-time species prediction
- View model confidence scores


## 🔧 How to Run

### ▶️ Run Locally

```bash
git clone https://github.com/your-username/iris-species-predictor.git
cd iris-species-predictor
streamlit run iris_streamlit_app.py
