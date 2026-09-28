# **AlzheimerClassifierTL: ResNet50-Based Alzheimer’s MRI Classifier**

AlzheimerClassifierTL applies **transfer learning with ResNet50** to classify **brain MRI scans** into four cognitive health categories:  
🟢 **NonDemented** • 🟡 **Very Mild Demented** • 🟠 **Mild Demented** • 🔴 **Moderate Demented**

This project delivers a complete **computer vision pipeline**—from **data exploration and augmentation** to **model training, evaluation, and real-time deployment** using **Streamlit** and **Hugging Face**.

---

## **Demo**
![App Demo](./Alzheimer%20Detection%20App.png)  
![No Dementia Example](./No%20Dementia.png)  
![Very Mild Dementia Example](./Very%20Mild%20Dementia.png)  
![Mild Dementia Example](./Mild%20Dementia.png)  
![Moderate Dementia Example](./Moderate%20Dementia.png)

---

## **Objective**

Develop and deploy a robust deep learning model to support **early detection of Alzheimer’s disease**, a progressive neurological disorder affecting **memory and cognitive function**.  
Accurate staging of dementia enables timely medical consultation and care.

---

## **Dataset**

- **Source**: [Kaggle – Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data)  
- **Preparation**:  
  - Images resized to **224×224 (RGB)**  
  - Normalized with **ResNet50 preprocess_input**  
  - Balanced **train/validation/test** splits  

---

## **Modeling Approach**

- **Base Network**: **ResNet50** pre-trained on ImageNet  
- **Custom Layers**: Global max pooling, batch normalization, dense layer (256 units, L1/L2 regularization), dropout (0.45)  
- **Output Layer**: Softmax activation for 4-class prediction  
- **Training Setup**:  
  - Loss: **Categorical Crossentropy**  
  - Optimizer: **Adamax** (learning rate 0.001)  
  - Augmentation: rotation, width/height shift, zoom (training set only)  
  - Callbacks: **EarlyStopping** & **ModelCheckpoint** to prevent overfitting  

---

## **Performance**

Final model results on the test set:  
- **Accuracy**: `98.3%`  
- **Precision**: `98.9%`  
- **Recall**: `93.8%`  
- **F1 Score**: `96.1%`

---

## **Deployment**

The trained model powers an interactive **Streamlit web app** and a **Hugging Face Space** for real-time predictions.  
Users can upload an MRI image and receive **instant, color-coded stage classification**.

---

## **Tech Stack**

**Languages & Libraries**  
- Python, Pandas, NumPy  
- TensorFlow / Keras, scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  

**Techniques**  
- Transfer Learning (**ResNet50 fine-tuning**)  
- Data Augmentation (rotation, shift, zoom)  
- EarlyStopping & ModelCheckpoint  
- Real-time deployment with **Streamlit** & **Hugging Face**
