# Yuvika-Gupta_Girl-Hackathon_SWE_2025
QUBITMED:

# Project idea: 
My project, QubitMed aims to revolutionize disease diagnosis by providing healthcare professionals with a powerful quantum
diagnostic assistant that can analyze medical images, patient history by looking at different reports and genetic data. It uses Quantum Neural Networks (QNN) which help the system diagnose early-stage, rare and even multiple diseases by detecting
intricate patterns that the classical models are often unable to capture. The main quantum advantage lies in the parallel
computing which allows the processing of the complex and large-scale medical data simultaneously (as quantum systems handle
multiple calculations and analyses at the same time) which speeds up the diagnosis by a huge amount. The project is also making
use of Google's TensorFlow quantum library which helps the system process massive amounts of patient data with unparalleled
speed and accuracy, generating personalized treatment plan recommendations based on the quantum- enhanced analysis of
similar cases. This project offers the healthcare sector

# Phase 1 Implementation:
This project explores the application of quantum computing in medical diagnosis by developing a hybrid quantum-classical neural network (QNN). Using the MIMIC-III dataset, the model integrates classical machine learning with quantum computing to predict patient outcomes, diagnosing medical conditions from intensive care data. The project uses TensorFlow Quantum (TFQ) and Cirq for quantum circuit simulation, enabling the representation of complex relationships in the data that classical models may struggle to capture.

## Tech Stack
- **TensorFlow 2.15**
- **TensorFlow Quantum**
- **Cirq** (Quantum Circuit Simulation)
- **SymPy** (Symbolic Mathematics)
- **Scikit-Learn** (Preprocessing & Evaluation)
- **Pandas & NumPy** (Data Handling)
- **Kaggle API** (Dataset Retrieval)


# Dataset
MIMIC-III (Medical Information Mart for Intensive Care III)
MIMIC-III is a comprehensive, de-identified dataset comprising health-related data from patients admitted to critical care units. It contains information such as demographic data, vital signs, laboratory measurements, procedures, medications, and diagnosis codes (ICD-9).

Medical Records: Over 60,000 intensive care unit (ICU) stays for more than 40,000 patients.
ICD-9 Codes: The primary source of diagnostic labels used to train the QNN. These codes represent various medical conditions and procedures, which the model aims to predict based on patient data.
Vital Signs and Lab Results: Used as input features in the quantum model, helping predict outcomes based on the quantum-enhanced learning.
- **Source:** MIMIC-III (10,000 patients)
- **Files Used:**
  - `DIAGNOSES_ICD_random.csv` (ICD Codes for Diagnoses)
  - `D_ICD_DIAGNOSES.csv` (ICD Code Descriptions)
  - `PATIENTS_random.csv` (Patient Demographics)
  - `ADMISSIONS_random.csv` (Hospital Admissions Data)



# Theoretical Background:
# Quantum Computing in Machine Learning
Quantum computing introduces fundamentally new ways to process information through qubits, which can exist in superpositions of states (|0⟩ and |1⟩) and can be entangled with each other. This allows quantum systems to represent and compute over vast amounts of information simultaneously, opening up new possibilities for solving problems intractable for classical systems.

# Superposition: 
A qubit can be in a state that is a linear combination of |0⟩ and |1⟩, providing the capacity to encode multiple states at once. This property allows the quantum model to process complex medical data in ways classical systems cannot.

# Entanglement: 
This phenomenon allows qubits to be correlated in such a way that the state of one qubit is dependent on the state of another, even when far apart. In our model, entanglement helps capture intricate correlations in medical data, such as relationships between different medical conditions.

# Quantum Parallelism: 
Quantum systems can compute many possibilities in parallel, leading to potential speed-ups for machine learning tasks. By encoding medical features into qubits, the quantum model can explore a much larger solution space efficiently, especially for data-intensive problems like those found in medical diagnostics.

# Hybrid Quantum-Classical Neural Networks (QNN)
A Quantum Neural Network (QNN) combines classical deep learning layers with quantum circuits to solve specific machine learning tasks. The classical part processes structured data and extracts higher-level features, while the quantum circuit enhances the model’s ability to represent and learn from more complex, high-dimensional data.

In this project, the Parameterized Quantum Circuit (PQC) is integrated into the neural network as a layer that applies quantum gates to the qubits. These gates, with tunable parameters, act as learnable components of the model. The classical optimization algorithms (such as gradient descent) update the parameters of both the classical neural network and the quantum circuit.

# Quantum Circuit Design:

-The quantum circuit uses Hadamard gates to create superposition states, allowing the quantum model to explore a wide range of potential data representations.
-Controlled-NOT (CNOT) gates are used to introduce entanglement, capturing non-linear correlations between medical features (e.g., relationships between diagnoses and physiological measurements).
Rotation gates (Rx, Ry, Rz) parameterize the quantum circuit, with the angles being adjusted during training to optimize the performance of the quantum model.
-The key idea is that the quantum circuit acts as a feature extractor, learning to map medical data into quantum states in such a way that the classical neural network can use the extracted information to make more accurate predictions.

# TensorFlow Quantum and Cirq
TensorFlow Quantum (TFQ) is a framework made by Google that integrates quantum computing with classical machine learning, enabling the construction of hybrid models. Cirq, a library for creating, simulating, and optimizing quantum circuits, is used here to design and run the quantum circuits in our hybrid neural network.

The classical part of the neural network is constructed using TensorFlow, while the quantum circuits (PQC) are built and simulated using Cirq. Together, they form a pipeline where classical and quantum layers cooperate to enhance the model’s diagnostic ability.

# Project Components
#Preprocessing: The MIMIC-III dataset undergoes extensive preprocessing. Features are selected and scaled before being encoded into quantum states. A large part of this involves one-hot encoding of the ICD-9 diagnosis codes and preparing continuous medical features for quantum encoding.

# Quantum Circuit:

-Qubit Representation: Each input feature from the dataset is mapped to a qubit. For instance, vital signs (like heart rate, blood pressure, etc.) are encoded as quantum states.
-Quantum Gates: The qubits undergo transformation through quantum gates. These gates (Rx, Ry, Rz) are parameterized, and their angles are updated during the model’s training.
# Hybrid Quantum-Classical Architecture:
Quantum Layer: The PQC transforms the quantum-encoded data and produces a quantum state output.
Classical Layer: The output of the quantum circuit is fed into a classical neural network for further processing and prediction. This layer contains fully connected (dense) layers that map the quantum features to diagnostic predictions.
# Training: 
The hybrid model is trained using a hybrid quantum-classical gradient descent method, optimizing both quantum gate parameters and classical neural network weights. The loss function evaluates the difference between predicted and actual medical diagnoses.
# Prediction: 
The final output of the hybrid QNN is a probability distribution over possible diagnoses. The model predicts the most likely diagnosis for the patient based on their medical data.

## Features
- **Quantum Neural Networks (QNNs):** Improves disease prediction accuracy using quantum circuits.
- **Data Preprocessing:** Standardization, encoding, and train-test splits using Scikit-Learn.
- **Automated Dataset Retrieval:** Uses Kaggle API to fetch and prepare data.
- **Medical Diagnosis Prediction:** Leverages QNNs to analyze patient history for more accurate diagnostics.

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YuvikaGupta875/Yuvika-Gupta_Girl-Hackathon_SWE_2025.git
   cd Yuvika-Gupta_Girl-Hackathon_SWE_2025
   ```
2. **Install dependencies:**
   ```bash
   pip install tensorflow==2.15.0 tensorflow-quantum scikit-learn numpy pandas cirq sympy
   ```
3. **Download the dataset:**
   - Create a Kaggle account & API key
   - Upload `kaggle.json` to the working directory
   - Run:
     ```bash
     !mkdir -p ~/.kaggle
     !mv kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     !kaggle datasets download -d bilal1907/mimic-iii-10k
     !unzip mimic-iii-10k.zip
     ```

## Running the Model
1. **Load & preprocess data**
2. **Train the QNN model**
3. **Evaluate & visualize results**

## Contributors
- **Yuvika Gupta**

## Future Improvements
- **Enhancing Quantum Model Performance**
- **Expanding Dataset for More Robust Training**
- **Integrating with Real-World Medical Applications**
