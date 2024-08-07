Quantum neural networks (QNNs) process input using qubits, the fundamental components of quantum information that can exist in superpositions of states, as opposed to classical neural networks, which only use classical bits. This unique property of QNNs may allow them to traverse a huge solution space faster than their classical counterparts, perhaps leading to exponential speedups for specific computational tasks. Reducing the error between the target and actual values is the main objective of the QNN's parameter optimization process. High precision dehumidification forecasts can be modelled using quantum-twisting deep learning and machine learning algorithms. A Quantum Neural Network (QNN) model combines the concepts of neural networks with quantum computing methods. It is mostly reliant on qubit neurons, which create a link between quantum states and neural states. Neurons modify their states through computational interactions based on quantum logic gates.

### **1. Data Collection and Preparation**

Due to the privacy of the data, I'm unable to share the dataset directly. The project began with collecting a comprehensive dataset that captures various features pertinent to the dehumidification process. This dataset was crucial for building a predictive model. Data preparation involved several key tasks:

- **Data Collection:** Gathering raw data from relevant sources, ensuring it includes all necessary variables such as temperature, humidity, and any other relevant metrics.
  
However, here is a detailed description of the dataset attributes:

- **δd (mm)**: Represents the thickness of the desiccant layer used in the system, measured in millimeters. This variable affects the efficiency of the dehumidification process.
  
- **L (mm)**: Denotes the length of the desiccant layer in millimeters. It is crucial for understanding the surface area available for moisture absorption.

- **d (mm)**: The diameter of the desiccant particles in millimeters, which influences the overall performance of the dehumidification process.

- **x (mm)**: A parameter representing the distance or spacing within the system, measured in millimeters, which can impact airflow and system efficiency.

- **Twi (℃)**: The inlet temperature of the air stream, measured in degrees Celsius. This variable affects the capacity of the desiccant to absorb moisture.

- **Tai (℃)**: The ambient air temperature surrounding the system, measured in degrees Celsius, which influences the system's performance.

- **Wai (g/kg)**: The humidity of the inlet air stream, measured in grams per kilogram. This is a key factor in determining how much moisture needs to be removed.

- **ṁa (kg/s)**: The mass flow rate of the air stream, measured in kilograms per second. This variable affects the rate at which air is processed through the system.

- **tc (s)**: The cycle time or operational duration of the system, measured in seconds. This parameter is important for understanding the system's operation over time.

- **Two (℃)**: The temperature of the air stream exiting the system, measured in degrees Celsius. It helps in assessing the cooling or heating effect of the dehumidification process.

- **Tao (℃)**: The ambient temperature at the outlet, measured in degrees Celsius, which can provide additional context on the system's thermal performance.

- **Wao (g/kg)**: The humidity of the air stream exiting the system, measured in grams per kilogram. This value is crucial for evaluating the effectiveness of the dehumidification process.

These attributes provide comprehensive information about the operational conditions and performance of the dehumidification system, allowing for in-depth analysis and optimization.

- **Data Cleaning:** Addressing issues like missing values, outliers, and inconsistencies in the dataset. This step is vital to ensure that the model is trained on high-quality data. Techniques such as imputation for missing values and filtering outliers were applied.


- **Feature Selection:** Identifying and selecting the most important features that significantly influence the prediction outcomes. This process helps in reducing the dimensionality of the data and focusing on the most relevant inputs.

### **2. Data Preprocessing**


Data preprocessing prepares the dataset for training the model by transforming and normalizing the data:

- **Normalization:** Scaling the features to a standard range, often using techniques like z-score normalization or Min-Max scaling. This process ensures that all features contribute equally to the model's learning process and helps in achieving better convergence during training.

- **Data Splitting:** Dividing the dataset into training and testing subsets. The training set is used to train the model, while the testing set is reserved for evaluating the model's performance. This split helps in assessing the model’s ability to generalize to new, unseen data.

### **3. Quantum Layer Setup**


This step focuses on integrating quantum computing elements into the model:

- **Quantum Circuit Design:** Constructing a quantum circuit with a specific number of qubits. In this project, three qubits were used to match the number of input features. The design of the circuit involves defining how quantum gates will manipulate the qubits.

- **AngleEmbedding:** Encoding classical data into quantum states by applying angle parameters to the qubits. This step transforms classical input features into quantum representations that can be processed by quantum algorithms.

- **StronglyEntanglingLayers:** Implementing quantum gates that create entanglement between qubits. This process introduces complex quantum interactions that enhance the model's capacity to learn intricate patterns and relationships in the data.
- 
The QNN integrates both quantum and classical computations to enhance the prediction accuracy by leveraging quantum entanglement and superposition principles.
The QNN model uses the following structure:
1.	Input Layer: Takes the input parameters  
2.	Dense Layer: A classical dense layer with ReLU activation function to process the input features.
3.	Quantum Layer: A quantum layer using the quantum node (QNode) which embeds the input features into quantum states using angle embedding and applies strongly entangling layers to these quantum states.
4.	Output Layer: A classical dense layer to generate the final predicted output  .
The relationship between the input and output can be expressed as:



### **4. Model Construction**


Constructing the Quantum Neural Network (QNN) involves combining classical and quantum components:

- **Classical Neural Network Layers:** Adding dense layers that process the output from the quantum circuit. These layers are responsible for further processing and learning from the quantum-enhanced features.

- **Quantum Layers:** Integrating the quantum components, such as `AngleEmbedding` and `StronglyEntanglingLayers`, into the model. These layers handle the quantum data processing and contribute to the overall model's predictive power.

- **Custom Loss Function:** Defining a Root Mean Squared Error (RMSE) loss function tailored to evaluate the model’s accuracy. RMSE measures the average magnitude of prediction errors, focusing on minimizing discrepancies between predicted and actual values.

### **5. Model Compilation**


Compiling the model involves configuring the training process:

- **Optimizer Selection:** Choosing an optimization algorithm (e.g., Adam, SGD) that adjusts the model’s weights based on the loss function. The optimizer plays a crucial role in improving the model's performance during training.

- **Loss Function Integration:** Incorporating the custom RMSE loss function into the model. This loss function guides the optimization process by penalizing larger prediction errors more heavily.

### **6. Model Training**


Training the model involves fitting it to the training data:

- **Training Process:** Iterating through the dataset multiple times (epochs) to adjust the model’s parameters. During each iteration, the model learns from the data by updating its weights to minimize the loss function.

- **Monitoring Progress:** Tracking the model’s performance on the training data to ensure it is learning effectively. Metrics like training loss and accuracy are monitored to gauge progress.

### **7. Model Evaluation**


Evaluating the model involves assessing its performance on unseen data:

- **Testing Data:** Using the reserved testing set to evaluate the model’s generalization ability. This step checks how well the model performs on new data it has not seen during training.

- **Evaluation Metrics:** Measuring the model’s performance using metrics such as RMSE. These metrics provide insights into the model’s accuracy and help identify any potential improvements.

### **8. Predictions and Results Analysis**


This step focuses on interpreting the model’s outputs:

- **Making Predictions:** Using the trained model to predict outcomes on new or test data. The predictions are generated based on the patterns learned during training.

- **Results Analysis:** Comparing predicted values to actual values to assess accuracy. Visualization techniques, such as plotting predicted vs. actual values, help in understanding how well the model performs.


![1](https://github.com/user-attachments/assets/6b00f46d-3bfc-4900-968c-e1d607dd65ef)

![2 1](https://github.com/user-attachments/assets/8b35f4ae-14ea-43e8-a771-f21c84916a7e)

![download (16)](https://github.com/user-attachments/assets/727da1d7-6de6-4dc4-bcf2-03fac44c15eb)

### **9. Model Optimization**


Optimizing the model involves refining it for better performance:

- **Hyperparameter Tuning:** Adjusting various parameters of the model, such as learning rate or number of epochs, to enhance performance. Techniques like grid search or random search may be used for this purpose.

- **Model Refinement:** Modifying the quantum circuit design or neural network architecture based on evaluation results. This could involve changing the number of qubits or layers to improve the model’s predictive capabilities.

- ![download (17)](https://github.com/user-attachments/assets/08beccec-96b3-42ce-b840-1bb58a679844)

-![2 2](https://github.com/user-attachments/assets/702258ab-1edb-404e-80c4-5044c45f7689)

![2 3](https://github.com/user-attachments/assets/86bdfb77-67ed-4c18-a6e6-bfb115f32324)

