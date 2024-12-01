# Hi!ckathon 5th Edition 2024: AI & Sustainability 🌲

## Event
<img src="images/image.png" alt="Hi! Paris Logo" width="200"/>
The present code was developed during a contest called 'HI!CKATHON' organized at the request of HEC Paris and Institut Polytechnique de Paris, in the context of the Hi! PARIS Center.
https://github.com/hi-paris/Hickathon5

## Project: Business and Scientific document - Work in progress - NOT FINAL

<img src="images/waterwizards_logo.png" alt="Water Wizards Logo" width="100"/>
**Water Wizards**, Forecasting Water, Fueling Progress


### **I. Overview** 

**Who we are 🤝**

Our start-up, Water Wizards, leverages real-world data to provide critical information to companies and municipalities by predicting the groundwater level, giving them the tools to plan ahead and avoid water shortages.

The issue at hand is to predict the groundwater levels in summer in regions in France. Indeed, due to climate change and increasingly hot summers, it is critical that communities are able to predict the quantities of water available, in order to guarantee access to a reliable and safe water supply, which is essential for agriculture, daily life and the local environment.

Using state-of-the-art machine learning solutions, our team of top-notch engineers built a model to accurately predict groundwater levels in summer, empowering municipalities and companies with precise knowledge they can leverage to balance their needs and uses of water responsibly.

**Meet the team 👋**

|            |                   |      |         |                                           |                                                                                |
| ---------- | ----------------- | ---- | ------- | ----------------------------------------- | ------------------------------------------------------------------------------ |
| First name | Last name         | Year | School  | Skills                                    | Roles and tasks                                                                |
| Yann       | Ntsama            | M2   | Telecom | Data science and computer science expert  | Preprocessing, dataset building, model creation                                |
| Giovanni   | Benedetti da Rosa | M2   | Telecom | Data science and computer science expert  | Git lord, preprocessing, Model Optimization                                    |
| Yassine    | Oj                | M2   | ENSTA   | Data science and Machine learning         | Data preprocessing, Feature engineering, Scientific model                      |
| Cristian   | Chávez            | M2   | Telecom | Data Science and Computer Science expert  | Data preprocessing supervisor, model optimization, feature engineering         |
| Olivier    | de Boissieu       | Y1   | ENSAE   | Project management, communication         | Preprocessing, communication (branding, business model and video presentation) |


Once the team felt confident they had grasped the problem, everyone dug into the dataset individually, communicating through WhatsApp, Discord and source code management tool Github, in order to maximize productivity, while maintaining good communication and discussing face-to-face. 

The team then split up according to their skills, focusing on the AI model itself, and the business solution, through a back-and-forth between the technical team and the business team in order to build a technical and sustainable solution fitting our vision for the future of water management.

### **II. Our business plan**

This insight on water groundwater levels allows municipalities to enact predictive measures allowing for the preservation of water access for all, individuals and companies alike. They also allow companies and particulars using large quantities of water, such as farmers and energy producers, to use water responsibly through planning, since our solution allows them to see ahead, predicting high and low groundwater levels in their region.

Our solution can be deployed in areas suffering from a water crisis, as well as in areas not yet worried by drought, but where water uses are important and which are vulnerable to water shortages. Using data harvested through piezometric and weather stations across the country, the data we provide gives oversight on the groundwater level anywhere, anytime, especially in summer, when water resources come under pressure. It can eventually be deployed anywhere provided there is accurate data on the underground water levels.

Used by corporate and public authorities over water usage, our solution aims to bring an end to water shortages and protect communities from droughts, by being the first start-up to leverage data to predict changes in the groundwater levels.

Our objective is to empower companies and municipalities to manage their water resources sustainably, using our data to preserve and protect the groundwater level.

### **III. Our expertise**

The team used an extensive dataset consisting of approximately 3 million entries and over 100 variables, featuring data harvested across France over a 4-year long span, to construct our AI model.

**Preprocessing ⏮️**

Preprocessing played a crucial role in our project. Initially, we focused on **data exploration**, a process that involved thoroughly examining, understanding, and analyzing the dataset to extract meaningful insights. This step was essential in preparing the data for deeper analysis and model development.

**Feature Engineering 💬**

The dataset contains 136 columns, which correspond to 136 features. Our task was to categorize these features and analyze how each one influences the target variable, water level. Once we completed this task, we shared insights about the different features and classified them into distinct categories:

- **Useful Features:** These features have a clear influence on the target variable and were retained.

- **Redundant Features:** These features provide the same information as another feature. We kept only one of them (for instance, department code and department name).

- **Erroneous Features:** Features containing only NaN values were discarded.

**Data Cleaning 🧹**

Data cleaning involves several important steps. The most repetitive task was eliminating null values from features. The methods used varied depending on the situation. In some cases, null values were replaced with the mean of the existing values, or with the most frequent value. Another aspect of cleaning involved converting data types to ensure consistency across the dataset.

**Data encoding ⭐**\
The dataset contained variables in different formats, so it was necessary to encode them before feeding the data into the model. Our strategy was as follows: we applied one-hot encoding to categorical variables with no more than 5 distinct values, and target encoding for categorical variables with more than 5 values.


**AI Model 🤖 - CatBoost Classifier**

We chose **CatBoost** as our AI model due to its high performance, robustness, and capability to handle categorical features efficiently. CatBoost is a gradient-boosting algorithm that is optimized for speed and accuracy, making it well-suited for tasks involving complex, non-linear data. Unlike other boosting algorithms, CatBoost handles categorical variables natively, without needing preprocessing like one-hot encoding, making it particularly advantageous for datasets with categorical columns. Additionally, CatBoost is less prone to overfitting and provides feature importance, which aids in model interpretability.

**Scalability ⚖️**

CatBoost scales well with larger datasets due to its efficient gradient-boosting algorithm. The model also benefits from its parallelization capabilities, which allow for faster training compared to traditional boosting methods. However, as with most gradient-boosting algorithms, it can still be memory-intensive due to the complexity of the trees and the large number of iterations.

For our use case:

- **Training scalability**: We used **Optuna** to optimize hyperparameters, focusing on parameters like the number of iterations, learning rate, and tree depth to ensure that training remains feasible even with larger datasets. We also explored **early stopping** to reduce unnecessary iterations and avoid overfitting, helping with both scalability and training time.

- **Inference scalability**: Once trained, CatBoost provides very fast inference, making it suitable for real-time applications. To improve scalability further, we plan to deploy the model on cloud platforms, which would allow us to process larger datasets and make predictions at scale.

**Environmental Impact Minimization** 🌍

In our effort to minimize environmental impact, we adopted several strategies:

- **Efficient Hyperparameter Tuning**: **Optuna's pruning** functionality helped us optimize the model's performance efficiently by stopping unpromising trials early, reducing resource consumption.

- **Model Simplicity**: We optimized the model for fewer iterations and shallower depths to ensure a balance between accuracy and computational efficiency, thus minimizing energy consumption during training.

- **Reduced Redundancy**: By using **feature selection**, we ensured that only the most important features were included, reducing the amount of data that needed to be processed and improving the overall computational efficiency.

Our approach aligns with sustainable business practices by optimizing the model to achieve high performance without unnecessary resource usage.


### **IV. Results and Future Potential**

- **Accuracy**: We achieved an **F1 score of 53.19%** on the test dataset, which demonstrates the effectiveness of the CatBoost model for groundwater level classification.

- **Explainability**: To enhance the explainability of our CatBoost Classifier, we analyzed the most important features contributing to the model's predictions. CatBoost provides built-in tools for interpreting the model, such as the feature importance ranking. This helps us understand which variables are influencing the groundwater level classification the most and provides transparency into the decision-making process.

**Future Solutions ✈️**

- **Model Optimization**: We plan to explore advanced techniques such as **reducing tree redundancy** or experimenting with alternative gradient boosting models like **LightGBM** for comparison to further improve performance and efficiency.

- **Environmental Focus**: We are investigating techniques like **model distillation** or **quantization** to reduce the computational requirements of the trained model, allowing us to maintain high performance while minimizing environmental impact.

- **Scaling Strategies**: We are exploring **distributed training** methods or utilizing cloud infrastructure to handle larger datasets and further enhance scalability. This will allow us to train and deploy CatBoost models more efficiently for larger-scale real-world applications.


- **Model Optimization**: We plan to explore more advanced techniques, such as reducing tree redundancy or using gradient boosting for comparison.

* **Environmental Focus**: Investigating model distillation or quantization to further reduce computational needs while maintaining performance.

* **Scaling Strategies**: Implementing distributed training or utilizing cloud infrastructure for larger datasets to enhance scalability.
