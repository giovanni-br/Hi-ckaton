# Hi!ckathon 5th Edition 2024: AI & Sustainability 🌲

## Event
<img src="images/image.png" alt="Hi! Paris Logo" width="200"/>
The present code is developed during a contest called 'HI!CKATHON' organized at the request of HEC Paris and Institut Polytechnique de Paris, in the context of the Hi! PARIS Center.
https://github.com/hi-paris/Hickathon5

## Project: Business and Scientific document - Work in progress - NOT FINAL

<img src="images/waterwizards_logo.png" alt="Water Wizards Logo" width="100"/>

**Water Wizards**, Forecasting Water, Fueling Progress

**I. Overview** 

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

**II. Our business plan**

This insight on water groundwater levels allows municipalities to enact predictive measures allowing for the preservation of water access for all, individuals and companies alike. They also allow companies and particulars using large quantities of water, such as farmers and energy producers, to use water responsibly through planning, since our solution allows them to see ahead, predicting high and low groundwater levels in their region.

Our solution can be deployed in areas suffering from a water crisis, as well as in areas not yet worried by drought, but where water uses are important and which are vulnerable to water shortages. Using data harvested through piezometric and weather stations across the country, the data we provide gives oversight on the groundwater level anywhere, anytime, especially in summer, when water resources come under pressure. It can eventually be deployed anywhere provided there is accurate data on the underground water levels.

Used by corporate and public authorities over water usage, our solution aims to bring an end to water shortages and protect communities from droughts, by being the first start-up to leverage data to predict changes in the groundwater levels, in order to allow companies to plan to avoid 

**III. Our expertise**

The team used an extensive dataset consisting of approximately 3 million entries and over 100 variables, featuring data harvested across France over a 4 year long span, to construct our AI model.

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

**AI model 🤖**

We chose the **Random Forest Classifier** because of its robustness, accuracy, and ability to handle complex, non-linear data. It effectively reduces overfitting by combining multiple decision trees and requires minimal preprocessing. Additionally, it provides feature importance for interpretability, and its parallel nature ensures scalability. With Optuna, we optimized its hyperparameters to balance performance and computational efficiency, making it an ideal choice for our classification task.

**Scalability**\
Random Forest scales well with larger datasets because it builds multiple decision trees independently and aggregates their outputs. This parallel nature makes it suitable for distributed computing. However, its scalability can be limited by memory usage, as each tree holds a copy of the data. For our use case:

- **Training scalability:** We used Optuna to optimize hyperparameters, focusing on reducing the number of trees and limiting depth, ensuring that model training remains feasible on larger datasets.

- **Inference scalability:** Once trained, the Random Forest model provides fast predictions, making it suitable for real-time applications. Deployment on cloud platforms can further enhance scalability.

To minimize environmental impact, we adopted several strategies:

- **Efficient Hyperparameter Tuning**: Optuna's pruning functionality reduces resource usage by stopping unpromising trials early, saving computation time and energy.

- **Model Simplicity**: We optimized for fewer trees and shallower depths to maintain accuracy while reducing computation. This lowers the carbon footprint compared to deeper, more complex models.

- **Reduced Redundancy:** Feature selection ensured that only relevant data was processed, avoiding unnecessary calculations.

- **Business Approach**: By tuning the model efficiently, we achieved better performance with less computational expense, aligning with sustainable business practices and cost efficiency.


**IV. Results and Future Potential**

- **Accuracy**: We achieved an F1 score of 37.32% on the test dataset.

- **Explainability**: To enhance the explainability of our Random Forest Classifier, we analyzed the **most important features** contributing to the model’s predictions. By examining feature importance, we identified which variables had the greatest impact on groundwater level classification. This approach helps us understand the model's decision-making process and provides transparency, ensuring that key factors influencing predictions are clear and interpretable.

• What is the future potential of your solution?

- **Model Optimization**: We plan to explore more advanced techniques, such as reducing tree redundancy or using gradient boosting for comparison.

* **Environmental Focus**: Investigating model distillation or quantization to further reduce computational needs while maintaining performance.

* **Scaling Strategies**: Implementing distributed training or utilizing cloud infrastructure for larger datasets to enhance scalability.
