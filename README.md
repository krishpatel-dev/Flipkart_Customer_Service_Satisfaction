# Flipkart Customer Service Satisfaction

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Computation-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3776AB?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Charts-11557C?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SMOTE](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-4B8BBE?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)


---

## Project Summary

This project analyzes **customer interactions and feedback** at **Flipkart**, one of India's leading e-commerce platforms, to understand and improve **Customer Satisfaction (CSAT)** scores across various support channels.

As competition in e-commerce continues to grow, delivering exceptional customer service has become critical for driving growth and maintaining customer loyalty. The dataset comprises customer support interactions from various channels such as chat, phone, and email, along with customer feedback and CSAT scores.

The main goal is to:
- Identify the **key factors that influence customer satisfaction**
- Assess the **performance of customer service teams and agents**
- Develop **strategies to improve service quality**
- Build a **classification model** to predict CSAT score categories

By analyzing patterns in feedback, issue resolution times, and agent performance, this project pinpoints areas for improvement. Key metrics such as handling time, time-to-resolution, and customer remarks are examined to evaluate the effectiveness of existing support strategies.

Enhancing these service-related KPIs enables Flipkart to resolve issues more efficiently while providing support tailored to diverse customer needs. These improvements will help optimize agent performance, streamline support processes, improve CSAT scores, and ultimately boost brand loyalty and customer retention.

---

## Problem Statement

Flipkart seeks to improve customer satisfaction (CSAT) across various support channels. However, the key factors influencing CSAT — such as issue resolution time, agent performance, and support channel — remain unclear. This project aims to:

1. **Identify key drivers** of customer satisfaction through EDA and statistical testing.
2. **Analyze agent and channel performance** to detect inefficiencies and underperforming segments.
3. **Build a classification model** to predict CSAT score categories from operational features.
4. **Recommend data-driven strategies** to improve service quality and customer loyalty.

---

## Dataset Profile

**File:** Customer_support_data.csv  
**Size:** ~20 MB  
**Records:** 85,907 rows (after wrangling)  
**Original Columns:** 20  
**Type:** Tabular — mix of categorical, numerical, and datetime fields

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| Unique id | Integer | Unique identifier for each support interaction |
| channel_name | Categorical | Support channel used (Chat, Phone, Email, Inbound, Outcall) |
| category | Categorical | Broad issue category (e.g., Returns, Payments) |
| Sub-category | Categorical | Specific sub-type of the issue |
| Customer Remarks | Text | Free-text feedback written by the customer |
| Order_id | String | Flipkart order ID associated with the issue |
| order_date_time | Datetime | Timestamp of the original order placement |
| Issue_reported at | Datetime | Timestamp when the customer reported the issue |
| issue_responded | Datetime | Timestamp when the agent first responded |
| Survey_response_Date | Date | Date the customer submitted the CSAT survey |
| Customer_City | Categorical | City from where the customer raised the issue |
| Product_category | Categorical | Product category involved in the complaint |
| Item_price | Float | Price of the product in the associated order |
| connected_handling_time | Float | Total agent-connected time for the interaction |
| Agent_name | Categorical | Name of the handling support agent |
| Supervisor | Categorical | Supervisor overseeing the agent |
| Manager | Categorical | Manager overseeing the supervisor |
| Tenure Bucket | Categorical | Agent experience bucket (e.g., 0-30, 31-60, 61-90 days) |
| Agent Shift | Categorical | Shift when the interaction occurred (Morning, Evening, Night) |
| CSAT Score | Integer | Customer Satisfaction Score rated 1-5 by the customer |

---

## Project Workflow

### 1. Know Your Data
- Loaded and viewed the dataset — 85,907 rows across 20 columns
- Identified data types: numerical, categorical, datetime, and free-text
- Checked for duplicate rows — **no duplicates found**
- Identified missing values across key columns: Customer Remarks, connected_handling_time, order_date_time, Item_price, Product_category, Customer_City

### 2. Understanding Variables
- Explored unique value counts per column (cardinality)
- Identified high-cardinality columns: Agent_name, Order_id, Customer_City
- Identified key target variable: CSAT Score (integer, 1–5 scale)

### 3. Data Wrangling

#### Missing Value Treatment
| Column | Null % | Treatment |
| :--- | :--- | :--- |
| order_date_time | > 70% | Dropped (too many nulls) |
| Customer Remarks | > 70% | Dropped (too many nulls) |
| connected_handling_time | > 70% | Dropped (too many nulls) |
| Order_id | Low | Forward-filled (fill) — assumes nearby entries share the same order |
| Item_price | Moderate | Dropped for ML stage |
| Product_category | Moderate | Dropped for ML stage |
| Customer_City | Moderate | Dropped for ML stage |

#### Feature Engineering
- **Datetime Parsing:** Issue_reported at and issue_responded parsed from string to datetime
- **Time difference:** Computed as the difference in minutes between Issue_reported at and issue_responded — this becomes the primary numerical predictor
- **CSAT Score Category:** Continuous CSAT Score binned into 5 labeled categories: Very Poor, Poor, Average, Good, Excellent

#### Text Preprocessing (on channel_name, category, Agent_name, etc.)
- Lowercasing all string columns
- Removing punctuation characters
- Removing URLs and digit-containing words
- Stripping extra whitespace

### 4. Data Visualization and EDA (15 Charts)

| Chart | Type | Key Insight |
| :--- | :--- | :--- |
| Chart 1 | Grouped Bar | Email channel has higher avg CSAT than Phone; channels vary significantly in effectiveness |
| Chart 2 | Boxplot | Morning shift shows narrower IQR and lower median time difference — more consistent responses |
| Chart 3 | Count Plot | Distribution of CSAT scores — most scores cluster at 3 and 5 |
| Chart 4 | Scatter Plot | Time Difference vs Item Price, hued by CSAT — no clear linear pattern |
| Chart 5 | Pie Chart | Inbound is the most used channel; Outcall is least |
| Chart 6 | Bar Chart | Hyderabad generates the most support requests, followed by New Delhi |
| Chart 7 | Boxplot | CSAT distribution by Agent Shift — roughly even across shifts |
| Chart 8 | Bar Chart | Average CSAT Score by Manager — Emily Chen has the highest average CSAT |
| Chart 9 | Boxplot | CSAT Score distribution by Product Category — electronics show wider variance |
| Chart 10 | Boxplot | CSAT Score distribution by Channel — inbound and email slightly outperform phone |
| Chart 11 | Bar Chart | Average CSAT Score by Tenure Bucket — Tenure 61-90 days has the highest CSAT score |
| Chart 12 | Line Plot | Trend of CSAT scores across tenure buckets — experience positively correlates with CSAT |
| Chart 13 | Bar Chart | Sub-category wise distribution of interactions |
| Chart 14 | Heatmap | Correlation heatmap — Time difference has the strongest numerical correlation with CSAT |
| Chart 15 | Pair Plot | Pairwise distributions of key numerical features |

### 5. Hypothesis Testing

#### Hypothesis 1 — Time Difference vs CSAT Score
- **H0:** There is no significant relationship between Time Difference and CSAT Score
- **H1:** There is a significant relationship between Time Difference and CSAT Score
- **Test Used:** Spearman Rank Correlation
- **Reason:** Spearman is used for non-normal, ordinal/ranked data — appropriate here since CSAT is on a 1-5 scale
- **Result:** Statistically significant correlation found — longer response times negatively impact CSAT

#### Hypothesis 2 — Agent Shift vs CSAT Score
- **H0:** There is no significant difference in average CSAT Score between different Agent Shifts
- **H1:** There is a significant difference in average CSAT Score between Agent Shifts
- **Test Used:** Kruskal-Wallis H Test
- **Reason:** Non-parametric test for comparing medians across 3+ independent groups (Morning, Evening, Night) without assuming normal distribution
- **Result:** Significant difference found across shifts — agent shift does impact CSAT

#### Hypothesis 3 — Item Price vs CSAT Score
- **H0:** There is no significant correlation between Item Price and CSAT Score
- **H1:** There is a significant correlation between Item Price and CSAT Score
- **Test Used:** Spearman Rank Correlation
- **Result:** No significant correlation — item price alone does not drive customer satisfaction

### 6. Feature Engineering and Preprocessing

| Step | Details |
| :--- | :--- |
| **Outlier Handling** | IQR method applied to CSAT Score; extreme outliers removed |
| **Label Encoding** | Applied to channel_name and Agent Shift for ML compatibility |
| **CSAT Categorization** | CSAT Score binned into 5 classes: Very Poor, Poor, Average, Good, Excellent |
| **Class Imbalance** | SMOTE (Synthetic Minority Over-sampling Technique) applied to training data |
| **Feature Selection** | Correlation matrix used — Time difference identified as most relevant predictor |

**Selected Features for ML Model:**
- Time difference — most correlated numerical feature with CSAT
- Agent Shift — significant categorical predictor (confirmed by hypothesis test)

### 7. ML Model — Logistic Regression (Classification)

#### Model Configuration
- **Algorithm:** Logistic Regression (class_weight='balanced')
- **Target Variable:** CSAT Score Category (5-class: Very Poor, Poor, Average, Good, Excellent)
- **Features:** Time difference, Agent Shift
- **Train/Test Split:** 80% training / 20% testing (
andom_state=42)
- **Class Imbalance:** Handled using SMOTE on training set

#### Evaluation
| Metric | Score |
| :--- | :--- |
| Accuracy | ~68% |
| Evaluation Method | Classification Report + Confusion Matrix |
| Cross-Validation | 5-Fold CV with scoring='accuracy' |

#### Hyperparameter Tuning (GridSearchCV)
`
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],   # Regularization strength
    'solver': ['liblinear', 'saga']  # Optimization algorithm
}
`

| Tuning Result | Value |
| :--- | :--- |
| Best C | 0.1 |
| Best solver | liblinear |
| Best CV Accuracy | ~68.7% |

**Improvement:** GridSearchCV improved accuracy over the baseline model, confirming the value of hyperparameter tuning for this classification task.

---

## Key Insights and Findings

1. **Response Time is Critical:** Longer time differences between issue reporting and response are strongly associated with lower CSAT scores. Fast response is the #1 operational lever for improving satisfaction.
2. **Agent Shift Matters:** Morning shifts show more consistent and quicker response times (narrower IQR). Evening and night shifts handle more complex queries, leading to higher variability in CSAT.
3. **Channel Performance Varies:** Email achieves higher CSAT than Phone. Inbound is the most-used channel. Phone-based support needs targeted improvement.
4. **Experience Drives Quality:** Agents in the 61-90 day tenure bucket deliver the highest CSAT scores — experience in the role directly correlates with customer satisfaction.
5. **Top Manager:** Emily Chen's team consistently delivers the highest average CSAT — her management style can be studied and replicated.
6. **City-Level Demand:** Hyderabad and New Delhi generate the most support requests — these cities should be prioritized for staffing and resource allocation.
7. **Item Price Does Not Drive CSAT:** There is no significant correlation between product price and CSAT score — satisfaction is driven by service quality, not order value.

---

## Strategic Business Recommendations

1. **Reduce Response Times:** Set SLA targets for issue_responded timestamps, especially for high-volume channels like Inbound and Phone. Automated first-response acknowledgements can reduce perceived wait time.
2. **Invest in Agent Training:** Agents in 0-30 and 31-60 day buckets have lower CSAT — structured onboarding and mentorship programs tied to experienced agents can accelerate improvement.
3. **Optimize Shift Allocation:** Morning shifts outperform evening/night shifts in response consistency. Redistributing complex issue types to morning teams or providing additional support for night shift agents can reduce CSAT variance.
4. **Scale Resources in Key Cities:** Hyderabad and New Delhi drive the highest request volume. Dedicated regional support teams or priority routing for these cities could improve resolution speed.
5. **Improve Phone Channel Experience:** Email outperforms phone for CSAT. Implementing call scripting, shorter hold times, and first-call resolution targets will help close this gap.
6. **Replicate High-Performing Management Practices:** Conduct internal reviews with Emily Chen's team to identify and scale their workflows and coaching methods across other supervisor groups.

---

## Repository Structure

```
Flipkart Project/
├── Customer_support_data.csv              # Raw dataset (~20 MB, excluded from Git)
├── Flipkart_EDA_plus_ML_Project.ipynb    # Main notebook: EDA + Hypothesis Testing + ML
├── Flipkart_ML_Project.ipynb             # Supplementary ML-focused notebook
├── .gitignore                             # Git ignore rules (CSV, pickle, checkpoints, etc.)
└── README.md                             # Project documentation (this file)
```

---

## Installation and Getting Started

### Prerequisites
- **Python 3.8+**
- **Jupyter Notebook** or **JupyterLab**

### 1. Clone the Repository
```bash
git clone https://github.com/krishpatel-dev/Flipkart_Customer_Service_Satisfaction.git
cd Flipkart_Customer_Service_Satisfaction
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
```

### 4. Add the Dataset
Download `Customer_support_data.csv` and place it in the root project folder (same level as the notebooks).

### 5. Launch Jupyter Notebook
```bash
jupyter notebook Flipkart_EDA_plus_ML_Project.ipynb
```

---

## Technologies Used

| Library | Purpose |
| :--- | :--- |
| `pandas` | Data loading, wrangling, and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Static chart rendering |
| `seaborn` | Statistical data visualization |
| `scipy` | Hypothesis testing (Spearman, Kruskal-Wallis) |
| `scikit-learn` | ML model, GridSearchCV, Label Encoding, train-test split |
| `imbalanced-learn` | SMOTE for class imbalance handling |
| `datetime` | Datetime parsing and feature engineering |

