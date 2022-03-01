# Bank-Loan-Application
### INTRODUCTION
![Loan-Image](intro.jpg).
This Bank seeks to automate (in real time) the loan qualifying procedure based on information given by customers while filling out an online application form.\
It is expected that the development of machine learning models that can help the company predict loan approval in accelerating decision-making process for determining whether an applicant is eligible for a loan or not.
The Data Set was Obtained from kaggle data sets.

# DATASET DESCRIPTION

*This dataset has 614 samples with 13 features 
Here is a list of the features in this dataset:

**Loan_ID**:	Loan reference number i.e (unique ID)	LP001002; LP001003; ...\
**Gender**:	Applicant gender (Male or Female)	Male; Female.\
**Married**:	Applicant marital status (Married or not married)	Married; Not Married.\
**Dependents**:	Number of family members	0; 1; 2; 3+.\
**Education**:	Applicant education/qualification (graduate or not graduate)	Graduate; Under Graduate.\
**Self_Employed**:	Applicant employment status (yes for self-employed, no for employed/others)	Yes; No.\
**ApplicantIncome**:	Applicant's monthly salary/income	5849; 4583; ...\
**CoapplicantIncome**:	Additional applicant's monthly salary/income	1508; 2358; ...\
**LoanAmount**:	Loan amount	128; 66; ...\
**Loan_Amount_Term**:	The loan's repayment period (in days)	360; 120; ...\
**Credit_History**:	Records of previous credit history (0: bad credit history, 1: good credit history)	0; 1.\
**Property_Area**:	The location of property (Rural/Semiurban/Urban)	Rural; Semiurban; Urban.\
**Loan_Status**:	Status of loan (Y: accepted, N: not accepted)	Y; N.

# This project intends to answer the following questions:
*What are the various relationship between information filled by loan applicants.\
*influence of this information on applicant's loan status.\
*What model best fits loan application automation process task and it's accuracy level

# Benefit of project
*Help Banks to automate (in real time) the loan qualifying procedure based on information given by customers while filling out an online application form.\
*It will help bank track previous loan status records of customers.

## MODELS EXPLORED
*Logistic Regression\
*K-NeighborsClassifier\
*DecisionTreeClassifier\
*Random Forest Classfier\
*Naive Bayes\
*Support Vector Machine.\
*Gradient Boost.

### EXPLORATORY DATA ANALYSIS OF DATA SET
#Gender 
![Gender](GendDist.png).

![Gender](GendCount.png).

There were more male applicants than female while 2.1% of applicants didn't specify gender.

#Marrital Status
![Marrital_STatus](MarrDist.png).
64.8% of applicants were married while 0.5% of applicants didn't indicate their marital status.

#Education
![Educational_STatus](Educa.png).
There were clearly more educated applicants than the uneducated.

#property Area
![Property_Location](Propty.png).
Most applicant properties were located at semiurban.

#Loan Status
![Loan_Status](LoStat.png).
From the data set we can infer that most applicants got their loan request approved.

#Dependents
![How Many depend on_applicant](Depend.png).
Most applicants had no dependant.

#Self Employed or Employee/Others
![Employment_Status](Employ.png).
Most applicant were not self employed.

#Credit History
![Credit_History](Credit.png).
Most applicant had good credit history, while 8.1% did not supply thier credit history.

#Loan Duartion
![Duration_of_Loan](Duration.png).
Most applicant opted for 360 days duration.

#Distribution of numerical values in data set
![Applicants_Income](Aplincom.png).
This is clearly left skewed and not well distributed.

![C0-applicants_Income](Coaplincom.png)
This is clearly left skewed and not well distributed.

![Loan_Amount](Loamount.png)
Slightly a better distribution though not the best.

#Relationship Between features in data set
![Gender vs Marital Status](GendvsMarry.png)

![Property Area vs Credit History](PropAreavsCredit.png)

![Property Area vs Loan Status](PropareavsLoanStat.png)

![Loan_status vs Marital Status](MarryvsLoanstat.png)

#Loan Status vs Applicantsincome
![Loan vs applicant_income](LoanvsIncome.png)
we can notice some outliers, this will affect our model but it will be addressed.

#Loan Amount vs Applicants income
![Loan vs applicant_income](Loanamvsincome.png)
there is a slight positive correlation between applicants income and amount of loan requested for.

#Loan_Amount and Application status
![Loan vs applicant_status](amountvsloanstat.png)

The amount of loan requested by customers does not influence thier application status.

## DATA PROCESSING
To clean up the data and prepare it for model building.\
I obtained the summary of missing values in each column of the data set using the python code. **df.isnull().sum()**.\

![Missing_values](missing.jpg)
We noticed that some values from some features are missing, this might likely be a case of customers leaving it blank while filling out thier information.\

To help clean up this, I replaced the missing values in the categorical variable using their mode. i.e the most frequent in that column while replacing the numerical variable using the average value of the column.\

I also dropped columns that was not so relevant in building our model, I dropped "Loan_ID".\

I also transformed all categorical variable with **string values** into a form that is allowed by models for our machine algorithm to make better prediction.\
This process is called **One-Hot Encoding** using the pandas library **df.get_dummies().\

The outliers earlier experienced in the numerical variables were removed.\

The skewness that was earlier experienced in distribution for ApplicantIncome, CoapplicantIncome, and LoanAmount was transformed.\
I used square root transformation to normalized the distribution using the numpy library.

## Normalized Distribution of Numerical Variables
![applicant_inc](Newincom.png)

![Coapplicat](NewCoapp.png)

![LoanAMA](NewLoanamt.png)

This Distribution can clearly be observed as being better than the one previously observed.

**Loan_Status** clearly is the target variable while every other features is taken as the independent variable.\
The train_test feature of sci-kit Learn was used to split the data set into 80% for training each of the model while 20% was used to test the model.

After building each model using the data set, the model accuracy for each was computed and arranged to provide information on models that performed well and a best fit for the bank automation plan.

### MODEL ACCURACY TABLE

![Model_accuracy](ModelTable.jpg)

##CONCLUSION
In general, it can be obsereved that all models can achieve up to **70%** accuracy. The highest accuracy is **91%** from the **Random Forest Model**.


