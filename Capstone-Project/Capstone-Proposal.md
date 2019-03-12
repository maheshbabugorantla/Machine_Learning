# **Capstone Project Proposal**

## **Kaggle Challenge: PetFinder.my Adoption Prediction**

> Mahesh Babu Gorantla - March 11, 2019

- [**Capstone Project Proposal**](#capstone-project-proposal)
  - [**Kaggle Challenge: PetFinder.my Adoption Prediction**](#kaggle-challenge-petfindermy-adoption-prediction)
    - [**Domain Background**](#domain-background)
    - [**Problem Statement**](#problem-statement)
    - [**Datasets and Inputs**](#datasets-and-inputs)
      - [**File Descriptions**](#file-descriptions)
      - [**Data Fields**](#data-fields)
      - [**AdoptionSpeed**](#adoptionspeed)
      - [**Dropping Irrelevant Columns**](#dropping-irrelevant-columns)
      - [**Descriptive Statistics**](#descriptive-statistics)
    - [**Solution Statement**](#solution-statement)
    - [**Benchmark Model**](#benchmark-model)
    - [**Evaluation Metrics**](#evaluation-metrics)
    - [**Project Design**](#project-design)
    - [**References**](#references)

<a name="domain-background"></a>

### **Domain Background**

Each year, approximately 7.6 million companion animals enter the animal shelters nationwide (ASPCA). Of those, approximately 3.9 million are dogs and 3.4 million are cats.
About 2.7 million shelter animals are adopted each year (1.4 million dogs and 1.3 million cats). This leaves around two-thirds of pets going unadopted.

This uncertainty of the unadopted pets future is further worsened because many a times a pet shelter's future is typically uncertain and many pet shelters cannot keep
the unadopted pets for extended periods of time due to logistical reasons and many of the pet shelters are run voluntarily (mostly dependent on generous donations).

If at all we can find those stray animals a forever home, many precious lives can be given a new life that they truly deserve and in turn we will see more happier families.

I am big fan of pets. My personal motivation to solve this problem explicitly is because doing so will make the [PetFinder.my](https://www.petfinder.my/) Agency streamline
their pet adoption services which in turn help more pets being adopted. Being able to predict how quickly a pet will be adopted will help pet adoption agencies optimize their
logistics and operate more efficiently reducing the percentage of unadopted pets.

The problem at hand currently is to create a model to predict the adoption speed of a pet given its features (mentioned below).

Using a machine learning algorithm(s) to predict the adoption speed of a pet should help us solve the above mentioned problem.

[PetFinder.my](https://www.petfinder.my/) has been Malaysia's leading animal welfare platform since 2008, with a database of more than 150,000 animals.
PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.

>> Relevant Academic Research has been cited in the `References` section

<a name="problem-statement"></a>

### **Problem Statement**

The Pet Adoption Prediction is a multi-class classification problem. In this problem, I need to develop algorithms to predict the adoptability of pets - specifically, how quickly is a pet adopted based on it features (which have been explained in the section below).

It is a classificiation problem because a pet's adoption speed is classified into 5 categories.

A relevant potential solution is to use a few classification algorithms such as

- k-Means Clustering
- DBSCAN
- Random Forests
- XGBoost (Baseline model)

And, choose the one with best `cross_validation` score on the training dataset.

<a name="datasets-and-inputs"></a>

### **Datasets and Inputs**

In this problem, I will predict the speed at which a pet is adopted, based on the pet’s listing on PetFinder. Sometimes a profile represents a group of pets. In this case, the speed of adoption is determined by the speed at which all of the pets are adopted. The data included text, tabular, and image data.

<a name="file-descriptions"></a>

#### **File Descriptions**

- `train.csv` - Tabular/text data for the training set
- `test.csv` - Tabular/text data for the test set
- `sample_submission.csv` - A sample submission file in the correct format
- `breed_labels.csv` - Contains Type, and BreedName for each BreedID. Type 1 is `dog`, 2 is `cat`.
- `color_labels.csv` - Contains ColorName for each `ColorID`
- `state_labels.csv` - Contains StateName for each `StateID`
<br><br>

<a name="data-fields"></a>

#### **Data Fields**

- `PetID`- Unique hash ID of pet profile
- `AdoptionSpeed` - Categorical speed of adoption. Lower is faster. `This is the value to predict`. See below section for more info.

- `Type` - Type of animal ( 1  = Dog, 2  = Cat)
- `Name` - Name of pet (Empty if not named)
- `Age` - Age of pet when listed, in months
- `Breed1` - Primary breed of pet (Refer to BreedLabels dictionary)

- `Breed2` - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)

- `Gender` - Gender of pet ( 1 = `Male`,  2 = `Female`, 3 = `Mixed`, if profile represents group of pets)

- `Color1` - Color 1 of pet (Refer to ColorLabels dictionary)

- `Color2` - Color 2 of pet (Refer to ColorLabels dictionary)

- `Color3` - Color 3 of pet (Refer to ColorLabels dictionary)

- `MaturitySize` - Size at maturity ( 1  = `Small`,  2  = `Medium`,  3  = `Large`,  4  = `Extra Large`,  0  = `Not Specified`)

- `FurLength` - Fur length ( 1 = `Short`, 2 = `Medium`, 3 = `Long`, 0 = `Not Specified`)

- `Vaccinated` - Pet has been vaccinated ( 1 = `Yes`,  2 = `No`, 3 = `Not Sure`)

- `Dewormed` - Pet has been dewormed ( 1 = `Yes`, 2 = `No`, 3 = `Not Sure`)

- `Sterilized` - Pet has been `spayed` / `neutered` ( 1  = `Yes`, 2 = `No`, 3 = `Not Sure`)

- `Health` - Health Condition ( 1 = `Healthy`, 2 = `Minor Injury`, 3 = `Serious Injury`, 0 = `Not Specified`)

- `Quantity` - Number of pets represented in profile
Fee - Adoption fee ( 0 = `Free`)

- `State` - State location in Malaysia (Refer to StateLabels dictionary)

- `RescuerID` - Unique hash ID of rescuer

- `VideoAmt` - Total uploaded videos for this pet

- `PhotoAmt` - Total uploaded photos for this pet

- `Description` - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

<a name="adoptionspeed"></a>

#### **AdoptionSpeed**

Contestants are required to predict this value. The value is determined by how quickly, if at all, a pet is adopted. The values are determined in the following way:

- 0 - Pet was adopted on the same day as it was listed.
- 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
- 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
- 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
- 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days)

<a name="dropping-irrelevant-columns"></a>

#### **Dropping Irrelevant Columns**

- `Name`, `PetID`, `RescuerID`
  - It makes sense to not have `Name` column because it does not help us predict how faster a pet animal will be adopted as we know that any prospective pet owner will not adopt a pet animal based on its name.
  - Likewise `PetID`, `RescuerID` also have no relevance to the task at hand

- However, for now to simplify the modeling let's ignore the following columns
  - `Fee` - Might have some implication if a pet will adopted (Need to check the correlation)
  - `State` - Might not influence the adoption speed (Need to check the correlation)
  - `VideoAmt`, `PhotoAmt` & `Description` - Will have some impact on the adoption speed. Because a beautiful looking pet might get adopted sooner. But just to make the modeling simpler let's ignore these columns as well for now.

<a name="descriptive-statistics"></a>

#### **Descriptive Statistics**

- The dataset has 14,993 datapoints

- **Dataset Distribution**

  | Adoption Speed | Data Points Size |
  |----------------|------------------|
  |       0        |       410        |
  |       1        |       3090       |
  |       2        |       4037       |
  |       3        |       3259       |
  |       4        |       4197       |

From the above observations we can determine that the dataset is almost **`imbalanced`**.

<a name="solution-statement"></a>

### **Solution Statement**

First, we will import the training dataset `/data/train/train.csv` and then drop irrelevant columns (that do not help our classification algorithm). Next, find the missing rows in each column and also generate descriptive statistics to understand the distribution of data. Usually on close analysis of  descriptive statistics we will often find outliers/abnormal values in the `min`/`max` rows for each column.

Create custom data imputers to fill in the missing values for both quantitative (`mean`/`median` value for the column) and categorical columns (`mode` value for the column). Create a pipeline to fit and transform the data using the custom data imputers. After imputing the missing values, plot histograms per pet category (1 - Dogs and 2 - Cats) to further examine the distributions per column, determine any anomalies found and correct those anomalies with appropriate data correction techniques. And, plot the column correlation heatmap to find any closely related columns (if any).

Once all the processes are implemented as mentioned above, the data at this point can now safely be assumed with no missing or anomalous values. Now, we need to dummify any categorical columns and scale each column either using `StandardScaler` or `MinMaxScaler` (I chose to use `MinMaxScaler` for now).

Now, lets split the cleaned and scaled data into training features and prediction variable (i.e. `AdoptionSpeed`).

The above mentioned problem is a supervised multi-class classification problem. To predict `AdoptionSpeed` based on the training features of a pet, I would like to use classification algorithms like

1. Classification and Regression Trees (CART)

2. kNN Classifier

3. Support Vector Machines (SVM)

4. Random Forests

5. XGBoost

to predict the adoptibility of a pet.

Find the model with best parameters using `GridSearchCV` and save the model.

<a name="benchmark-model"></a>

### **Benchmark Model**

As we know that the above mentioned prediction task is a supervised classification problem, we should use tree based classification models which typically outperform other classification models. Hence, I would like to use `kNN Classfier` and `XGBoost` algorithms as a benchmark and beat the benchmark perfomance both in terms of time to fit the model as well as increase in prediction accuracy.

Currently, the baseline prediction accuracy (without feature transformation) is as follows

| Classification Model | Model Fit Time (in mins) | Cross Validation Accuracy |
|----------------------|--------------------------|---------------------------|
| KNeighborsClassifier | 1.01                     | 31.17%                    |
| XGBoost              | 0.99                     | 38.01%                    |

From the above results, we can infer that `XGBoost` should outperform all the above mentioned tree-based classification algoritms because we are able to get higher prediction accuracy given the higher dimensional space in lesser time than `kNN`.

Check [Benchmark Model Notebook](`https://github.com/maheshbabugorantla/Udacity_Machine_Learning/blob/Capstone_Project/Capstone-Project/Kaggle-Notebook.ipynb`) for information on feature engineering techniques used to impute the missing data in the given dataset.

To improve the performance of the model over the benchmark model, we will need to find more anomalies in the data (if there are any and correct them). Reduce the dimensionality of the dataset using **`LDA`** - **`Linear Discriminant Analysis`** techique and test for the performance on the `train` and `test` datasets.

<a name="evaluation-metrics"></a>

### **Evaluation Metrics**

I would like to propose an evaluation metric called `Cross Validation Score`. The simplest way to use cross-validation is to call the `cross_val_score` helper function on the estimator and the dataset.

Given, the problem is to develop a supervised multi-class classification model, the `cross_val_score` method uses `StratifiedKFold` cross-validation technique. `StratifiedKFold` cross-validation method is a variation of `KFold` that returns startified folds. The folds are made by preserving the percentage of samples for each class.

The model whose cross validation score is greater than the benchmark model will be my solution model.

<a name="project-design"></a>

### **Project Design**

Theoretical Workflow for designing the Solution Model

- **Import the Training Dataset**
  - Load the necessary libraries such as `numpy`, `pandas`
  - Load the `data/train/train.csv` dataset
  - Drop irrelevant columns

- **Data Exploration**
  - Find missing rows in the each column using `df.isnull().sum()`
  - Obtain and Understand Descriptive Statistics

- **Feature Engineering**
  - Decipher more missing values from descriptive statistics (if any)
  - Impute missing values using Custom Quantitative/Categorical Imputers
  - Visualize per column distribution using histograms
  - Find any anomalous data distributions from the plots
  - Dummify Categorical Columns
  - Scale all the columns either by using `z-scaling` (`StandardScaler`) or `min-max scaling` (`MinMaxScaler`) technique

- **Evaluate Classification Models**
  - Choose a few classification models
  - Split the dataset into train/split test using `StratifiedKFold` cross-validation technique
  - Score each classification model using `cross_val_score`
  - Choose the classification model with best cross validation score

- **Feature Transformation**
  - Reduce the dimensionality of features
    - Using LDA - Linear Discriminant Analysis (Best for multi-class classification)

- **Tune the Classification Model**
  - Use `GridSearchCV` to improve the accuracy

- **Save the best model**
  - with best prediction time and accuracy

- **Final Inference and Conclusion**

> Please check the [Notebook](https://github.com/maheshbabugorantla/Udacity_Machine_Learning/blob/Capstone_Project/Capstone-Project/Kaggle-Notebook.ipynb) to understand what I meant by Feature Engineering

<a name="references"></a>

### **References**

1. Weiss, E., Miller, K., Mohan-Gibbons, H., & Vela, C. (2012). Why Did You Choose This Pet?: Adopters and Pet Selection Preferences in Five Animal Shelters in the United States. Animals, 2(2), 144-159. doi:10.3390/ani2020144

2. Weiss, E., Dolan, E., Garrison, L., Hong, J., & Slater, M. (2013). Should Dogs and Cats be Given as Gifts? Animals, 3(4), 995-1001. doi:10.3390/ani3040995

3. Garrison, L., & Weiss, E. (2014). What Do People Want? Factors People Consider When Acquiring Dogs, the Complexity of the Choices They Make, and Implications for Nonhuman Animal Relocation Programs. Journal of Applied Animal Welfare Science, 18(1), 57-73. doi:10.1080/10888705.2014.943836

4. PetFinder.my Adoption Prediction. (n.d.). Retrieved from https://www.kaggle.com/c/petfinder-adoption-prediction

5. Raschka, S. (2014, August 03). Linear Discriminant Analysis. Retrieved from https://sebastianraschka.com/Articles/2014_python_lda.html

6. Cross-validation: Evaluating estimator performance¶. (n.d.). Retrieved from https://scikit-learn.org/stable/modules/cross_validation.html

7. Sklearn.metrics.f1_score¶. (n.d.). Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score.html

8. Sklearn.model_selection.cross_val_score¶. (n.d.). Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

9. Sklearn.model_selection.StratifiedKFold¶. (n.d.). Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.html

10. Sklearn.base.TransformerMixin¶. (n.d.). Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html

11. Ozdemir, S., & Susarla, D. (2018). Feature engineering made easy: Identify unique features from your dataset in order to build powerful machine learning systems. Birmingham, UK: Packt Publishing
