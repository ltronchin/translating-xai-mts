# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Tensorflow import
import tensorflow as tf

# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions

dataset = helpers.load_adult_income_dataset()

print(dataset.head())

# description of transformed features
adult_info = helpers.get_adult_data_info()
print(adult_info)

# Split the dataset into train and test sets.
target = dataset["income"]
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=0, stratify=target)
x_train = train_dataset.drop('income', axis=1)
x_test = test_dataset.drop('income', axis=1)

# Given the train dataset, we construct a data object for DiCE. Since continuous and discrete features have different ways of
# perturbation, we need to specify the names of the continuous features. DiCE also requires the name of the output variable that the ML model will predict.
# Step 1: dice_ml.Data
d = dice_ml.Data(dataframe=train_dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')

numerical = ["age", "hours_per_week"]
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(transformers=[  ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', RandomForestClassifier())])
model = clf.fit(x_train, y_train)

# Generating counterfactual examples using DiCE
# We now initialize the DiCE explainer, which needs a dataset and a model. DiCE provides local explanation for the model  m and requires an query input whose outcome needs to be explained.
# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")
# Using method=random for generating CFs
exp = dice_ml.Dice(d, m, method="random")

e1 = exp.generate_counterfactuals(x_test[0:1], total_CFs=2, desired_class="opposite")
e1.visualize_as_dataframe(show_only_changes=True)

# https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_getting_started.ipynb