# Importing the necessary Python modules.
from cachetools.cache import Cache
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header=None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns=0, inplace=True)
    column_headers = [
        'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'
    ]
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis=1, inplace=True)
    return df


glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

# main page
st.title('Glass Type Predictor')

# sidebar
st.sidebar.title("Exploratory Data Analysis")

# Using the 'if' statement, display raw data on the click of the checkbox.

# adding checkbox
show_data = st.sidebar.checkbox('Show raw data')

# displaying data
if show_data == True:
    st.subheader('Displaying data')
    st.dataframe(glass_df)

# creating plots for the data

# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)

# Scatter Plot between the features and the target variable.
# Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader('Scatter Plot')

# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect(
    'Select the X-axis values for scatter plot'.title(),
    ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# create the actual plot
for param in features_list:
    st.subheader(f'Scatter plot between {param} and glass type'.title())
    fig = plt.figure(figsize=(20, 5))
    plt.title(f'Scatter plot between {param} and glass type'.title())
    sns.scatterplot(x=param, y='GlassType', data=glass_df)
    plt.xlabel(param.title())
    plt.ylabel("Type of the glass")
    st.pyplot(fig=fig)

# Sidebar for histograms.
st.sidebar.subheader('Hsitogram')

# Choosing features for histograms.
features_list = st.sidebar.multiselect(
    'Select the values for histogram',
    ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# Create histograms.
for param in features_list:
    st.subheader(f'HIstogram for {param}'.title())
    fig = plt.figure(figsize=(20, 5))
    plt.title(f'HIstogram for {param}'.title())
    sns.distplot(glass_df[param],
                 bins='sturges',
                 kde=st.checkbox(f"KDE Line for {param}"),
                 hist=st.checkbox(f"Histogram for {param}", value=True))
    plt.xlabel(param.title())
    plt.ylabel("Type of the glass")
    st.pyplot(fig=fig)

# Sidebar for box plots.
st.sidebar.subheader('Boxplots')

# Choosing columns for box plots.
features_list = st.sidebar.multiselect(
    'Select the values on X-axis for boxplot',
    ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# Create box plots.
for param in features_list:
    st.subheader(f'Boxplot for {param}'.title())
    fig = plt.figure(figsize=(20, 5))
    plt.title(f'Boxplot for {param}'.title())
    sns.boxplot(x=param, data=glass_df)
    plt.xlabel(param.title())
    st.pyplot(fig=fig)

# Sidebar for countplots.
st.sidebar.subheader('Countplots')

# Choosing columns for box plots.
features_list = st.sidebar.multiselect(
    'Select the values on X-axis for count plot',
    ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# Create count plots.
for param in features_list:
    st.subheader(f'countplot for {param}'.title())
    fig = plt.figure(figsize=(20, 5))
    plt.title(f'countplot for {param}'.title())
    sns.countplot(x=param, data=glass_df)
    plt.xlabel(param.title())
    st.pyplot(fig=fig)

# Sidebar for pairplots.
st.sidebar.subheader('Pairplots')
yes = st.sidebar.checkbox("show pairplots")

# Create pair plots.
if yes == True:
    st.subheader(f'pairplot for our dataframe'.title())
    fig = sns.pairplot(glass_df)
    st.pyplot(fig=fig)

# Sidebar for box plots.
st.sidebar.subheader('Corelation heatmap')
yes = st.sidebar.checkbox("show")

# Choosing columns for box plots.
features_list = st.sidebar.multiselect(
    'Select the values for Corelation heatmap',
    ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))

# Create box plots.
if yes == True and len(features_list) > 1:
    st.subheader('Corelation heatmap')
    fig = plt.figure(figsize=(20, 5))
    plt.title(f'correlation heatmap'.title())
    sns.heatmap(glass_df[features_list].corr(), annot=True)
    st.pyplot(fig=fig)

# adding subheader for pie chart
st.sidebar.subheader("Pie chart")

# creating a check box
show = st.sidebar.checkbox("Show")

# creating values
if show == True:
    value_series = glass_df['GlassType'].value_counts()
    explode = np.linspace(0.06, 0.12, len(value_series))

    # creating plot
    fig_pie = plt.figure(dpi=160)
    plt.title(
        "Pie chart showing distribution of target classes in our dataframe".
        title())
    plt.pie(value_series,
            labels=value_series.index,
            explode=explode,
            autopct="%0.2f%%",
            startangle=120)
    st.pyplot(fig=fig_pie)

#taking inputs to predict data through sliders
st.sidebar.subheader("Set the values for predicting the glass type".title())
input = {}
for param in ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'):
    input[param] = st.sidebar.slider(f"Input {param}",
                                     float(glass_df[param].min()),
                                     float(glass_df[param].max()))

# creating a dropdown box for selecting the model
st.sidebar.subheader("Select Algorithm")
chosen_model = st.sidebar.selectbox("Select the classification model", [
    "Support Vector Machine", "Random FOrest Classifier", "Logistic Regression"
])

#Choosing Hyperparameters
st.sidebar.header("Select hyperparameters")

# Support Vector Machines
if chosen_model == "Support Vector Machine":
    c_val = st.sidebar.number_input("C (error margin)", 0, 100, step=1)
    kernel = st.sidebar.radio("kernel", ['linear', 'rbf', 'ploy'])
    gamma_inp = st.sidebar.number_input("gamma value", 0.00, 1.00, step=0.1)
    classify = st.sidebar.checkbox("Classify")
    if classify:
        svm = SVC(C=c_val, kernel=kernel,
                  gamma=gamma_inp).fit(X_train, y_train)
        #y_test_pred = SVC.predict(X_test)
        accuracy = svm.score(X_test, y_test)
        st.subheader("Predicted as: ")
        glass_type = svm.predict(
            np.array(list(input.values())).reshape(1, 9).astype(float))
        if glass_type == 1:
            st.write("building windows float processed")

        elif glass_type == 2:
            st.write("building windows non float processed")

        elif glass_type == 3:
            st.write("vehicle windows float processed")

        elif glass_type == 4:
            st.write("vehicle windows non float processed")

        elif glass_type == 5:
            st.write("containers")

        elif glass_type == 6:
            st.write("tableware")

        else:
            st.write("headlamp")

        st.write("Accouracy of the model is: ", round(accuracy, 2))
        plt.figure()
        plot_confusion_matrix(svm, X_test, y_test)
        st.pyplot()

# Random FOrest Classifier
elif chosen_model == "Random FOrest Classifier":
    n_est_inp = st.sidebar.number_input(
        "Number of classification trees in the model".title(),
        100,
        500,
        step=10)
    max_depth_inp = st.sidebar.number_input(
        "Maximun depth of the tree".title(), 1, 100, step=1)

    if st.sidebar.checkbox("Classify"):
        rand_for_clf = RandomForestClassifier(n_estimators=n_est_inp,
                                              max_depth=max_depth_inp).fit(
                                                  X_train, y_train)
        accuracy = rand_for_clf.score(X_test, y_test)
        st.subheader("Predicted as: ")
        glass_type = rand_for_clf.predict(
            np.array(list(input.values())).reshape(1, 9).astype(float))
        if glass_type == 1:
            st.write("building windows float processed")

        elif glass_type == 2:
            st.write("building windows non float processed")

        elif glass_type == 3:
            st.write("vehicle windows float processed")

        elif glass_type == 4:
            st.write("vehicle windows non float processed")

        elif glass_type == 5:
            st.write("containers")

        elif glass_type == 6:
            st.write("tableware")

        else:
            st.write("headlamp")

        st.write("Accouracy of the model is: ", round(accuracy, 2))
        plt.figure()
        plot_confusion_matrix(rand_for_clf, X_test, y_test)
        st.pyplot()

# Logistic Regression
else:
    c_val = st.sidebar.number_input("C (error margin)", 0, 100, step=1)
    max_iter_inp = st.sidebar.slider(
        "Maximum number cost function should run".title(), 100, 500, step=10)
    if st.sidebar.checkbox("Classify"):
        lr = LogisticRegression(max_iter=max_iter_inp, C= c_val).fit(X_train, y_train)
        accuracy = lr.score(X_test, y_test)
        st.subheader("Predicted as: ")
        glass_type = lr.predict(
            np.array(list(input.values())).reshape(1, 9).astype(float))
        if glass_type == 1:
            st.write("building windows float processed")

        elif glass_type == 2:
            st.write("building windows non float processed")

        elif glass_type == 3:
            st.write("vehicle windows float processed")

        elif glass_type == 4:
            st.write("vehicle windows non float processed")

        elif glass_type == 5:
            st.write("containers")

        elif glass_type == 6:
            st.write("tableware")

        else:
            st.write("headlamp")

        st.write("Accouracy of the model is: ", round(accuracy, 2))
        plt.figure()
        plot_confusion_matrix(lr, X_test, y_test)
        st.pyplot()