import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_target_dist(df):
    sns.set(style = 'whitegrid')
    sns.set_context('paper', font_scale = 2)
    fig = plt.figure(figsize = (20, 10))
    plt.subplot(121)
    plt.pie(df.churn.value_counts(),labels = ['No Churn', 'Churn'], autopct = '%.1f%%', radius = 1, textprops={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Churn Outcome Pie Chart', fontsize = 20, fontweight = 'bold')
    plt.subplot(122)
    t = sns.countplot(df.churn)
    t.set_xlabel('Churn', fontweight = 'bold', fontsize = 20)
    t.set_ylabel('Count', fontweight = 'bold', fontsize = 20)
    plt.title('Churn Outcome Distributions', fontsize = 20, fontweight = 'bold')
    plt.tight_layout()

    
def plot_cat_dist(df, feature):
    sns.set(style = 'whitegrid')
    sns.set_context('paper', font_scale = 2)
    fig = plt.figure(figsize = (20, 10))
    plt.subplot(121)
    plt.pie(df[df.churn == 'yes'][feature].value_counts(), autopct = '%.1f%%', radius = 1, textprops = {'fontsize':20, 'fontweight':'bold'})
    
    
    
    
    
def plot_kde(df, feature):
    plt.figure(figsize = (15, 5))
    plt.title(f"KDE Plot: {feature}", fontsize = 20, fontweight = 'bold')
    ax = sns.kdeplot(df[df.churn == 'No'][feature].dropna(), label = 'No Churn', lw = 2, legend = True)
    plt.legend = True
    ax1 = sns.kdeplot(df[df.churn == 'Yes'][feature].dropna(), label = 'Churn', lw = 2, legend = True)
    if feature == 'tenure':
        plt.xlabel('Tenure Length (Months)')
    else:
        plt.xlabel('Charge Amount ($)')
    plt.tight_layout()
    
    
def plot_hist(df, feature):
    plt.figure(figsize = (15, 5))
    plt.title(f'Histogram: {feature}', fontsize = 20, fontweight = 'bold')
    ax = sns.distplot(df[df.churn == 'No'][feature].dropna(), label = 'No Churn', alpha = .7)
    plt.legend(labels = ['No Churn', 'Churn'])
    ax1 = sns.distplot(df[df.churn == 'Yes'][feature].dropna(), label = 'Churn', alpha = .7)
    if feature == 'tenure':
        plt.set_xlabel('Tenure Length (Months)')
    else:
        plt.set_xlabel('Charge Amount ($)')
    plt.tight_layout()
    
    
def tenure_groups(df):
    if df.tenure <= 12:
        return "less_than_1"
    elif (df.tenure > 12) & (df.tenure <= 24):
        return "less_than_2"
    elif (df.tenure > 24) & (df.tenure <= 36):
        return "less_than_3"
    elif (df.tenure > 36) & (df.tenure <= 48):
        return "less_than_4"
    elif (df.tenure > 48) & (df.tenure <= 60):
        return "less_than_5"
    else:
        return "greater_than_5"
    
def tenure_group_counts(df):
    plt.figure(figsize = (20,13))
    t = sns.countplot(data = df, x = 'grouped_tenure', hue = 'churn', order = ['less_than_1', 'less_than_2', 'less_than_3', 'less_than_4', 'less_than_5', 'greater_than_5'])
    t.set_title('Churn Counts by Tenure Groups', fontsize = 20, fontweight = 'bold')
    t.set_xlabel('Tenure Groups',fontsize = 20, fontweight = 'bold', labelpad = 1.5)
    t.set_ylabel('Count', fontsize = 20, fontweight = 'bold')
    t.legend(loc = 'upper right', fontsize = 30, labels = ['No Churn', 'Churn'], edgecolor = 'black', bbox_to_anchor = (1.27, 1))
    plt.tight_layout()
    
    
def plot_numerical_averages(df, feature):
    fig = plt.figure(figsize = (13, 10))
    b = sns.barplot(data = df, x = 'grouped_tenure', y = feature, hue = 'churn', order = ['less_than_1', 'less_than_2', 'less_than_3', 'less_than_4', 'less_than_5', 'greater_than_5'])
    b.set_xlabel('Tenure Groups', fontweight = 'bold', fontsize = 20)
    b.set_ylabel(f'{feature} ($)')
    b.set_title(f'Average {feature} by Tenure Group', fontsize = 30, fontweight = 'bold')
    b.legend(fontsize = 20, loc = 'upper left', edgecolor = 'black')
    plt.tight_layout()
    

def plot_gender_dist(df):
    sns.set(style = 'whitegrid')
    sns.set_context('paper', font_scale = 2)
    fig = plt.figure(figsize = (20,10))
    plt.subplot(121)
    plt.pie(df.gender.value_counts(), labels = ['Male', 'Female'], autopct = '%.1f%%', radius = 1, textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Overall Data Gender Composition', fontweight = 'bold', fontsize = 30)
    plt.subplot(122)
    a = sns.countplot(data = df, x = 'gender', hue = 'churn')
    a.set_title('Gender Distribution by Churn', fontsize = 30, fontweight = 'bold')
    a.set_xlabel('Gender', fontweight = 'bold', fontsize = 20)
    a.set_ylabel('Count', fontweight = 'bold', fontsize = 20)
    
def plot_age_dist(df):
    sns.set(style = 'whitegrid')
    sns.set_context('paper', font_scale = 2)
    fig = plt.figure(figsize = (20,10))
    plt.subplot(121)
    plt.pie(df.seniorcitizen.value_counts(), labels = ['Senior Citizen', 'Not'], autopct = '%.1f%%', radius = 1, textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Overall Data Age Composition', fontweight = 'bold', fontsize = 30)
    plt.subplot(122)
    a = sns.countplot(data = df, x = 'seniorcitizen', hue = 'churn')
    a.set_title('Age Distribution by Churn', fontsize = 30, fontweight = 'bold')
    a.set_xlabel('seniorcitizen', fontweight = 'bold', fontsize = 20)
    a.set_ylabel('Count', fontweight = 'bold', fontsize = 20)