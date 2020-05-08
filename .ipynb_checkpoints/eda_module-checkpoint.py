import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec


def plot_target_dist(df):
    sns.set(style = 'whitegrid')
    sns.set_context('paper', font_scale = 2)
    fig = plt.figure(figsize = (20, 10))
    plt.subplot(121)
    plt.pie(df.churn.value_counts(),labels = ['No Churn', 'Churn'], autopct = '%.1f%%', radius = 1, textprops={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Churn Outcome Pie Chart', fontsize = 30, fontweight = 'bold')
    plt.subplot(122)
    t = sns.countplot(df.churn)
    t.set_xlabel('Churn', fontweight = 'bold', fontsize = 20)
    t.set_ylabel('Count', fontweight = 'bold', fontsize = 20)
    plt.title('Churn Outcome Distributions', fontsize = 30, fontweight = 'bold')
    plt.tight_layout()
    
    
def plot_kde(df, feature):
    plt.figure(figsize = (15, 5))
    plt.title(f"KDE Plot: {feature}", fontsize = 30, fontweight = 'bold')
    ax = sns.kdeplot(df[df.churn == 'No'][feature].dropna(), label = 'No Churn', lw = 2, legend = True)
    plt.legend = True
    ax1 = sns.kdeplot(df[df.churn == 'Yes'][feature].dropna(), label = 'Churn', lw = 2, legend = True)
    if feature == 'tenure':
        plt.xlabel('Tenure Length (Months)', fontsize = 20, fontweight = 'bold')
    else:
        plt.xlabel('Charge Amount ($)', fontsize = 20, fontweight = 'bold')
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
    plt.figure(figsize = (13,10))
    t = sns.countplot(data = df, x = 'grouped_tenure', hue = 'churn', order = ['less_than_1', 'less_than_2', 'less_than_3', 'less_than_4', 'less_than_5', 'greater_than_5'])
    t.set_title('Churn Counts by Tenure Groups', fontsize = 30, fontweight = 'bold')
    t.set_xlabel('Tenure Groups',fontsize = 20, fontweight = 'bold', labelpad = 1.5)
    t.set_ylabel('Count', fontsize = 20, fontweight = 'bold')
    t.legend(loc = 'upper right', fontsize = 20, labels = ['No Churn', 'Churn'], edgecolor = 'black', bbox_to_anchor = (1.2, 1))
    plt.tight_layout()
    
    
def plot_numerical_averages(df, feature):
    fig = plt.figure(figsize = (13, 10))
    b = sns.barplot(data = df, x = 'grouped_tenure', y = feature, hue = 'churn', order = ['less_than_1', 'less_than_2', 'less_than_3', 'less_than_4', 'less_than_5', 'greater_than_5'])
    b.set_xlabel('Tenure Groups', fontweight = 'bold', fontsize = 20)
    b.set_ylabel(f'{feature} ($)', fontsize = 20, fontweight = 'bold')
    b.set_title(f'Average {feature} by Tenure Group', fontsize = 30, fontweight = 'bold')
    b.legend(fontsize = 20, loc = 'upper left', edgecolor = 'black')
    plt.tight_layout()
    

def plot_gender_dist(df):
    sns.set(style = 'whitegrid')
    sns.set_context('paper', font_scale = 2)
    fig = plt.figure(figsize = (30,10))
    
    plt.subplot(131)
    plt.pie(df.gender.value_counts(), labels = ['Male', 'Female'], autopct = '%.1f%%', radius = 1, textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Overall Data Gender Composition', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(132)
    a = sns.countplot(data = df, x = 'gender', hue = 'churn')
    a.set_title('Gender Distribution by Churn', fontsize = 30, fontweight = 'bold')
    a.set_xlabel('Gender', fontweight = 'bold', fontsize = 20)
    a.set_ylabel('Count', fontweight = 'bold', fontsize = 20)
    
    plt.subplot(133)
    x = sns.violinplot('gender', 'monthlycharges', 'churn', df, split = True)
    x.set_title('Violin Plot: Monthly Charges by Gender', fontsize = 30, fontweight = 'bold')
    x.set_xlabel('Gender', fontsize = 20, fontweight = 'bold')
    x.set_ylabel('Monthly Charges ($)', fontweight = 'bold', fontsize = 20)
    
    plt.tight_layout()
    
    
def plot_age_dist(df):
    
    fig = plt.figure(figsize = (30,10))
    
    plt.subplot(131)
    plt.pie(df.seniorcitizen.value_counts(), labels = ['Non-Senior Citizen', 'Senior'], autopct = '%.1f%%', radius = 1, textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Age Composition of Overall Data', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(132)
    g = df.copy()
    g = g.groupby('seniorcitizen')['churn'].value_counts().to_frame()
    g = g.rename({'churn':'pct_total'}, axis = 1).reset_index()
    g['pct_total'] = (g['pct_total']/len(df)) * 100
    t = sns.barplot('seniorcitizen', y = 'pct_total', hue = 'churn', data = g)
    t.set_title('Churn % by Age', fontsize = 30, fontweight = 'bold')
    t.set_xlabel('')
    t.set_ylabel('Percentage of Customers', fontsize = 20, fontweight = 'bold')
    t.set_xticklabels(labels = ['Non-Senior Citizen', 'Senior Citizen'], fontweight = 'bold', fontsize = 20)
    
    plt.subplot(133)
    x = sns.violinplot('seniorcitizen', 'monthlycharges', 'churn', df, split = True)
    x.set_title('Violin Plot: Monthly Charges by Age', fontsize = 30, fontweight = 'bold')
    x.set_xlabel('')
    x.set_ylabel('Monthly Charges ($)', fontsize = 20, fontweight = 'bold')
    x.set_xticklabels(labels = ['Non-Senior Citizen', 'Senior Citizen'], fontsize = 20, fontweight = 'bold')
    
    plt.tight_layout()
    
def plot_partner_dependents(df):
    
    fig = plt.figure(figsize = (25,25))
    x = df.copy()
    plt.subplot(321)
    plt.pie(df.partner.value_counts(), labels = ['No Partner', 'Partner'], autopct = '%.1f%%', radius = 1, textprops = {'fontsize':20, 'fontweight':'bold'}, startangle = 90)
    plt.title('Partner Composition of Overall Data', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(322)
    plt.pie(df.dependents.value_counts(), labels = ['No Dependents', 'Dependents'], autopct = '%.1f%%', radius = 1,  textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Dependent Composition of Overall Data', fontsize = 30, fontweight = 'bold')
    
    plt.subplot(323)
    x = df.copy()
    x = x.groupby('partner')['churn'].value_counts().to_frame()
    x = x.rename({'churn':'pct_total'}, axis = 1).reset_index()
    x['pct_total'] = (x['pct_total']/len(df)) * 100
    u = sns.barplot('partner', y = 'pct_total', hue = 'churn', data = x)
    u.set_title('Churn % by Partner', fontweight = 'bold', fontsize = 30)
    u.set(xticklabels = ['No Partner', 'Partner'])
    u.set_xlabel('')
    u.set_ylabel('Percentage of Total', fontweight = 'bold', fontsize = 20)
    
    plt.subplot(324)
    y = df.copy()
    y = y.groupby('dependents')['churn'].value_counts().to_frame()
    y = y.rename({'churn':'pct_total'}, axis = 1).reset_index()
    y['pct_total'] = (y['pct_total']/len(df)) * 100
    v = sns.barplot('dependents', y = 'pct_total', hue = 'churn', data = y)
    v.set_title('Churn % by Dependents', fontweight = 'bold', fontsize = 30)
    v.set(xticklabels = ['No Dependents', 'Dependents'])
    v.set_xlabel('')
    v.set_ylabel('')
    
    plt.subplot(325)
    y = sns.violinplot('partner', 'monthlycharges', 'churn', df, split = True)
    y.set_title('Violin Plot: Monthly Charges by Partner', fontweight = 'bold', fontsize = 30)
    y.set_xticklabels(['Partner', 'No Partner'])
    y.set_xlabel('')
    
    plt.subplot(326)
    z = sns.violinplot('dependents', 'monthlycharges', 'churn', df, split = True)
    z.set_title('Violin Plot: Monthly Charges by Dependents', fontweight = 'bold', fontsize = 30)
    z.set_xticklabels(['No Dependents', 'Dependents'])
    z.set_xlabel('')
    z.set_ylabel('Monthly Charges', fontweight = 'bold', fontsize = 20)

def plot_services(df):
    
    phone_only = df[(df.phoneservice == 'Yes') & (df.internetservice == 'No')]
    
    fig = plt.figure(figsize = (30, 20))

    plt.subplot(231)
    plt.pie(phone_only.churn.value_counts(), labels = ['No Churn', 'Churn'], autopct = '%.1f%%', radius = 1, textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customer Churn - Phone Service Only', fontsize = 30, fontweight = 'bold')
    
    plt.subplot(232)
    z = df.copy()
    z = z.groupby('phoneservice')['churn'].value_counts().to_frame()
    z = z.rename({'churn':'pct_total'}, axis = 1).reset_index()
    z['pct_total'] = (z['pct_total']/len(df)) * 100
    a = sns.barplot('phoneservice', y = 'pct_total', hue = 'churn', data = z)
    a.set_title('% Churn by Phone Service', fontsize = 30, fontweight = 'bold')
    a.set(xticklabels = ['No Phone', 'Phone'])
    a.set_xlabel('')
    a.set_ylabel('% of Customers', fontweight = 'bold')
    
    
    plt.subplot(233)
    v1 = sns.violinplot('phoneservice', 'monthlycharges', 'churn', df, split = True)
    v1.set_title('Violin Plot: Monthly Charges by Phone Service', fontsize = 30, fontweight = 'bold')
    v1.set_xlabel('')
    v1.set_ylabel('Monthly Charges ($)', fontsize = 20, fontweight = 'bold')
    v1.set(xticklabels = ['No Phone', 'Phone'])
    
    plt.subplot(234)
    plt.pie(df.multiplelines.value_counts(), labels = ['Singular Line', 'Multiple Lines', 'No Phone Service'], autopct = '%.1f%%', radius = 1, textprops = {'fontweight':'bold', 'fontsize': 20}, startangle = 180)
    plt.title('Customer Churn - Qty. of Lines', fontsize = 30, fontweight = 'bold')
    

    
    plt.subplot(235)
    bb = df.copy()
    bb = bb.groupby('multiplelines')['churn'].value_counts().to_frame()
    bb = bb.rename({'churn':'pct_total'}, axis = 1).reset_index()
    bb['pct_total'] = (bb['pct_total']/len(df)) * 100
    c = sns.barplot('multiplelines', y = 'pct_total', hue = 'churn', data = bb)
    c.set(xticklabels = ['Singular Line', 'No Phone Service', 'Multiple Lines'])
    c.set_title('')
    c.set_xlabel('')
    c.set_ylabel('% of Customers', fontweight = 'bold', fontsize = 20)
    c.set_title('% Churn by Phone Line Qty.', fontsize = 30, fontweight = 'bold')
    
    plt.subplot(236)
    v = sns.violinplot('multiplelines', 'monthlycharges', 'churn', df, split = True)
    v.set_title('Violin Plot: Monthly Charges by Line Quantity', fontweight = 'bold', fontsize = 30)
    v.set_xlabel('')
    v.set_ylabel('Monthly Charges ($)', fontweight = 'bold')
    v.set(xticklabels = ['No Phone Service', 'Singular Line', 'Multiple Lines'])
    
    fig.suptitle('Phone Services -  Line Quantity', fontweight = 'bold', fontsize = 40)
    
def plot_internet_services(df):
    
    copy = df.copy()
    fig = plt.figure(figsize = (30, 10))

    plt.subplot(131)
    plt.pie(copy.internetservice.value_counts(), labels = ['Fiber Optic', 'DSL', 'No Internet'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight': 'bold'}, startangle = 180)
    plt.title('Internet Service Composition of Customers', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(132)
    copy = copy.groupby('internetservice')['churn'].value_counts().to_frame()
    copy = copy.rename({'churn':'pct_total'}, axis = 1).reset_index()
    copy['pct_total'] = (copy['pct_total']/len(df)) * 100
    d = sns.barplot('internetservice', y = 'pct_total', hue = 'churn', data = copy)
    d.set_title('% Churn by Internet Service', fontweight= 'bold', fontsize = 30)
    d.set_xlabel('')
    d.set_ylabel('% of Customers', fontweight = 'bold', fontsize = 20)
    d.set(xticklabels = ['DSL', 'Fiber Optic', 'No Internet Service'])
    
    plt.subplot(133)
    e = sns.violinplot('internetservice', 'monthlycharges', 'churn', df, split = True)
    e.set_title('Violin Plot: Monthly Charges by Internet Service', fontweight = 'bold', fontsize = 30)
    e.set_xlabel('')
    e.set(xticklabels = ['DSL', 'Fiber Optic', 'No Internet Service'])
    e.set_ylabel('Monthly Charges($)', fontweight = 'bold', fontsize = 30)

    fig.tight_layout()
    

def plot_services(df):
    copy = df[df.internetservice != 'No']
    
    fig = plt.figure(figsize = (40, 15))
    
    plt.subplot(261)
    plt.pie(copy.onlinesecurity.value_counts(), labels = ['Yes', 'No'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Online Security', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(262)
    plt.pie(copy.onlinebackup.value_counts(), labels = ['Yes', 'No'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Online Backup', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(263)
    plt.pie(copy.deviceprotection.value_counts(), labels = ['Yes', 'No'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Device Protection', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(264)
    plt.pie(copy.techsupport.value_counts(), labels = ['Yes', 'No'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Tech Support', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(265)
    plt.pie(copy.streamingtv.value_counts(), labels = ['Yes', 'No'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Streaming TV', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(266)
    plt.pie(copy.streamingmovies.value_counts(), labels = ['Yes', 'No'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Customers w/ Movie Streaming', fontweight = 'bold', fontsize = 25)
    
    plt.subplot(267)
    copy1 = copy[copy.onlinesecurity == 'Yes']
    plt.pie(copy1.churn.value_counts(), labels = ['No Churn', 'Churn'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Online Security - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(268)
    copy2 = copy[copy.onlinebackup == 'Yes']
    plt.pie(copy2.churn.value_counts(), labels = ['No Churn', 'Churn'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Online Backup - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(269)
    copy3 = copy[copy.deviceprotection == 'Yes']
    plt.pie(copy3.churn.value_counts(), labels = ['No Churn', 'Churn'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Device Protection - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(2,6,10)
    copy4 = copy[copy.techsupport == 'Yes']
    plt.pie(copy4.churn.value_counts(), labels = ['No Churn', 'Churn'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Tech Support - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(2,6,11)
    copy5 = copy[copy.streamingtv == 'Yes']
    plt.pie(copy5.churn.value_counts(), labels = ['No Churn', 'Churn'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Streaming TV - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.subplot(2,6,12)
    copy6 = copy[copy.streamingmovies == 'Yes']
    plt.pie(copy6.churn.value_counts(), labels = ['No Churn', 'Churn'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'})
    plt.title('Streaming Movies - Churn %', fontsize = 25, fontweight = 'bold')
    
    plt.tight_layout()
    
    
def plot_service_charges(df):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (40, 10), sharex = True, sharey = True)
    
    copy1 = df[df.onlinesecurity != 'No internet service']
    a = sns.violinplot('onlinesecurity', 'monthlycharges', 'churn', copy1, split = True, ax = axes[0])
    a.set_title('Online Security', fontsize = 30, fontweight = 'bold')
    a.set_xlabel('Has Service', fontweight = 'bold', fontsize = 20)
    a.set_ylabel('Monthly Charges ($)', fontweight = 'bold', fontsize = 20)
    a.set_yticklabels(a.get_yticks(), size = 25, weight = 'bold')
    a.legend(loc = 'upper left', )
    a.legend_.set_title('Churn', prop = {'size': 20, 'weight':'bold'})

    copy2 = df[df.onlinebackup != 'No internet service']
    b = sns.violinplot('onlinebackup', 'monthlycharges', 'churn', copy2, split = True, ax = axes[1])
    b.set_title('Online Backup', fontweight = 'bold', fontsize = 30)
    b.set_xlabel('Has Service', fontweight = 'bold', fontsize = 20)
    b.set_ylabel('')
    b.legend_.remove()
    
    copy3 = df[df.deviceprotection != 'No internet service']
    c = sns.violinplot('onlinebackup', 'monthlycharges', 'churn', copy3, split = True, ax = axes[2])
    c.set_title('Device Protection', fontweight = 'bold', fontsize = 30)
    c.set_xlabel('Has Service', fontweight = 'bold', fontsize = 20)
    c.set_ylabel('')
    c.legend_.remove()
    
    copy4 = df[df.techsupport != 'No internet service']
    d = sns.violinplot('techsupport', 'monthlycharges', 'churn', copy3, split = True, ax = axes[3])
    d.set_title('Tech Support', fontsize = 30, fontweight = 'bold')
    d.set_xlabel('Has Service', fontsize = 20, fontweight = 'bold')
    d.set_ylabel('')
    d.legend_.remove()
    
    copy5 = df[df.streamingtv != 'No internet service']
    e = sns.violinplot('streamingtv', 'monthlycharges', 'churn', copy4, split = True, ax = axes[4])
    e.set_title('Streaming TV', fontsize = 30, fontweight = 'bold')
    e.set_xlabel('Has Service', fontweight = 'bold', fontsize = 20)
    e.set_ylabel('')
    e.legend_.remove()
    
    copy5 = df[df.streamingmovies != 'No internet service']
    f = sns.violinplot('streamingmovies', 'monthlycharges', 'churn', copy5, split = True, ax = axes[5])
    f.set_title('Streaming Movies', fontweight = 'bold', fontsize = 30)
    f.set_xlabel('Has Service', fontsize = 20, fontweight = 'bold')
    f.set_ylabel('')
    f.legend(loc = 'upper left')
    f.legend_.set_title('Churn', prop = {'size': 20, 'weight':'bold'})
    
    
def plot_contracts(df):
    
    copy = df.copy()
    
    plt.figure(figsize = (30, 10))
    
    plt.subplot(131)
    plt.pie(copy.contract.value_counts(), labels = ['Monthly', '1-Year', '2-Year'], autopct = '%.1f%%', textprops = {'fontweight':'bold', 'fontsize': 20})
    plt.title('Customer Contract Composition', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(132)
    plt.title('Churn % by Contract Type', fontsize = 30, fontweight = 'bold')
    copy = copy.groupby('contract')['churn'].value_counts().to_frame()
    copy = copy.rename({'churn':'pct_total'}, axis = 1).reset_index()
    copy['pct_total'] = (copy['pct_total']/len(df)) * 100
    a = sns.barplot('contract', y = 'pct_total', hue = 'churn', data = copy)
    a.set_title('% Churn - Contract Type', fontsize = 30, fontweight = 'bold')
    a.set(xticklabels = ['Monthly', '1-Year', '2-Year'])
    a.set_xlabel('')
    a.set_ylabel('% of Customers', fontweight = 'bold')
    
    plt.subplot(133)
    b = sns.violinplot('contract', 'monthlycharges', 'churn', df, split = True)
    b.set_title('Violin Plot: Monthly Charge - Contract Types', fontweight = 'bold', fontsize = 30)
    b.set_xlabel('')
    b.set_ylabel('Monthly Charges ($)', fontweight = 'bold', fontsize = 20)
    b.set(xticklabels = ['Monthly', '1-Year', '2-Year'])
    b.legend(loc = 'upper left')
    b.legend_.set_title('Churn', prop = {'weight':'bold', 'size':20})
    
def plot_paperless(df):
    
    copy = df.copy()
    
    plt.figure(figsize = (30, 10))
    
    plt.subplot(131)
    plt.pie(copy.paperlessbilling.value_counts(), labels = ['Paperless', 'Not Paperless'], autopct = '%.1f%%', textprops = {'fontweight':'bold', 'fontsize': 20})
    plt.title('Customer Paperless Billing Composition', fontweight = 'bold', fontsize = 30)
    
    plt.subplot(132)
    plt.title('Churn % by Billing Type', fontsize = 30, fontweight = 'bold')
    copy = copy.groupby('paperlessbilling')['churn'].value_counts().to_frame()
    copy = copy.rename({'churn':'pct_total'}, axis = 1).reset_index()
    copy['pct_total'] = (copy['pct_total']/len(df)) * 100
    a = sns.barplot('paperlessbilling', y = 'pct_total', hue = 'churn', data = copy)
    a.set_title('% Churn - Paperless Billing', fontsize = 30, fontweight = 'bold')
    a.set(xticklabels = ['Paperless', 'Not Paperless'])
    a.set_xlabel('')
    a.set_ylabel('% of Customers', fontweight = 'bold')
    
    plt.subplot(133)
    b = sns.violinplot('paperlessbilling', 'monthlycharges', 'churn', df, split = True)
    b.set_title('Violin Plot: Monthly Charge - Contract Types', fontweight = 'bold', fontsize = 30)
    b.set_xlabel('')
    b.set_ylabel('Monthly Charges ($)', fontweight = 'bold', fontsize = 20)
    b.set(xticklabels = ['Paperless', 'Not Paperless'])
    b.legend(loc = 'upper right')
    b.legend_.set_title('Churn', prop = {'weight':'bold', 'size':20})
    
    plt.tight_layout()
    
    
def plot_pay_methods(df):
    
    copy = df.copy()
    
    plt.figure(figsize = (30, 10))
    
    plt.subplot(131)
    plt.pie(copy.paymentmethod.value_counts(), labels = ['E-Check', 'Mail Check' , 'Bank Transfer (Auto)', 'Credit Card (Auto)'], autopct = '%.1f%%', textprops = {'fontsize':20, 'fontweight':'bold'}, startangle = -90)
    plt.title('Customer Payment Method Composition', fontsize = 30, fontweight = 'bold')
    
    plt.subplot(132)
    copy = copy.groupby('paymentmethod')['churn'].value_counts().to_frame()
    copy = copy.rename({'churn':'pct_total'}, axis = 1).reset_index()
    copy['pct_total'] = (copy['pct_total']/len(df))*100
    a = sns.barplot('paymentmethod', 'pct_total', 'churn', data = copy)
    a.set_title('% Churn - Payment Methods', fontsize = 30, fontweight = 'bold')
    a.set_xlabel('')
    a.set_ylabel('% of Customers', fontsize = 20, fontweight = 'bold')
    a.set_xticklabels(a.get_xticklabels(), rotation = 45)
    
    plt.subplot(133)
    c = sns.violinplot('paymentmethod', 'monthlycharges', 'churn', df, split = True)
    c.set_title('Violin Plot: Monthly Charge - Payment Methods', fontsize = 30, fontweight = 'bold')
    c.set_xlabel('')
    c.set_ylabel('Monthly Charges ($)', fontweight = 'bold', fontsize = 30)
    c.set_xticklabels(a.get_xticklabels(), rotation = 45)
    
    plt.tight_layout()