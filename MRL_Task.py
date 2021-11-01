#!/usr/bin/env python
# coding: utf-8

# # MRL Task Notebook
# #### Work Completed By: Tinu Daboiku (w/ help from ppl of the interwebs)
# #### Start date: 10/29/21

# In[1]:


import psycopg2
import pandas as pd

#establishing connection to postgresql
conn = psycopg2.connect(host="localhost", port = 5432, 
                        database= 'MRLTask', user="tnu")

# pulling in the data
sql =  """
select * from hftax
"""
df = pd.read_sql_query(sql, conn)
conn.close()

print(type(df))
df


# In[2]:


# plotting the data
import plotly.express as px
import plotly.graph_objects as go ## in order to layer plots

fig_hfta = px.scatter(df, x = 'time', y = 'hfta')
fig_hftanb = px.scatter(df, x= 'time', y = 'hftanb', color_discrete_sequence=['black'])
fig_hftaw = px.scatter(df, x = 'time', y = 'hftaw', color_discrete_sequence=['green'])
fig_hftamo = px.scatter(df, x = 'time', y = 'hftamo', color_discrete_sequence=['red'])

layout = go.Layout(
    title = 'HfTa-X Oxidation Kinetics',
    xaxis = dict(title = 'Time (hr)'),
    yaxis = dict(title = "Specific Mass Change (mg cm^-2)"))


fig = go.Figure(layout = layout, data = fig_hfta.data + fig_hftanb.data + 
                fig_hftaw.data + fig_hftamo.data)
fig.show()


# In[40]:


# fitting curves to mass-change equation
from scipy.optimize import curve_fit
import numpy as np

# mass-change eq
def fit_func(x, a, b):
    return (b*(x**a))

xdata = df['time'].to_numpy()

ydata_HfTa = df['hfta'].to_numpy()
popt_HfTa, pcov_HfTa = curve_fit(fit_func, xdata, ydata_HfTa)
fit_HfTa = px.line(xdata, y = fit_func(xdata, *popt_HfTa))


fig2 = go.Figure(data = fit_HfTa.data + fig.data)

fig2.show()



# In[44]:


# trying fitting withanother approach
import matplotlib.pyplot as plt
import scipy.optimize

# mass-change eq
def fit_func(x, a, b):
    return (b*(x**a))

#parameters
real_a = 0.5
real_b = 1

x = np.arange(0, 1, .001) 
y= fit_func(x, real_a, real_b)

popt, pcov = scipy.optimize.curve_fit(fit_func, x, y)
fit_a, fit_b = popt
y_fit = fit_func(x, fit_a, fit_b)

fig3, ax = plt.subplots(1)
ax.scatter(xdata, ydata_HfTa)
ax.plot(x, y_fit)


# ### still not there yet...

# In[ ]:




