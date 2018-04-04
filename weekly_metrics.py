# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:12:44 2018

@author: hmamin
"""

import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

# Load and generate data
x = np.arange(0,101,1)
y = np.random.randn(101) * x

df = pd.read_csv(r'service_repair_costs.csv',\
                   usecols=range(12),
                   na_values='_',
                   parse_dates=['Date'])
df['Total'] = df['Total'].str[1:].replace(',', '', regex=True).fillna(0).astype(float)
df['Month'] = df['Date'].dt.month                
print(df.dtypes)

# Filter df to limit to past 30 days
today = datetime.now()
start_day = today - timedelta(days=90)
month_df = df[(df['Date'] > start_day) & (df['Date'] < today)]
print(month_df.shape)
monthly_vendor_sum = month_df['Total'].groupby(month_df['Vendor']).sum()\
                             .sort_values(ascending=False)
monthly_site_costs = month_df['Total'].groupby(month_df['Project']).sum()\
                             .sort_values(ascending=False)


# Create and run Dash app
auth_info = [['engiestorage', 'greenstation']]
app = dash.Dash('auth')
auth = dash_auth.BasicAuth(app, auth_info)
stylesheet_url = 'https://codepen.io/hmamin_gcn/pen/PQRLWQ.css'
app.css.append_css({'external_url': stylesheet_url})
app.layout = html.Div([
    html.H1('Weekly Metrics'),
    html.H3(today.strftime('%B %d, %Y')),

    dcc.Graph(
        id = 'graph3',
        figure = {
            'data': [{'x': monthly_vendor_sum.index,
                      'y': monthly_vendor_sum, 'type': 'bar', 
                      'name': 'inverter'}],
            'layout': {'title': 'Contractor Spending',
                       'plot_bgcolor': '#F2F2F2',
                       'paper_bgcolor': '#F2F2F2'}
        },
            style={'width': '80%'},
        ),

    dcc.Graph(
        id='graph4',
        figure = {
                'data': [{'x': monthly_site_costs.index,
                          'y': monthly_site_costs,
                          'type': 'bar',
                          'name': 'Service Costs by Site'}],
                'layout': {'title': 'Service Costs by Site',
                           'plot_bgcolor': '#F2F2F2',
                           'paper_bgcolor': '#F2F2F2',
                           'xaxis': {'title': 'Project', 'tickangle': '45'},
                            'yaxis': {'title': 'Cost ($)'}
                            }
                },
        style={'width': '80%', 'min-width': '600px', 'margin-bottom': '80px'
               })
    ],
style={'width': '80%', 'max-width': '1000px', 'margin': '0 auto'})


if __name__ == '__main__':
    # Set debug=True to run, threaded=True to quit
    #app.run_server(threaded=True, port=8001)
    app.run_server(debug=True, port=8001)
