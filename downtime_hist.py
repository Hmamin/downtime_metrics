# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:09:08 2018

@author: hmamin
"""

import os
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly import figure_factory as ff
import numpy as np
import pandas as pd
import gspread as gs
from oauth2client.service_account import ServiceAccountCredentials
from google.auth.service_account import Credentials
from gspread_dataframe import get_as_dataframe
import json


def auth_gdrive():
    '''Read in environment variables from Heroku
    to authorize access to Google drive.'''
    scope = ['https://spreadsheets.google.com/feeds']
    json_contents = json.load(os.environ['json_content'])
    creds = Credentials.from_service_account_file(json_contents)
    
    #creds = ServiceAccountCredentials.from_json(os.environ['json_content'])
    
    #email = os.environ['google_client_email']
    #key = os.environ['google_private_key']
    #creds = ServiceAccountCredentials(email, key, scope)
    client = gs.authorize(creds)    
    return client


def get_auth():
    '''Read username and password in from environment variables set in
    Heroku.
    '''
    username = os.environ['username']
    password = os.environ['password']
    return [username, password]
    
 
def time_from_percentile(p, timeseries):
    '''Pass in percentile value and unsorted timeseries.
    Return corresponding time value, sorted series, and # of rows.
    '''
    rows = len(timeseries)
    times_sorted = timeseries.sort_values()
    time = times_sorted.iloc[round(p*rows/100)]
    return round(time, 2), times_sorted, rows


#==============================================================================
# def filter_df(df, gen_val=None, site_or_tower=None, outlier=False):
#     '''Filter dataframe based on user-selected parameters on page.'''
#     # Filter df on user-selected options - alternative to storing as JSON.
#     if outlier is not False:
#         try:
#             outlier = float(outlier)
#         except Exception as e:
#             print(e)
#             outlier = 800.0
#             pass
#         df = df.loc[hw_df['Total Time'] < outlier, :]
#     
#     # Filter df to selected generation.
#     if gen_val is not None:
#         if gen_val == 'gen3':
#             df = df.loc[df['Model Type'].isin(gen3_names), :]
#         elif (gen_val == 'gen2') | (gen_val == 'gen1'):
#             df = df.loc[df['Model Type'] == gen_val, :]
#     
#     # Filter df to selected ticket type (sitewide or single tower).
#     site_condition = (df['Symptom (red text=guess)']\
#         .isin(site_symptoms)) | (df['DPG #'].isin(site_dpg)) |\
#                          (df['# of Systems Affected'] > 1)
#     if site_or_tower == 'site':
#         df = df.loc[site_condition, :]
#     elif site_or_tower == 'tower':
#         df = df.loc[~site_condition, :]
#     return df
# 
#==============================================================================
txt_auth = get_auth()
#creds = auth_gdrive()

app = dash.Dash('auth')
auth = dash_auth.BasicAuth(app, txt_auth)
server = app.server

# App layout
app.layout = html.Div([
        html.H3('test')
])
    
if __name__ == '__main__':
    app.run_server(debug=True)

#==============================================================================
# # Authorize google sheets and load live tickets into dataframe
# client = auth_gdrive()
# field_metrics = client.open('Field Services Metrics & Uptime')
# tickets = field_metrics.worksheet('LIVE Tickets')
# df = get_as_dataframe(tickets, skiprows=2, usecols=list(range(21)),
#                       parse_dates=[3, 4, 5], evaluate_formulas=True,
#                       na_values=['#DIV/0!', '#VALUE!'])
# 
# # Get total operating time from fleet metrics sheet.
# fleet_metrics = field_metrics.worksheet('Fleet Metrics')
# total_site_time = fleet_metrics.cell(5,28).value
# total_tower_time = fleet_metrics.cell(5,29).value
# 
# # Filter df to include only hardware-related issues
# time_delta_cols = ['Total Time', 'time past SLA', 'Downtime of ticket']
# hw_issues = ['battery', 'cbox', 'connectivity', 'enclosure',\
#              'inverter', 'site controller', 'HW firmware', 'alg software']
# four_non_issues = ['166', '259', '260', '754']
# site_symptoms = ['No Controller Connectivity', 'Inaccurate \
#                  Data (Netops)', 'No Site Data (Operational)']
# site_dpg = ['building', 'gs1', 'gs2', 'gs3', 'pv']
# hw_df = df.loc[(df['Point of Failure'].isin(hw_issues)) &\
#        (~df['Work Order # (ticket #, FD=Freshdesk)'].isin(four_non_issues))]
# dtype_dict = hw_df.dtypes.to_dict()
# rows, cols = hw_df.shape
# 
# # Create global lists to filter df on through callbacks
# gen3_names = ['gen3', 'gen3N', 'gen3T', 'gen3L', 'gen3-e2']
# site_symptoms = ['No Controller Connectivity', 'Inaccurate Data (Netops)',\
#                  'No Site Data (Operational)']
# site_dpg = ['building', 'gs1', 'gs2', 'gs3', 'pv']
# 
# # Convert Total Time column from timedelta to float
# hw_df['Total Time'] = hw_df['Total Time'] / pd.Timedelta(hours=1)
# p_time_98, hw_times, rows_98 = time_from_percentile(98, hw_df['Total Time'])
# 
# # Create list of site dicts for dropdown labels
# dropdown_labels = [{'label': site, 'value': site} for site in\
#                    sorted(df['Site'].unique())]
#  
# # Read in auth info
# file = 'hist_auth.txt'
# txt_auth = get_auth(file)
# 
# # Create app login
# app = dash.Dash('auth')
# auth = dash_auth.BasicAuth(app, txt_auth)
# 
# # Link to default CSS
# external_css = 'https://codepen.io/hmamin_gcn/pen/YeOKvd.css'
# app.css.append_css({'external_url': external_css})
# 
# # Create app layout
# app.layout = html.Div([
#         html.H1('Hardware Tickets'),
#                
#         # Row containing outlier cutoff and operating time
#         html.Div([
#             html.Div([
#                 html.H6('Outlier cutoff (hours)'),
#                 dcc.Input(
#                         id='outlier', value=800, type='text'
#                         )],
#                 style={'width': '40%', 'box-sizing': 'border-box', 
#                        'min-width': '150px', 'margin': '0px 3px'}
#             ),
#             html.Div(id='operating_time')],
#         style={'display': 'flex', 'flex-wrap': 'wrap'}),
# 
#         # Row containing dropdowns for downtime type and model type
#         html.Div([
#             html.Div([
#                 html.H6('Downtime type'),
#                 dcc.Dropdown(
#                         id='site_or_tower',
#                         options=[
#                                 {'label': 'All', 'value': 'all'},
#                                 {'label': 'Tower', 'value': 'tower'},
#                                 {'label': 'Site', 'value': 'site'}
#                                 ],
#                         multi=False,
#                         value='all'
#                         )],
#                 style={'width': '40%', 'margin': '3px',
#                        'box-sizing': 'border-box', 'min-width': '150px'}
#             ),
#             html.Div([
#                 html.H6('Model type'),
#                 dcc.Dropdown(
#                         id='gen_type',
#                         options=[
#                                 {'label': 'All', 'value': 'all'},
#                                 {'label': 'Gen 1', 'value': 'gen1'},
#                                 {'label': 'Gen 2', 'value': 'gen2'},
#                                 {'label': 'Gen 3', 'value': 'gen3'}
#                                 ],
#                         multi=False,
#                         value='all'
#                         )],
#                 style={'width': '55%', 'margin': '3px',
#                        'box-sizing': 'border-box', 'min-width': '150px'}
#             )],
#             style={'display': 'flex', 'flex-wrap': 'wrap'}
#         ),
#         html.H6('Bin size (hours):', style={'margin': '20px 0px 10px'}),
#         dcc.Slider(id='bin_size', min=1, max=40, value=16, 
#            marks={i: str(i) for i in range(2, 41, 2)}),
#         html.Div(style={'height': '60px'}),
#         html.Div(id='output_hist'),
#         html.Div(id='output_distplot'),
#         html.Div(id='site_boxplot'),
#         html.Div(id='output_percentile', style={'margin-top': '20px'}),
#         html.Div([
#             html.Div([
#                     html.H6('Percentile'),
#                     dcc.Input(id='percentile_choice', type='text', value=50)],
#                     style={'width': '30%', 'margin': '0px 45px 0px 75px',
#                            'box-sizing': 'border-box'}),
#             html.Div([
#                     html.H6('Corresponding Time', 
#                             style={'margin-bottom': '10px'}),
#                     html.Div(id='output_time')],
#                     style={'width': '45%'})],
#             style={'display': 'flex', 'flex-wrap': 'wrap'}),
#         html.Div(id='output_graph'),
#         html.Div(id='site_graph', style={'margin-bottom': '20px'}),
#         dcc.Dropdown(id='site_dropdown', options=dropdown_labels, multi=False,
#            value=dropdown_labels[0]['label']),
#         html.Div(id='footer', style={'width': '100%', 'height': '100px'})
#     ],
#     style={'width': '100%', 'max-width': '950px', 'min-width': '500px'}, 
# id='main_div')
# 
# 
# @app.callback(
#         Output('operating_time', 'children'),
#         [Input('site_or_tower', 'value')]
#         )
# def update_operating_time(site_tower_val):    
#     '''Update displayed operating time based on user-selected values.'''
#     if site_tower_val == 'all' or site_tower_val == 'tower':
#         time = total_tower_time
#         time_type = 'Tower'
#     else:
#         time = total_site_time
#         time_type = 'Site'
#     output_txt = html.H6([
#                 f'{time_type} operating time (business days)',
#                 html.P(time)
#             ],
#             style={'width': '100S%'})
#     return output_txt
# 
# 
# @app.callback(
#     Output('output_hist', 'children'),
#     [Input('outlier', 'value'),
#     Input('bin_size', 'value'),
#     Input('gen_type', 'value'),
#     Input('site_or_tower', 'value')])
# def update_hist(outlier_val, bin_size_val, gen_num, site_tower_val):
#     '''Filter df from user selections and update histogram.'''
#     df = filter_df(hw_df, gen_val=gen_num, site_or_tower=site_tower_val, 
#                    outlier=outlier_val)
#     bin_width = int(bin_size_val)
#         
#     # Create trace and layout, then pass to figure in graph object
#     trace1 = go.Histogram(
#                 x=df['Total Time'],
#                 xbins={'start': 0,
#                        'end': outlier_val,
#                        'size': bin_width},
#                 marker=go.Marker(color='rgb(234, 65, 65)'),
#                 opacity=0.8
#                 )
#     layout1 = go.Layout(title='Hardware Tickets',
#                         height=500,
#                         width='100vw',
#                         xaxis={'title': 'Downtime (hours)'},
#                         yaxis={'title': 'Ticket Count'},
#                         bargap=0.1
#                         )
#     g = dcc.Graph(
#             id='output_histogram',
#             figure=go.Figure(
#                     data=[trace1],
#                     layout=layout1
#                     )
#             )
#     return g 
# 
# 
# @app.callback(
#         Output('output_distplot', 'children'),
#         [Input('bin_size', 'value'),
#         Input('outlier', 'value'),
#         Input('gen_type', 'value'),
#         Input('site_or_tower', 'value')]
#         )
# def update_distplot(bin_size_val, outlier_val, gen_num, site_tower_val):
#     bin_width = int(bin_size_val)
#     df = filter_df(hw_df, gen_val=gen_num, site_or_tower=site_tower_val, 
#                    outlier=outlier_val)
#     times = df['Total Time'].tolist()
#     
#     # Create figure, edit layout, then pass to graph
#     fig = ff.create_distplot(
#                         [times],
#                         bin_size=bin_width,
#                         group_labels=['Downtime per Ticket'],
#                         colors=['rgb(184, 119, 216)']
#                         )
#     fig['layout'].update(title='Downtime Density Estimation', showlegend=False)
#     g = dcc.Graph(
#             id='output_dist',
#             figure=fig,
#             style={'height': '700px'}
#             )
#     return g  
# 
# 
# @app.callback(
#         Output('output_percentile', 'children'),
#         [Input('percentile_choice', 'value')]
#         )
# def graph_percentile(percentile_val):
#     '''Update graph showing percentile of user-entered downtime value.'''
#     try:
#         p = int(percentile_val)
#     except Exception as e:
#         print(e)
#         p = 1
#     p_time, sorted_time, rows = time_from_percentile(p, hw_df['Total Time'])
#     y = [100*i/rows_98 for i in range(1, rows_98+1)]
#     trace1 = go.Scatter(x=sorted_time.loc[sorted_time<=p_time_98], y=y, 
#                         mode='lines', line=go.Line(color='rgb(232, 161, 20)', 
#                         shape='spline'), fill='tozeroy')
#     trace2 = go.Bar(x=[p_time], y=[p], width=[2,500])
#     layout1 = go.Layout(title='Downtime Density', height=500,
#                        xaxis={'title': 'Downtime', 'dtick': 40, 
#                               'tickangle': 45, 'showgrid': False, 
#                               'range': [0, p_time_98]},
#                        yaxis={'title': 'Percentile', 'dtick': 10, 
#                               'showgrid': False},
#                        showlegend=False)
#     g = dcc.Graph(
#             id='percentile_graph',
#             figure=go.Figure(
#                 data=[trace1, trace2],
#                 layout=layout1
#             )
#         )
#     return g
# 
# 
# @app.callback(
#         Output('output_time', 'children'),
#         [Input('percentile_choice', 'value')])
# def update_time(p_val):
#     error = 'Please enter an integer between 1 and 99.'
#     try:
#         p = int(p_val)
#         if (p > 99) | (p < 1):
#             return error
#     except Exception as e:
#         return error
#     else:
#         return time_from_percentile(p, hw_df['Total Time'])[0]
# 
# 
# @app.callback(
#     Output('output_graph', 'children'),
#     [Input('outlier', 'value'),
#     Input('gen_type', 'value'),
#     Input('site_or_tower', 'value')]
#     )
# def graph_scatter(outlier_val, gen_num, site_tower_val):
#     '''Create scatter plot of ticket downtimes after removing
#     outliers based on input value.
#     '''
#     try:
#         outlier_val = float(outlier_val)
#     except Exception as e:
#         print(e)
#         outlier_val = float(560)
#     df = filter_df(hw_df, gen_val=gen_num, site_or_tower=site_tower_val, 
#                    outlier=outlier_val)
#     
#     # Create trace and layout, then pass to graph figure.
#     trace1 = go.Scatter(
#                 x=df.iloc[:,5],
#                 y=df['Total Time'],
#                 name='Software-Related Downtime',
#                 mode='markers',
#                 opacity=0.8
#              )
#     layout1 = go.Layout(title='Ticket Downtimes',
#                                height=500,
#                                width='80vw',
#                                xaxis={'title': 'Ticket Close Date'},
#                                yaxis={'title': 'Downtime (work hours)'})
#     g = dcc.Graph(
#             id = 'downtime_graph',
#             figure = go.Figure(
#                      data=[trace1],
#                      layout=layout1)
#             )
#     return g
#  
#     
# @app.callback(
#         Output('site_graph', 'children'),
#         [Input('site_dropdown', 'value')]
#         )
# def graph_single_site(site):
#     '''Graph downtimes for software-related tickets at selected site.'''
#     site_df = hw_df.loc[df['Site'] == site]
#     site_df = site_df.sort_values('Ticket Opened')
#     
#     # Set trace and layout, then pass them to figure in graph.
#     trace1 = go.Scatter(
#         x=site_df['Ticket Opened'],
#         y=site_df['Total Time'],
#         mode='lines',
#         line=dict(width=4, color='rgb(96, 219, 166)'),
#         fill='tozeroy'
#         )
#     layout1 = go.Layout(title='Tickets at {}'.format(site),
#                   height=500,
#                   xaxis={'title': 'Date of Failure'},
#                   yaxis={'title': 'Downtime (work hours)'}
#                   )
#     g = dcc.Graph(
#             id='selected_site_graph',
#             figure=go.Figure(
#                 data=[trace1],
#                 layout=layout1
#                 )
#             )
#     return g
# 
# 
# @app.callback(
#         Output('site_boxplot', 'children'),
#         [Input('gen_type', 'value'),
#         Input('site_or_tower', 'value'),
#         Input('outlier', 'value')
#         ])
# def update_boxplot(gen_num, site_tower_val, outlier_val):
#     '''Filter df to update boxplots.'''
#     df = filter_df(hw_df, gen_val=gen_num, site_or_tower=site_tower_val)
#     traces = []
#     if gen_num == 'all' or gen_num == 'gen3':
#         for gen in df['Model Type'].unique():
#             current_trace = go.Box(
#                             y=df.loc[df['Model Type']==gen, 'Total Time'],
#                             name=gen)
#             traces.append(current_trace)
#     else:
#         trace1 = go.Box(y=df['Total Time'],
#                  name=gen_num       
#                  )
#         traces.append(trace1)
#     layout1 = go.Layout(
#             title='{} Ticket Downtime Quartiles'.format(gen_num.title()),
#             height=1000,
#             xaxis={'title': 'Model'},
#             yaxis={'range': [0, outlier_val], 
#                    'title': site_tower_val.title() + ' ticket downtimes'}
#             )
#     g = dcc.Graph(
#             id='filtered_box',
#             figure=go.Figure(
#                     data=traces,
#                     layout=layout1)
#             )
#     return g, str(df.shape)
# 
# 
# if __name__ == '__main__':
#     app.run_server(debug=True, port=8000)
# 
# 
#==============================================================================
