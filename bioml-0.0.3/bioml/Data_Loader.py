"""@package docstring
This module contains functions to load data from database in standard "app" structure

L_*** functions are Low level functions that work with pandas dataframe containing only the data and time as index
H_*** functions are High level functions that create pandas dataframe with standard "app" structure (with columns 'subject_id' and 'condition') and will call L_*** functions
HH_*** functions are even Higher level functions that accept a list of pandas dataframe with standard "app" structure and will call H_*** functions
"""

import pandas as pd
import bioml.DB_connection as DBC

def H_extract_data(engine, recs, channels, references, exp_info):
    """
    Extract selected data from DB

    :param engine: sqlalchemy engine Object, connection to DB created in DB_connection module
    :param recs: List, recordings to retrieve from database 
    :param channels: List, channels selected
    :param references: List, references electrodes if any
    :param exp_info: DataFrame containing information on Experiment
    :return: List of dataframes containing data for each stream in standard "app" structure and sampling frequencies extracted from exp_indo 
    """
    #Sampling frequencies are direcly written in exp_info
    FSs = exp_info[exp_info['is_target'] == False]['sampling_frequencies'].values

    #One table per stream
    Tables = [[] for _ in range(len(FSs))]
    
    #Loop on non target streams
    for i, Table_name in enumerate(exp_info[exp_info['is_target'] == False]['stream_names'].values):
        #Load selected channels            
        columns = 'index'
        #Channels + references but we removed the duplicates if any
        all_needed_columns = channels[i] + list(set(references[i]) - set(channels[i]))
        
        #columns of data should be in the format TableName_channelN
        for column in all_needed_columns:
            if column.startswith(Table_name):
                columns += ',"' + column + '"'

        columns += ', subject_id, condition'
        if columns != 'index, subject_id, condition': #If not empty
            #Each recording will be treated separately
            tables_recs = []
            for rec in recs:
                subj, cdt, nb = rec.split('_')
                cdt_nb = cdt + '_' + nb
                tables_recs.append(L_extract_data(engine, Table_name, columns, subj, cdt_nb))

            Tables[i] = pd.concat(tables_recs, axis = 0)
    return Tables, FSs.astype('float')

def H_load_labels(engine, cur, recs):
    """
    Extract selected labels from DB

    :param engine: sqlalchemy engine Object, connection to DB created in DB_connection module
    :param cur: psycopg2 cursor object created in DB_connection module
    :param recs: List, recordings to retrieve from database 
    :return: dataframe containing labels in standard "app" structure and table name in database
    """
    # Retrieve labels and interpolate values between labels and each window, cut X to
    #have windows only when there is prediction to do.
    label_table_name = DBC.get_label_table(cur)

    Labels_i = []
    for rec in recs:
        subj, cdt, nb = rec.split('_')
        cdt_nb = cdt + '_' + nb
        
        #Retrieve labels from Database
        Labels_i.append(L_extract_data(engine, label_table_name, '*', subj, cdt_nb))


    Labels = pd.concat(Labels_i, axis = 0)

    #Drop duplicates
    Labels =Labels[~Labels.index.duplicated(keep='first')]

    return Labels, label_table_name


def L_extract_data(engine, Table_name, columns, subj = None, cdt_nb = None, rec = None):
    """
    low level function to retrieve data from database

    :param engine: sqlalchemy engine Object, connection to DB created in DB_connection module
    :param Table_name: name of table in database to get data from
    :param columns: columns to retrieve from table with table name=Table_name 
    :param subj: value of subject in column 'subject_id' to retrieve
    :param cdt_nb: value of cdt_nb in column 'cdt_nb' to retrieve
    :param rec: value of rec in column 'rec' to retrieve (deprecated)
    :return: dataframe containing labels in standard "app" structure and table name in database
    """
    #Read Database
    if subj == None and cdt_nb == None and rec != None:
        warnings.warn('Using Recordings is deprecated please use subject and cdt_nb')
        data = pd.read_sql_query("SELECT {columns} FROM \"{table}\"  where \"Recording\" = '{rec}' order by index".format(columns = columns,
                                                                                                                        table = Table_name.lower(), 
                                                                                                                        rec = rec), engine)
    elif subj != None and cdt_nb != None and rec == None:
        data = pd.read_sql_query("SELECT {columns} FROM \"{table}\"  where (subject_id = '{subject}' and condition = '{cdt}') order by index".format(columns = columns,
                                                                                                                                                        table = Table_name.lower(), 
                                                                                                                                                        subject = subj, 
                                                                                                                                                        cdt = cdt_nb), engine)      
    elif subj == None and cdt_nb == None and rec == None:
        data = pd.read_sql_query("SELECT {columns} FROM \"{table}\" order by index".format(columns = columns,
                                                                                       table = Table_name.lower()), engine)  
    else:
        raise ValueError('Undefined situation')      

    data = data.set_index('index')
    
    #Drop duplicate indices
    data = data[~data.index.duplicated(keep='first')]

    return data