import sqlite3
from sqlite3 import Error
import json
from .query_wiki import query_wiki
import re
from . import database_ops as dbo

def table_col_info(cursor, tableName, printOut=False):
    """
    Returns a list of tuples with column informations:
    (id, name, type, notnull, default_value, primary_key)
    """
    cursor.execute('PRAGMA TABLE_INFO({})'.format(tableName))
    info = cursor.fetchall()

    if printOut:
        print("Column Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey")
        for col in info:
            print(col)
    return info

def create_tables(cursor, tableNames, columnInfo, clobber = False):
    """
    create_tables attempts to create a table for each table in the list tableNames with
    columns as defined by columnInfo. For each if table = tableNames[k] then the columns for
    table are defined by columns = columnInfo[k]. Note that each element in columnInfo must
    be a list of strings of the form column[j] = "jth_column_name jth_column_data_type"

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        tableNames (list(string)): string labels for tableNames
        columnInfo (list(list(string))): list of string labels for each column of each table
        clobber (bool): flag to determine if old tables should be overwritten

    Returns:
        status (int): 0 if table creation failed, 1 if table creation was successful
    """

    for (table, colInfo) in zip(tableNames, columnInfo):
        columnInfoString = ", ".join(colInfo)
        try:
            if clobber:
                cursor.execute("DROP TABLE IF EXISTS {tableName}".format(tableName=table))
            cursor.execute("CREATE TABLE {tableName} ({columnInfo})".format(tableName=table,columnInfo=columnInfoString))
        except Error as e:
            print(e)
            print("Table {} already exists! Here's it's column info:".format(table))
            table_col_info(cursor, table, True)
            print("***")
            return 0

    return 1
