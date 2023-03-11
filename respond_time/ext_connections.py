"""
External connections module
"""


from enum import Enum
import numpy as np
import pandas as pd
from clickhouse_driver import Client
import pyodbc
import requests
import xml.etree.ElementTree as ET
from io import StringIO
import json


class ConnectionType(Enum):
    """Possible connection types"""

    SQLServer = 1
    ClickHouseServer = 2
    API = 3


class ConnectionData:
    """Describes connection input data"""

    def __init__(self, connection_type=None, server_name='', port=0, user_name='', password=''):
        self.server_name = server_name
        self.port = port
        self.user = user_name
        self.password = password
        self.connection_type = connection_type

    def __str__(self):
        return 'Connection type: {}\nHost: {}\nPort: {}\nUser: {}\nPassword: {}'.format(
            self.connection_type,
            self.server_name,
            self.port,
            self.user,
            self.password)

    @property
    def connection_type(self):
        return self._connection_type

    @connection_type.setter
    def connection_type(self, new_type):
        self._connection_type = new_type

    @property
    def server_name(self):
        return self._server_name

    @server_name.setter
    def server_name(self, new_server):
        self._server_name = new_server

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, new_port):
        self._port = new_port

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, new_user):
        self._user = new_user

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, new_password):
        self._password = new_password


class Connection:
    """A class implements connections to servers"""

    def __init__(self, connection_type, server, port, user, password):
        connection_data = ConnectionData(
            connection_type=connection_type,
            server_name=server,
            port=port,
            user_name=user,
            password=password
        )

        self._connection_data = connection_data
        self._connection = None
        self.connect()

    def __str__(self):
        return str(self.connection_data)

    @property
    def connection_data(self):
        return self._connection_data

    @property
    def connection(self):
        return self._connection

    def connect(self):
        """Connects to the server"""

    def disconnect(self):
        """Disconnects from the server"""

    def execute(self, query, *pars):
        """
        Executes query with parameters if existed

        returns DataFrame
        """


class SQL(Connection):
    """A class implements a connection to SQL Server"""

    def __init__(self, server, port=, user=, password):
        super().__init__(
            connection_type=ConnectionType.SQLServer,
            server=server,
            port=port,
            user=user,
            password=password
        )
        self._cursor = None

    def connect(self):
        """See base class"""

        connection_data = self.connection_data

        server_name = connection_data.server_name
        port = connection_data.port
        user = connection_data.user
        password = connection_data.password

        full_server_name = [server_name, str(port)]
        server = ','.join(full_server_name)

        p_str = 'driver={}; server={}; uid={}; pwd={}'.format(
            '{SQL Server}',
            server,
            user,
            password
        )
        new_connection = pyodbc.connect(p_str)
        self._connection = new_connection

    def disconnect(self):
        """See base class"""

        if self._cursor is pyodbc.Cursor:
            self._cursor.close()

    def execute(self, query, *pars):
        """
        query - string
        *pars - tuple
        """

        self._run(query, *pars)
        table = self._to_df()
        return table

    def _run(self, query, *pars):
        """
        Runs a query with parameters if existed

        Attributes:
            -query - string, a query to execute
            -*pars - tuple, sequence of parameters
        """

        if self._cursor is pyodbc.Cursor:
            self._cursor.close()
        self._cursor = self.connection.cursor()
        self._cursor.execute(query, *pars)

    def _to_df(self):
        """Converts a Cursor data into a DataFrame"""

        cursor_data = self._cursor.fetchall()
        desc = self._cursor.description
        columns = [col[0] for col in desc if col]
        table = pd.DataFrame(np.array(cursor_data), columns=columns)
        return table


class ClickHouse(Connection):
    """A class implements a connection to ClickHouse Server"""

    def __init__(self, server, port, user, password):
        super().__init__(
            connection_type=ConnectionType.ClickHouseServer,
            server=server,
            port=port,
            user=user,
            password=password
        )
        self._query_result = ''

    @property
    def query_result(self):
        return self._query_result

    def connect(self):
        """See base class"""

        con_data = self.connection_data

        server = con_data.server_name
        port = con_data.port
        user = con_data.user
        password = con_data.password

        new_connection = Client(host=server, user=user, password=password)
        self._connection = new_connection

    def disconnect(self):
        """See base class"""

        if self.connection is Client:
            self.connection.disconnect()

    def execute(self, query, *pars):
        """
        query - string
        *pars - dict
        """

        self._run(query, *pars)
        table = self._to_df()
        return table

    def _run(self, query, *pars):
        """
        Runs a query with parameters if existed

        Attributes:
            -query - string, a query to execute
            -*pars - dict, sequence of parameters
        """

        self._query_result = self.connection.execute(query, *pars, with_column_types=True)

    def _to_df(self):
        """Converts a Client data into a DataFrame"""

        strings = []
        columns = []
        if self._query_result:
            if len(self._query_result) > 1:
                columns = [col[0] for col in self._query_result[1] if col]

            result = self._query_result[0]
            strings = np.array(result).reshape(len(result), len(columns))

        table = pd.DataFrame(strings, columns=columns)
        return table


class API(Connection):
    def __init__(self, server):
        super().__init__(
            connection_type=ConnectionType.API,
            server=server,
            port=0,
            user='',
            password=''
        )
        self._query_result = ''

    @property
    def query_result(self):
        return self._query_result

    def connect(self):
        """See base class"""

        new_connection = requests.Session()
        self._connection = new_connection

    def disconnect(self):
        """See base class"""

        self.connection.close()

    def execute(self, query, *pars):
        """
        query - string
        *pars - dict
        """

        self._run(query, *pars)

        file_type = 'csv'
        if len(pars) > 2:
            file_type = pars[2]
        table = self._to_df(file_type)

        return table

    def _run(self, query, *pars):
        """
        Runs a query with parameters if existed

        Attributes:
            -query - string, a query to execute
            -*pars - dict, sequence of parameters
        """

        with self.connection as session:
            url = query
            method = 'GET'
            if pars:
                method = pars[0]
            headers = {}
            if len(pars) > 1:
                headers = pars[1]
            respond = session.request(method=method, url=url, headers=headers, timeout=600)
            if respond.ok:
                self._query_result = respond.text
            else:
                self._query_result = ''

    def _to_df(self, data_type='csv'):
        """Converts a Client data into a DataFrame"""

        if not self.query_result:
            return pd.DataFrame()

        df = None

        if data_type == 'csv':
            data = StringIO(self.query_result)
            df = pd.read_csv(data, sep=',')
        elif data_type == 'tsv':
            data = StringIO(self.query_result)
            df = pd.read_csv(data, sep='\t')
        elif data_type == 'json':
            res = self.query_result
            res = str.replace(res, '{"Items":', "")
            res = res[:-1]
            res_dict = json.loads(res)
            df = pd.DataFrame.from_dict(res_dict)

        return df


class Parser:
    @staticmethod
    def parse_xml(xml_file):
        """
        Parse the input XML file and store the result in a pandas DataFrame.
        """

        tree = ET.parse(xml_file)
        root = tree.getroot()

        temp_dict = {}
        for node in root:
            for key, value in node.attrib.items():
                temp_dict[key] = 1

        cols = [key for key, value in temp_dict.items()]

        rows = []
        for node in root:
            if node is None:
                continue

            res = [node.attrib.get(col) for col in cols]
            rows.append({cols[i]: res[i] for i, _ in enumerate(cols)})

        out_df = pd.DataFrame(rows, columns=cols)
        return out_df

    @staticmethod
    def df_to_xml(df, full_path, item_name='item'):
        items = ET.Element(item_name + 's')
        for ind in df.index:
            item = ET.SubElement(items, item_name)
            for col in df.columns:
                value = df.iloc[ind][col]
                if isinstance(value, np.float64):
                    value = "{:.2f}".format(value)
                else:
                    value = str(value)
                item.set(col, value)

        xml_str = ET.tostring(items, encoding='utf-8')
        with open(full_path, 'wb') as new_file:
            new_file.write(xml_str)

    @staticmethod
    def df_to_csv(df, full_path):
        df.to_csv(full_path, index=False)
