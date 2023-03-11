"""
A module for calculating pharmacies' current respond time.
"""

import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import ext_connections as ext_con
import requests
import os
import json


class RespondTime:
    """
    A class for calculating pharmacies' current respond time.

    The class is designed for realtime calculating and previous calculation
    of pharmacy's respond time. After input data is received
    it's possible to create calculation math. models.

    There are 3 math. models represented:
    1. Linear interpolation with a median smoothing (a large sample of data)
    2. Linear interpolation with a Savitzky-Golay smoothing (a medium sample of data)
    3. Canonical averaging (a small sample of data)
    """

    def __init__(self):
        self.settings = RespondSettings()
        self.connection = ext_con.API()

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, connection):
        self._connection = connection

    def get_pharmacy_data_table(self, sn):
        """Creates an input table from server"""

        query = self.settings.get_setting('reaction_api')
        if not query:
            return None

        query += '/?sn=' + str(sn)

        auth = self.settings.get_setting('auth')
        if not auth:
            return None

        method = 'GET'
        headers = {
            "Authorization": ' '.join(["Basic", auth]),
            "Content-type": "application/json",
            "Accept": "application/json"
        }
        file_type = 'json'
        table = self.connection.execute(query, method, headers, file_type)

        if table is None or table.empty:
            return None

        table.FixedDT = table.FixedDT.apply(pd.to_datetime)
        table.TotalTimeSec = table.TotalTimeSec.astype(int)
        table['Hour'] = table['FixedDT'].dt.hour
        table['Minute'] = table['FixedDT'].dt.minute
        table['Weekday'] = table['FixedDT'].dt.dayofweek + 1

        return table

    def get_pharmacies_schedules(self):
        query = self.settings.get_setting('schedule_api')
        if not query:
            return None

        auth = self.settings.get_setting('auth')
        if not auth:
            return None

        method = 'GET'
        headers = {
            "Authorization": ' '.join(["Basic", auth]),
            "Content-type": "application/json",
            "Accept": "application/json"
        }
        file_type = 'json'
        table = self.connection.execute(query, method, headers, file_type)

        return table

    def get_respond_time_table(self, return_full_table=False):
        """Returns a table of calculated respond time"""

        interval_mins = int(self.settings.get_setting('interval_minutes'))
        min_res = int(self.settings.get_setting('min_res'))
        max_res = int(self.settings.get_setting('max_res'))

        current_sn = 0
        sn_str = self.settings.get_setting('pharmacy_sn')
        if sn_str:
            current_sn = int(sn_str)

        columns = ['ID_Branch', 'Weekday', 'StartTime', 'EndTime', 'RespondTime']
        result_table = pd.DataFrame([], columns=columns)
        full_table = pd.DataFrame([], columns=columns)

        is_sent = False
        sent_count = 0
        schedule_table = self.get_pharmacies_schedules()
        count = len(schedule_table)
        print(count, 'pharmacies:')
        for index, row in schedule_table.iterrows():
            is_sent = False

            num = index + 1
            pharmacy_sn = int(row.SerialNumber)
            if current_sn and pharmacy_sn != current_sn:
                continue

            print(str(pharmacy_sn), '[', str(num), ']')

            pharmacy_table = self.get_pharmacy_data_table(pharmacy_sn)
            if pharmacy_table is None or pharmacy_table.empty:
                continue
            elif len(pharmacy_table) <= min_res:
                schedule = row.Schedule.split(';')
                respond_table = self._calculate_small_sample(pharmacy_table, interval_mins, schedule)
            elif len(pharmacy_table) <= max_res:
                respond_table = self._calculate_medium_sample(pharmacy_table, interval_mins)
            else:
                respond_table = self._calculate_large_sample(pharmacy_table, interval_mins)

            if respond_table is None:
                continue

            pharmacy_id = row.ID
            respond_table['ID_Branch'] = pharmacy_id

            result_table = result_table.append(respond_table, sort=False)
            if return_full_table:
                full_table = full_table.append(respond_table, sort=False)
            sent_count += 1

            if sent_count == 5 or (num == count and not result_table.empty):
                self.send_data(result_table)
                sent_count = 0
                result_table = pd.DataFrame([], columns=columns)
                is_sent = True

        if not is_sent and not result_table.empty:
            self.send_data(result_table)

        print('Calculated respond time and sent to the server... ' + datetime.datetime.now().strftime('%H:%M:%S'))

        return full_table

    def send_data(self, table: pd.DataFrame):
        """Sends data of calculated respond time to the SQL using API"""

        if table is None or table.empty:
            return False

        query = self.settings.get_setting('send_api')
        if not query:
            return False

        auth = self.settings.get_setting('auth')
        if not auth:
            return False

        table.Weekday = table.Weekday.astype(str)
        table.RespondTime = table.RespondTime.astype(str)
        table.StartTime.map(lambda x: x.strftime('%H:%M:%S'))
        table.EndTime.map(lambda x: x.strftime('%H:%M:%S'))

        items = table.to_json(orient='records')
        items_data = json.loads(items)
        json_data = {'Items': items_data}

        headers = {
            "Authorization": ' '.join(["Basic", auth]),
            "Content-type": "application/json",
            "Accept": "application/json"
        }

        try:
            respond = requests.post(url=query, headers=headers, json=json_data)
            if respond.status_code != 200:
                return False

            json_response = respond.json()
            if json_response['Status'] == 'Error':
                print('Error:', json_response['Description'])
                return False
        except ConnectionResetError as e:
            print('Error:', e)
            return False
        except requests.exceptions.ConnectionError as e:
            print('Error:', e)
            return False

        return True

    def finish(self):
        self.connection.disconnect()

    @staticmethod
    def save_table(table: pd.DataFrame, path):
        """Saves a .csv file of calculated respond time"""

        if not path:
            return ''

        if path[-1] != '\\':
            path += '\\'

        file_date = datetime.datetime.now().strftime('%Y%m%d')
        file_name = 'respond_time_' + file_date
        file_ext = '.csv'
        full_path = path + file_name + file_ext

        table.to_csv(full_path, header=True, index=False)

        return path

    def _calculate_large_sample(self, table: pd.DataFrame, interval_minutes: int):
        """1st math. model"""

        pharmacy_table = table.copy()
        pharmacy_table.sort_values(
            ['Weekday', 'Hour', 'Minute', 'FixedDT'],
            ascending=[True, True, True, True],
            inplace=True
        )

        intervals, responds = self._get_x_y(pharmacy_table)
        median_responds = Smoothing.median_smoothing(responds, 3)

        itp_f = self._get_itp_f(intervals, median_responds)

        new_intervals = np.linspace(intervals.min(), intervals.max(), 10000)
        new_responds = itp_f(new_intervals)
        itp_func = self._get_itp_f(new_intervals, new_responds)

        respond_table = self._get_output_table(itp_func, interval_minutes, by_weekday=True)

        return respond_table

    def _calculate_medium_sample(self, table: pd.DataFrame, interval_minutes: int):
        """2nd math. model"""

        pharmacy_table = table.copy()
        pharmacy_table.sort_values(
            ['Hour', 'Minute', 'FixedDT'],
            ascending=[True, True, True],
            inplace=True
        )

        intervals, responds = self._get_x_y(pharmacy_table, by_weekday=False)

        lin_intervals = np.linspace(intervals.min(), intervals.max(), 1000)
        sg_responds = Smoothing.savgol_smoothing(intervals, responds, lin_intervals)

        itp_func = self._get_itp_f(lin_intervals, sg_responds)

        new_intervals = np.linspace(lin_intervals.min(), lin_intervals.max(), 1000)
        new_responds = itp_func(new_intervals)
        new_itp_func = self._get_itp_f(new_intervals, new_responds)

        respond_table = self._get_output_table(new_itp_func, interval_minutes, by_weekday=False)

        return respond_table

    def _calculate_small_sample(self, table: pd.DataFrame, interval_minutes: int, schedule: list):
        """3rd math. model"""

        if not schedule or not schedule[0]:
            return None

        pharmacy_table = table.copy()
        pharmacy_table.sort_values(
            ['Hour', 'Minute', 'FixedDT'],
            ascending=[True, True, True],
            inplace=True
        )

        # Calculating average respond time
        intervals, responds = self._get_x_y(pharmacy_table, by_weekday=True)

        respond_sum = responds.sum()
        respond_count = len(responds)
        avg_respond = int(respond_sum // respond_count)

        # Constants
        days_per_week = 7
        hours_per_day = 24
        minutes_per_hour = 60
        minutes_per_day = hours_per_day * minutes_per_hour

        time_format = '%H:%M'

        day_end_time_str = str(23) + ':' + str(59)
        day_end_dt = datetime.datetime.strptime(day_end_time_str, time_format)

        day_start_time_str = str(0) + ':' + str(0)
        day_start_dt = datetime.datetime.strptime(day_start_time_str, time_format)

        # Filling in minutes and respond time due to schedule
        minutes_list = []
        respond_list = []
        for weekday in range(days_per_week):
            day_schedule = ''
            if weekday < len(schedule):
                day_schedule = schedule[weekday]

            start_dt, end_dt = self._start_end_time(day_schedule)

            # Find a count of next closed days
            next_start_dt_work = start_dt
            next_closed_days = 0
            for ind in range(days_per_week):
                next_day = weekday + ind + 1
                if next_day >= days_per_week:
                    next_day -= days_per_week

                next_day_schedule = ''
                if next_day < len(schedule):
                    next_day_schedule = schedule[next_day]

                next_start_dt, next_end_dt = self._start_end_time(next_day_schedule)
                if next_start_dt == next_end_dt:
                    next_closed_days += 1
                else:
                    # Nearest work day start in a schedule
                    next_start_dt_work = next_start_dt
                    break

            for hour in range(hours_per_day):
                for minute in np.arange(0, minutes_per_hour, 1):
                    # Current calculating datetime with average respond time included
                    cur_time_str = str(hour) + ':' + str(minute)
                    cur_dt = datetime.datetime.strptime(cur_time_str, time_format)
                    cur_dt_delta = cur_dt + datetime.timedelta(seconds=avg_respond)

                    respond_time = 0
                    if cur_dt_delta >= end_dt:
                        # After schedule day's end
                        diff_from_day_start = (next_start_dt_work - day_start_dt).seconds
                        diff_till_day_end = (day_end_dt - end_dt).seconds
                        diff_schedule = diff_till_day_end + diff_from_day_start
                        diff_cur_to_end = (cur_dt - end_dt).seconds
                        diff_schedule_wo_cur = diff_schedule - diff_cur_to_end
                        if cur_dt < end_dt:
                            diff_cur_to_end = (end_dt - cur_dt).seconds
                            diff_schedule_wo_cur = diff_schedule + diff_cur_to_end
                        diff_total = diff_schedule_wo_cur + avg_respond

                        respond_time = 24 * 60 * 60 * next_closed_days + diff_total
                    elif start_dt < cur_dt_delta < end_dt:
                        # In work day schedule
                        diff_after = (start_dt - cur_dt).seconds
                        if cur_dt > start_dt:
                            diff_after = 0
                        respond_time = diff_after + avg_respond
                    elif cur_dt_delta <= start_dt:
                        # Before schedule day's start
                        diff_before = (start_dt - cur_dt).seconds
                        respond_time = diff_before + avg_respond

                    respond_list.append(respond_time)

                    cur_minute = weekday * minutes_per_day + hour * minutes_per_hour + minute
                    minutes_list.append(cur_minute)

        intervals = np.array(minutes_list)
        responds = np.array(respond_list)

        itp_func = self._get_itp_f(intervals, responds)

        respond_table = self._get_output_table(itp_func, interval_minutes, by_weekday=False)

        return respond_table

    @staticmethod
    def _get_output_table(func: interp1d, interval_minutes: int, by_weekday: bool = True):
        """Gets an output table of calculated respond time"""

        columns = ['Weekday', 'StartTime', 'EndTime', 'RespondTime']
        strings = []

        days_per_week = 7
        hours_per_day = 24
        minutes_per_hour = 60
        minutes_per_day = hours_per_day * minutes_per_hour

        for weekday in range(days_per_week):
            for hour in range(hours_per_day):
                for minute in np.arange(0, minutes_per_hour, interval_minutes):
                    new_weekday = weekday
                    if not by_weekday:
                        new_weekday = 0

                    today_minutes = new_weekday * minutes_per_day + hour * minutes_per_hour + minute
                    start_time = datetime.time(hour, minute, 0)
                    if minute + interval_minutes == minutes_per_hour:
                        if hour + 1 == hours_per_day:
                            end_time = datetime.time(0, 0, 0)
                        else:
                            end_time = datetime.time(hour + 1, 0, 0)
                    else:
                        end_time = datetime.time(hour, minute + interval_minutes, 0)
                    respond_time = int(abs(func(today_minutes)))
                    strings.append([weekday + 1, start_time, end_time, respond_time])

        respond_table = pd.DataFrame(strings, columns=columns)

        return respond_table

    @staticmethod
    def _get_x_y(table: pd.DataFrame, by_weekday: bool = True):
        """Gets x as minute of datetime and y as respond time"""

        minutes_list = []
        respond_list = []

        for index, row in table.iterrows():
            total_minute = (row['Weekday'] - 1 if by_weekday else 0) * 24 * 60 + row['Hour'] * 60 + row['Minute']
            if total_minute in minutes_list:
                continue
            minutes_list.append(total_minute)
            respond_list.append(row['TotalTimeSec'])

        if minutes_list:
            min_minute = 0
            max_minute = (7 if by_weekday else 1) * 24 * 60 - 1

            if min_minute not in minutes_list:
                first_respond = respond_list[0]
                minutes_list.insert(0, min_minute)
                respond_list.insert(0, first_respond)

            if max_minute not in minutes_list:
                last_respond = respond_list[-1]
                minutes_list.append(max_minute)
                respond_list.append(last_respond)

        arr_x = np.array(minutes_list)
        arr_y = np.array(respond_list)

        return arr_x, arr_y

    @staticmethod
    def _get_itp_f(minutes: np.ndarray, respond_time: np.ndarray):
        """Gets an interpolating function respond_time(minutes)"""

        return interp1d(minutes, respond_time, kind='linear')

    @staticmethod
    def _start_end_time(schedule):
        time_format = '%H:%M'
        default_time = datetime.datetime.strptime('0:0', time_format)
        if not schedule:
            return default_time, default_time

        start_end_list = schedule.split('-')

        # Schedule start/end time of weekday
        if start_end_list:
            start_time_str = start_end_list[0].strip()
            try:
                start_dt = datetime.datetime.strptime(start_time_str, time_format)
            except ValueError as e:
                start_dt = default_time
        else:
            start_dt = default_time

        if len(start_end_list) > 1:
            end_time_str = start_end_list[1].strip()
            try:
                end_dt = datetime.datetime.strptime(end_time_str, time_format)
            except ValueError as e:
                end_dt = default_time
        else:
            end_dt = default_time

        return start_dt, end_dt


class RespondSettings:
    _settings = {}

    def __init__(self):
        curr_dir = os.getcwd()
        if curr_dir[-1] != '\\':
            curr_dir += '\\'

        file_name = curr_dir + 'settings.ini'
        with open(file_name, 'r') as file:
            for line in file.readlines():
                set_arr = line.split('=')
                key = set_arr[0].strip()
                value = line.replace(key + '=', '').strip()
                self._settings[key] = value

    def get_setting(self, name):
        return self._settings.get(name, '')


class Smoothing:
    """A class of smoothing methods"""

    @staticmethod
    def savgol_smoothing(
            x: np.ndarray,
            y: np.ndarray,
            new_x: np.ndarray,
            window: int = 101,
            order: int = 3):
        """Savitzky-Golay smoothing"""

        itp_func = interp1d(x, y, kind='linear')
        y_smooth = savgol_filter(itp_func(new_x), window, order)

        return y_smooth

    @staticmethod
    def mab_smoothing(y: np.ndarray, box_pts: int = 3):
        """Moving average boxes smoothing"""

        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')

        return y_smooth

    @staticmethod
    def triangle_smoothing(
            y: np.ndarray,
            degree: int = 3,
            drop_values: bool = False):
        """Triangle smoothing"""

        triangle = np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
        y_smooth = []

        for i in range(degree, len(y) - degree * 2):
            point = y[i:i + len(triangle)] * triangle
            y_smooth.append(sum(point) / sum(triangle))

        if drop_values:
            return y_smooth

        y_smooth = [y_smooth[0]] * int(degree + degree / 2) + y_smooth
        while len(y_smooth) < len(y):
            y_smooth.append(y_smooth[-1])

        return y_smooth

    @staticmethod
    def median_smoothing(y: np.ndarray, kernel_size: int = 3):
        """Median smoothing"""

        y_smooth = medfilt(y, kernel_size)

        return y_smooth
