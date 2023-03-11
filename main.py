#!/usr/bin/env python

import respond_time.respond_time as respond_time


def start():
    pharmacy_calc = respond_time.RespondTime()
    pharmacy_calc.get_respond_time_table()
    pharmacy_calc.finish()


def main():
    start()


if __name__ == '__main__':
    main()
