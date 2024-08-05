import re
import datetime
import holidays


def check_holiday(date_info):
    all_holidays = holidays.country_holidays('US', years=date_info.year)
    if date_info in all_holidays:
        return all_holidays[date_info]
    return ''


def process_date(date_str, year=datetime.date.today().year):
    if date_str.lower() in ['today', 'current', 'now']:
        date_info = datetime.date.today()
        holiday_str = check_holiday(date_info)
        date_info_str = str(date_info) + ', ' + str(date_info.strftime('%A'))
        if holiday_str != '':
            date_info_str += ', ' + holiday_str + ' of ' + str(date_info.year)
        return date_info, date_info_str
    elif re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date_str):  # 2024-6-25 format
        date_info = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        holiday_str = check_holiday(date_info)
        date_info_str = str(date_info.strftime('%A'))
        if holiday_str != '':
            date_info_str += ', ' + holiday_str + ' of ' + str(date_info.year)
        return date_info, date_info_str
    else:
        all_holiday_names = {name: date_info for date_info, name in holidays.country_holidays('US', years=year).items()}
        for name, date_info in all_holiday_names.items():
            if date_str.lower() in name.lower():
                date_info_str = str(date_info) + ', ' + str(date_info.strftime('%A'))
                return date_info, date_info_str
        return None, ''


def calendar(text):
    calendar_match = re.search(r'Calendar\((.*)\)', text)
    if calendar_match:
        dates = calendar_match.group(1).split(',')
        dates = [date.strip() for date in dates]
        if len(dates) == 1:
            date_info, date_info_str = process_date(dates[0])
            if date_info:
                return text[: text.find(']')] + '-> ' + dates[0] + ' is ' + date_info_str + ' ]'
        elif len(dates) == 2:
            date_info1, date_info_str1 = process_date(dates[0])
            date_info2, date_info_str2 = process_date(
                dates[1], year=date_info1.year if date_info1 else datetime.date.today().year)
            if date_info1 or date_info2:
                final = text[: text.find(']')] + '-> '
                if date_info1:
                    final += dates[0] + ' is ' + date_info_str1 + '. '
                if date_info2:
                    final += dates[1] + ' is ' + date_info_str2 + '. '
                date_delta = date_info2 - date_info1 if date_info1 and date_info2 else None
                if date_delta and date_info2 > date_info1:
                    final += ('The days from ' + dates[0] + ' until ' + dates[1] + ' is ' +
                              str(date_delta.days) + ' days. ')
                elif date_delta and date_info2 < date_info1:
                    final += ('The days from ' + dates[1] + ' until ' + dates[0] + ' is ' +
                              str(abs(date_delta.days)) + ' days. ')
                elif date_delta and date_info2 == date_info1:
                    final += dates[0] + ' is ' + dates[1] + '. '
                final += ' ]'
                return final
    return text
