import re
import requests


def search(query, limit=10):
    # TODO: Implement the search API
    search_match = re.search(r'WikiSearch\((.*)\)', query)
    if search_match:
        return query
    return query
