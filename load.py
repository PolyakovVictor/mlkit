import requests
import os
import pathlib
import re

from helper import DEBUG
from typing import Optional


class Load():
    def __init__(self,) -> None:
        pass

    @staticmethod
    def fetch_by_url(url: str, filename: str = 'file', dir: Optional[str] = None) -> str | pathlib.Path:
        if pathlib.Path(filename).exists(): return pathlib.Path(filename)
        if DEBUG >= 1: print(f'Starting to download the file from URL: {url}')
        response = requests.get(url)
        assert response.status_code in {200, 206}
        cd = response.headers.get('Content-Disposition')
        if cd: filename = _m.group(1) if (_m:=re.search(r'filename="?([^"]+)"?', cd)) else filename
        with open(filename, 'wb') as f: f.write(response.content)
        print(f'File `{filename}` downloaded successfully.')
        return (os.getcwd() + '/' + filename)
