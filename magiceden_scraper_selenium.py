import time

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pathlib, pickle
from collections import defaultdict

if __name__ == '__main__':
    url = 'https://magiceden.io/item-details/'
    base_dir = pathlib.Path.cwd()
    data_dir = pathlib.Path.cwd() / 'data'

    with open(str(data_dir / 'sample_data.pickle'), 'rb') as file:
        cm_data = pickle.load(file)

    degods_hashlist = cm_data['sample_cmid_to_hashlist']['degods']
    all_data = {}
    timeout = 40
    errors = []
    for count, sample in enumerate(degods_hashlist):
        try:
            if count % 10 == 0:
                print(f'Moving on to {count}')
            driver = webdriver.Chrome(executable_path='./chromedriver')
            cur_url = url + sample
            driver.implicitly_wait(10)
            driver.get(cur_url)
            driver.implicitly_wait(10)
            element_present = EC.presence_of_element_located((By.CLASS_NAME, 'accordion-button'))
            WebDriverWait(driver, timeout).until(element_present)
            button = driver.find_element_by_class_name('accordion-button')
            button.click()
            driver.implicitly_wait(3)
            hashmap = {}
            next_button = None
            page = 0
            row_num = 0
            while next_button or page == 0:
                soup_level = BeautifulSoup(driver.page_source, 'lxml')
                subclass = soup_level.find('div', {'class': 'accordion accordion-flush open', 'id': 'accordion-6'})
                table = subclass.find('table', {'role': 'table', 'class': 'me-table pinky'})
                tbody = table.find('tbody', {'role': 'rowgroup'})
                rows = tbody.find_all('tr', {'role': 'row'})
                for row in rows:
                    data = []
                    cells = row.find_all('td', {'role' : 'cell'})
                    for index, cell in enumerate(cells):
                        if index == 0 or index == 2:
                            point = cell.find('a')
                            text = point.attrs['href']
                            data.append(text)
                        elif index == 1:
                            point = cell.find('a')
                            text = point.text
                            data.append(text)
                        elif 2 < index <= 3:
                            point = cell.find('span')
                            text = point.text
                            data.append(text)
                        elif 3 < index <= 5:
                            text = cell.text
                            data.append(text)
                        elif 5 < index <= 7:
                            point = cell.find('span', {'title' : "cursor-pointer d-inline-flex align-items-center"})
                            if point is not None:
                                text = point.get_attribute('title')
                            else:
                                text = ''
                            data.append(text)
                    hashmap[row_num] = data
                    row_num += 1
                page += 1
                try:
                    next_button = driver.find_element_by_xpath('//button[text()=">"]')
                except:
                    next_button = None
                if next_button and next_button.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView();", next_button)
                    driver.execute_script("arguments[0].click();", next_button)
                else:
                    break
                driver.implicitly_wait(3)
            print(f'Length of hashmap is {len(hashmap)} for {sample}')
            all_data[sample] = hashmap
            driver.quit()
        except Exception as e:
            print(f'Ran into error for {sample}')
            errors.append(sample)
