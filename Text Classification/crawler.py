from requests_html import HTMLSession
import process_data
import urllib3
import requests
import socket
import lxml
import random

ECONOMY_LINKS = "Data/Economy/links.txt"
ENTERTAINMENT_LINKS = "Data/Entertainment/links.txt"
POLITICS_LINKS = "Data/Politics/links.txt"
SPORT_LINKS = "Data/Sports/links.txt"
TECH_LINKS = "Data/Technology/links.txt"

countT = 0

def get_paragraph(path , category):
    session = HTMLSession()
    LINKS = process_data.readData(path)
    LINKS = process_data.remove_N(LINKS)
    LINKS = LINKS[0 : 6000]
    count = countT
    while count < len(LINKS):
        try:
            r = session.get(LINKS[count])
            contents = r.html.find('div[class="content-list-component yr-content-list-text text"]')
            if contents:
                for content in contents:
                    texts = content.find("p")
                    if texts:
                        for txt in texts:
                            with open("Data_Raw" + "/" + category + "/" + str(count) + ".txt" , 'a' , encoding="utf8") as f:
                                f.write(txt.text + "\n")
            
        except urllib3.exceptions.NewConnectionError as e:
            print(e)
        except urllib3.exceptions.MaxRetryError as e:
            print(e)
        except requests.exceptions.ConnectionError as e:
            print(e)
        except socket.gaierror as e:
            print(e)
        except lxml.etree.ParserError as e:
            print(e)
        count += 3

get_paragraph(ECONOMY_LINKS , "Economy")
countT = 0
get_paragraph(ENTERTAINMENT_LINKS , "Entertainment")
countT = 0
get_paragraph(POLITICS_LINKS , "Politics")
countT = 0
get_paragraph(SPORT_LINKS , "Sports")
countT = 0
get_paragraph(TECH_LINKS , "Technology")

