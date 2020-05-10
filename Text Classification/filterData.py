import json

def readData(input):
    with open(input) as json_file:
        data = json_file.readlines()
        return data

# remove "\n"
def remove_N(contents):
    for line in contents:
        if line == "\n":
            contents.remove(line)
    comments = []
    for line in contents:
        if line.endswith('\n'):
            newline = line.replace('\n' , '')
            comments.append(newline)
    return comments

Data = readData("Data.json")

def getLinks(Data , category , pathDataLink):
    count = 0
    for data in Data:
        dataJson = json.loads(data)
        if dataJson["category"] == category:
            count += 1
            '''
            with open(pathDataLink , 'a' , encoding="utf8") as f:
                f.write(dataJson["link"] + "\n")
            with open(pathDataLink , 'a' , encoding="utf8") as f:
                if dataJson["headline"] == "":
                    dataJson["headline"] = category
                head = dataJson["headline"]
                head = head.replace('\n' , '')
                f.write(head + "\n")
            with open(pathDataLink , 'a' , encoding="utf8") as f:
                if dataJson["short_description"] == "":
                    head = dataJson["headline"]
                    head = head.replace('\n' , '')
                    f.write(head + "\n")
                else:
                    descipt = dataJson["short_description"]
                    descipt = descipt.replace('\n' , '')
                    f.write(descipt + "\n")
            '''
    print(count)

getLinks(Data , "" , "Data/Technology/links.txt")

