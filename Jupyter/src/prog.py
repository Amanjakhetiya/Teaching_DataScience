#!/usr/bin/env python

import sys
import re

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isdoublequotestring(value):
    matchObj = re.match(r'\"(.+?)\"', value, re.M | re.I)
    if matchObj:
        return True
    else:
        return False

class JsonParser(object):
    def parse(self, input_str):
        tokens = [x for x in re.split(',| ',input_str) if x!='']
        for tok in tokens:
            if  isfloat(tok):
                yield float(tok)
            elif isdoublequotestring(tok):
                yield str(tok)



def main(argv):
    json_parser = JsonParser()

    print("First Step")
    for value in json_parser.parse(" [ 10, 20, 30.1 ] "):
        print(value)

    print("\nSecond Step")
    for value in json_parser.parse(" [ 10 , 20, \"hello\", 30.1 ] "):
        print(value)
    #
    # print "\nThird Step"
    # for key, value in json_parser.parse("""{
    #         "hello": "world",
    #         "key1": 20,
    #         "key2": 20.3,
    #         "foo": "bar" }""").items():
    #     print key, value
    #
    # print "\nFourth Step"
    # for key, value in json_parser.parse("""{
    #         "hello": "world",
    #         "key1": 20,
    #         "key2": 20.3,
    #         "foo": {
    #             "hello1": "world1",
    #             "key3": [200, 300]
    #         } }""").items():
    #     if isinstance(value, dict):
    #         for key2, value2 in value.items():
    #             print key2, value2
    #     else:
    #         print key, value

main("")

