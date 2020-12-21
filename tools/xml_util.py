# -*- coding:utf-8 -*-

import xml.sax
import xml.sax.handler
import pprint


class XMLHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.buffer = ""
        self.mapping = {}

    def startElement(self, name, attributes):
        self.buffer = ""

    def characters(self, data):
        self.buffer += data

    def endElement(self, name):
        self.mapping[name] = self.buffer

    def getDict(self):
        return self.mapping


data = '<?xml version="1.0" encoding="UTF-8"?><note><to>World</to><from>Linvo</from><heading>Hi</heading><body>Hello World!</body></note>'
data = '<?xml version="1.0"?><msg>	<img aeskey="8abcc594836468cdcdc4fc87338cbf98" encryver="0" cdnthumbaeskey="8abcc594836468cdcdc4fc87338cbf98" cdnthumburl="3053020100044c304a0201000204da55e2e602033d11fd0204ef3e5b6502045d1af3ee0425617570696d675f633933353731656337313730353736655f313536323034373436383731300204010418020201000400" cdnthumblength="5923" cdnthumbheight="217" cdnthumbwidth="104" cdnmidheight="0" cdnmidwidth="0" cdnhdheight="0" cdnhdwidth="0" cdnmidimgurl="3053020100044c304a0201000204da55e2e602033d11fd0204ef3e5b6502045d1af3ee0425617570696d675f633933353731656337313730353736655f313536323034373436383731300204010418020201000400" length="26511" md5="9b5208386dcd42f0b7cb47ee669942be" /></msg>'

xh = XMLHandler()
xml.sax.parseString(data.encode(), xh)
ret = xh.getDict()

pprint.pprint(ret)