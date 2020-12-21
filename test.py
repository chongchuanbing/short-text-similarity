# -*- coding:utf-8 -*-

import unittest
import re

from tools.config_util import ConfigParser


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


class MyTest(unittest.TestCase):

    def tearDown(self):
        # 每个测试用例执行之后做操作
        print('111')

    def setUp(self):
        # 每个测试用例执行之前做操作
        print('22222')

    @classmethod
    def tearDownClass(self):
        # 必须使用 @ classmethod装饰器, 所有test运行完后运行一次
        print('4444444')

    @classmethod
    def setUpClass(self):
        # 必须使用@classmethod 装饰器,所有test运行前运行一次
        print('33333')


    def test_cleanhtml(self):
        str = '<p style="line-height: 11pt;"><font size="3"></font></p>'

        # str = '<?xml version="1.0"?><msg>	<img aeskey="8abcc594836468cdcdc4fc87338cbf98" encryver="0" cdnthumbaeskey="8abcc594836468cdcdc4fc87338cbf98" cdnthumburl="3053020100044c304a0201000204da55e2e602033d11fd0204ef3e5b6502045d1af3ee0425617570696d675f633933353731656337313730353736655f313536323034373436383731300204010418020201000400" cdnthumblength="5923" cdnthumbheight="217" cdnthumbwidth="104" cdnmidheight="0" cdnmidwidth="0" cdnhdheight="0" cdnhdwidth="0" cdnmidimgurl="3053020100044c304a0201000204da55e2e602033d11fd0204ef3e5b6502045d1af3ee0425617570696d675f633933353731656337313730353736655f313536323034373436383731300204010418020201000400" length="26511" md5="9b5208386dcd42f0b7cb47ee669942be" /></msg>'

        print(cleanhtml(str))

    def test_configparser(self):
        con = ConfigParser()
        res = con.get_config('model', 'base_path')
        print(res)

if __name__ == '__main__':

    unittest.main()
