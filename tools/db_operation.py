#!/usr/bin/env python
# -*- encoding: utf-8; py-indent-offset: 4 -*-

import pymysql
from tools.json_common import JsonCommon
from sshtunnel import SSHTunnelForwarder


class DbOperation:
    def __init__(self):
        self.connection = None
        self.json_util = JsonCommon()

    def connect(self, host, port, user, passwd, database):
        self.connection = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=database, charset='utf8')

    def ssh_connect(self, ssh_host, ssh_port, ssh_user, ssh_pwd, db_host, db_port, db_user, db_pwd, database):
        with SSHTunnelForwarder(
                (ssh_host, ssh_port),  # B机器的配置
                ssh_password=ssh_pwd,
                ssh_username=ssh_user,
                remote_bind_address=(db_host, db_port)) as server:  # A机器的配置

            self.connection = pymysql.connect(host='127.0.0.1',  # 此处必须是是127.0.0.1
                                   port=server.local_bind_port,
                                   user=db_user,
                                   passwd=db_pwd,
                                   db=database)

    def excute(self, sql, debug=False):
        try:
            if debug :
                print('sql excute : %s' % sql)

            cursor = self.connection.cursor()

            effect_row = cursor.execute(sql)

            results = cursor.fetchall()

            # results = cursor.fetchone()

            if debug :
                print('sql result : %s' % self.json_util.to_json(results))

            return results
        except (BaseException) as e:
            print('sql excute error {}'.format(e))
            self.connection.rollback()


    def close_connect(self):
        self.connection.close()
