#!/usr/bin/env python
# -*- encoding: utf-8; py-indent-offset: 4 -*-

import json
import sys

class JsonCommon(object):
    def __init__(self):
        pass

    def to_json(self, object):
        return json.dumps(object, default=convert_to_builtin_type)


#转换函数
def convert_to_builtin_type(obj):
    # 把MyObj对象转换成dict类型的对象
    d = { '__class__':obj.__class__.__name__,
          '__module__':obj.__module__,
        }
    d.update(obj.__dict__)
    return d


if __name__ == '__main__':
    json_common = JsonCommon()
