# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-07-18 10:46:25
    @Brief  :
"""
# -*- coding: utf-8 -*-
import os
from elasticapm.contrib.flask import ElasticAPM
from flask import Flask

# 获取部署环境信息
apm_env = "dev"
app = Flask(__name__)
# 配置apm的相关参数
app.config['ELASTIC_APM'] = {
    'SERVICE_NAME': 'cv-service',
    'SERVER_URL': 'http://elk-apm.dm-ai.com:8200',
    'ENVIRONMENT': apm_env
}

@app.route('/metrics')
def metrics():
    return 'Hello world'

apm = ElasticAPM(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)