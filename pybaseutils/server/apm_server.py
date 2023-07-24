# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-07-18 09:26:39
    @Brief  : pip install elastic-apm
"""


def apm_service_flask(app, name, url, env):
    """
    Flask接入APM监控
    :param app:  Flask
    :param name: 服务名称
    :param url:  APM Server地址，域名+端口
    :param env:  服务部署的环境，可填写项有：dev/stage/partner/prd。
                 通过SCI部署的服务，可通过配置RUNTIME_ENV变量获取。,
                 env = os.getenv('RUNTIME_ENV')
    :return:
    """
    from elasticapm.contrib.flask import ElasticAPM
    # 配置apm的相关参数
    app.config['ELASTIC_APM'] = {
        'SERVICE_NAME': name,
        'SERVER_URL': url,
        'ENVIRONMENT': env
    }

    apm = ElasticAPM(app)
    print("APM Server config:{}".format(app.config))
    return apm


def apm_service_fastapi(app, name, url, env):
    """
    FastAPI接入APM监控
    :param app:  FastAPI
    :param name: 服务名称
    :param url:  APM Server地址，域名+端口
    :param env:  服务部署的环境，可填写项有：dev/stage/partner/prd。
                 通过SCI部署的服务，可通过配置RUNTIME_ENV变量获取。,
                 env = os.getenv('RUNTIME_ENV')
    :return:
    """
    from elasticapm.contrib.starlette import make_apm_client, ElasticAPM
    # 配置apm的相关参数
    config = {
        'SERVICE_NAME': name,
        'SERVER_URL': url,
        'ENVIRONMENT': env,
        # 'GLOBAL_LABELS': 'platform=Demo, application=demo_testing'
    }

    apm = make_apm_client(config)
    app.add_middleware(ElasticAPM, client=apm)
    print("APM Server config:{}".format(config))
    return apm


if __name__ == "__main__":
    from flask import Flask

    # 获取部署环境信息
    # apm_
    env = "dev"
    url = 'http://elk-apm.dm-ai.com:8200'
    app = Flask(__name__)
    apm_service_flask(app, name="demo-service", url=url, env=env)
    app.run(host='0.0.0.0', port=8080)
