import requests
import base64
import json
import os
import time
import uuid
# import threadpool
import numpy as np
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

dmid = '3be9b92408ac4d3ba954c8edd2d8d649'  # 联系DM技术支持获得

def tts_synthesis_request(text,audio_path=None,sample_rate=24000,format='wav',speech_rate=1.0,volume=60):
    # url = "http://localhost:9900/tts/synthesis"
    # url = "http://tts-stream.dev.dm-ai.com/tts/synthesis"
    # url = "http://tts-stream.stage.dm-ai.com/tts/synthesis"
    url = "http://tts-stream.dm-ai.com/tts/synthesis"
    task_id = uuid.uuid4().hex
    data = {
                'text': text,
                'voice_type' : "sasa",
                "emotion":"neutral",
                'sample_rate': sample_rate,
                'speech_rate': speech_rate,
                'volume': volume,
                'format': "wav",
                'use_cache': False,  
            }
    headers = {'Content-type': 'application/json', 'dmid':dmid,'task_id':task_id}
    req_dict = {"header":headers,"data":data}
    print(req_dict)

    r = requests.post(url, data=json.dumps(data), headers=headers)
    print(r.status_code)
    resp_body = json.loads(r.content)
    # if r.headers["status"] != '200000000':
    #     return r.headers
    d = {}
    d['headers']  = r.headers
    d['data'] = {}
    if audio_path is not None:
        audio = resp_body["audiodata"]
        audio = base64.b64decode(audio)
        with open(audio_path,"wb") as f:
            f.write(audio)
    d['data']['wordstamp'] = resp_body['wordstamp']
    d['data']['phonestamp'] = resp_body['phonestamp']
    d['data']['timecost'] = resp_body['timecost']
    return d




def askOne(url,text):
    start_time = time.time()
    resp = tts_synthesis_request(url,text,None)
    end_time = time.time()
    timecost = end_time-start_time
    return timecost


if __name__ == '__main__':
    
    # text = '看到您来自*{usr.userCity}，颐和园：山水相依，构筑精巧，希望有机会可以到*{usr.userCity}逛一逛，也预祝您此行玩得愉快，玩得尽兴！'
    # text = "为您查询到上海市2023-02-11的天气小雨，参考气温11℃，最高温度11℃，最低温度8℃，旁边屏幕为您展示了穿衣及出行建议哦"
    text = "聚焦中小学课后服务素质教育场景，融合精品书法教学资源，基于高精度自研AI算法，打造硬笔书法教学评测一体化课堂方案，以学生为中心，构建 “教学练测评辅” 的全流程教学闭环。"
    start_time = time.time()
    resp = tts_synthesis_request(text,'test.wav')
    end_time = time.time()
    timecost = end_time-start_time
    print(resp)
    print(timecost)

    # timecosts = []
    # # 多线程调用
    # N = 1000
    # thread_num = 8
    # error_num = 0
    # with ThreadPoolExecutor(max_workers=thread_num) as executor:
    #     future_to_ask = {executor.submit(askOne, url, text): _ for _ in range(N*thread_num)}
    #     for future in as_completed(future_to_ask):
    #         ask_param = future_to_ask[future]
    #         try:
    #             timecost = future.result()
    #         except Exception as exc:
    #             del future_to_ask[future]
    #             del future  #future._result = None
    #             print('%r generated an exception: %s' % (ask_param, exc))
    #             error_num += 1
    #         else:
    #             del future_to_ask[future]
    #             del future  #future._result = None
    #             print(timecost)
    #             timecosts.append(timecost)


    # timecosts = np.array(timecosts)
    # timecosts = np.sort(timecosts)
    # print(timecosts)
    # print("mean timecost:{:.3f} max:{:.3f} min:{:.3f} 50%:{:.3f} 90%:{:.3f}".format(
    #         np.mean(timecosts),timecosts.max(),timecosts.min(),timecosts[int(timecosts.shape[0]*0.5)],timecosts[int(timecosts.shape[0]*0.9)]))
    # print("error_num:{}".format(error_num))
    
