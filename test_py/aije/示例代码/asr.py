# -*- coding: utf-8 -*-
import os
import json
import shutil
from modelscope.pipelines import pipeline
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import Tasks
from funasr.datasets.ms_dataset import MsDataset
from funasr.utils.compute_wer import compute_wer


def modelscope_finetune(params):
    if not os.path.exists(params["model_dir"]):
        os.makedirs(params["model_dir"], exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = MsDataset.load(params["dataset_path"])
    kwargs = dict(
        model=params["modelscope_model_name"],
        data_dir=ds_dict,
        dataset_type=params["dataset_type"],
        work_dir=params["model_dir"],
        batch_bins=params["batch_bins"],
        max_epoch=params["max_epoch"],
        lr=params["lr"])
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()
    pretrained_model_path = os.path.join(os.environ["HOME"], ".cache/modelscope/hub",
                                         params["modelscope_model_name"])
    required_files = ["am.mvn", "decoding.yaml", "configuration.json"]
    for file_name in required_files:
        shutil.copy(os.path.join(pretrained_model_path, file_name),
                    os.path.join(params["model_dir"], file_name))


def modelscope_infer(params):
    # prepare for decoding
    with open(os.path.join(params["model_dir"], "configuration.json")) as f:
        config_dict = json.load(f)
        config_dict["model"]["am_model_name"] = params["decoding_model_name"]
    with open(os.path.join(params["model_dir"], "configuration.json"), "w") as f:
        json.dump(config_dict, f, indent=4, separators=(',', ': '))
    decoding_path = os.path.join(params["model_dir"], "decode_results")
    if os.path.exists(decoding_path):
        shutil.rmtree(decoding_path)
    os.mkdir(decoding_path)

    # decoding
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=params["model_dir"],
        output_dir=decoding_path,
        batch_size=64
    )
    audio_in = os.path.join(params["test_data_dir"], "wav.scp")
    inference_pipeline(audio_in=audio_in)

    # computer CER if GT text is set
    text_in = os.path.join(params["test_data_dir"], "text")
    if os.path.exists(text_in):
        text_proc_file = os.path.join(decoding_path, "1best_recog/token")
        compute_wer(text_in, text_proc_file, os.path.join(decoding_path, "text.cer"))
        os.system("tail -n 3 {}".format(os.path.join(decoding_path, "text.cer")))


if __name__ == '__main__':
    finetune_params = {}
    finetune_params["modelscope_model_name"] = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    finetune_params["dataset_path"] = "./example_data/"
    finetune_params["model_dir"] = "./checkpoint"
    finetune_params["dataset_type"] = "small"
    finetune_params["batch_bins"] = 2000
    finetune_params["max_epoch"] = 20
    finetune_params["lr"] = 0.00005

    modelscope_finetune(finetune_params)

    infer_params = {}
    infer_params["model_dir"] = "./checkpoint"
    infer_params["decoding_model_name"] = "20epoch.pb"
    infer_params["test_data_dir"] = "./example_data/test/"
    modelscope_infer(infer_params)
