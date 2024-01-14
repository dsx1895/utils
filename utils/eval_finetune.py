import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import numpy as np
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset_return_noise import LibriMix_noise
from asteroid.data.librimix_dataset_return_noise_reverb import LibriMix_noise_reverb
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import e2e_noisy_sep_DPRNNTasNet,DPRNNTasNet,e2e_noisy_sep_signal_DPRNNTasNet,reverb_noisy_sep_signal_verse_DPRNNTasNet
from asteroid.models import reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_spp_sep
from asteroid.models import reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_spp_enhance
from asteroid.models import reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_dereverb
from asteroid.models import reverb_noisy_sep_signal_verse_DPRNNTasNet_subtask_all
from asteroid.models import save_publishable
from asteroid.models import reverb_noisy_sep_signal_verse_DPRNNTasNet_subtask_all
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from asteroid.metrics import WERTracker, MockWERTracker

# from .reverb_signal_verse_sep_dprnn_tasnet_sub_spp_sep import reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_spp_sep
# from .reverb_signal_verse_sep_dprnn_tasnet_sub_spp_enhance import reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_spp_enhance
# from .reverb_signal_verse_sep_dprnn_tasnet_sub_dereverb import reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_dereverb

from eval import eval_composite
parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, default="data/2spk_wav8k/min/test",  help="Test directory including the csv files"
)
#data/2spk_wav8k/min/test
#data/wav8k/min/test
parser.add_argument(
    "--task",
    type=str,
    default="sep_noisy",
    # required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="librimix_test_ll", 
    help="Directory in exp_dir where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)
# exp/subtask/spp_all_120
# 
parser.add_argument("--exp_dir", default="exp/subtask/spp_all_120_r1", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=1, help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--compute_wer", type=int, default=0, help="Compute WER using ESPNet's pretrained model"
)

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi","pesq"]
ASR_MODEL_PATH = (
    "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
)


def update_compute_metrics(compute_wer, metric_list):
    if not compute_wer:
        return metric_list
    try:
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader
    except ModuleNotFoundError:
        import warnings

        warnings.warn("Couldn't find espnet installation. Continuing without.")
        return metric_list
    return metric_list + ["wer"]


def main(conf):
    compute_metrics = update_compute_metrics(conf["compute_wer"], COMPUTE_METRICS)
    # anno_df = pd.read_csv(Path(conf["test_dir"]).parent.parent.parent / "test_annotations.csv")
    # wer_tracker = (
    #     MockWERTracker() if not conf["compute_wer"] else WERTracker(ASR_MODEL_PATH, anno_df)
    # )
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    # model_path = os.path.join(conf["exp_dir"], "checkpoints","epoch=96-step=337075.ckpt")
    # DPRNNTasNet,e2e_noisy_sep_DPRNNTasNet
    # e2e_noisy_sep_signal_DPRNNTasNet
    # DPRNNTasNet
    # reverb_noisy_sep_signal_verse_DPRNNTasNet
    sep_module = reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_spp_sep.from_pretrained('exp/subtask/sep_noisy_reverb/best_model.pth')
    enh_module = reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_spp_enhance.from_pretrained('exp/subtask/sep_enhance/best_model.pth')
    dereverb_module = reverb_noisy_sep_signal_verse_DPRNNTasNet_sub_dereverb.from_pretrained('exp/subtask/spp_sep_dereverb/best_model.pth')
    model = reverb_noisy_sep_signal_verse_DPRNNTasNet_subtask_all( 
        # **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"],
        n_src=2,
                sep_module=sep_module,enh_module=enh_module,dereverb_module=dereverb_module,
    )
    # # print(model_path)
    # model = model.from_pretrained(model_path)
    # for name in model.state_dict():
    #     # sep_module,  enh_module ,  dereverb_module
    #     if name == "sep_module.sep_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(model.state_dict()['sep_module.sep_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
    #     if name == "enh_module.enh_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(model.state_dict()['enh_module.enh_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
    #     if name == "dereverb_module.reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(model.state_dict()['dereverb_module.reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
    # for name in sep_module.state_dict():
    #     if name == "sep_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(sep_module.state_dict()['sep_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])

    # for name in enh_module.state_dict():
    #     if name == "enh_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(enh_module.state_dict()['enh_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])

    # for name in dereverb_module.state_dict():
    #     if name == "reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(dereverb_module.state_dict()['reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
    model = reverb_noisy_sep_signal_verse_DPRNNTasNet_subtask_all.from_pretrained(model_path)
    checkpoint = torch.load("exp/subtask/spp_all_120_r1/checkpoints/epoch=117-step=410050.ckpt")
    new_state_dict = model.state_dict()

    for key in new_state_dict.keys():
        if "model."+key in checkpoint['state_dict'].keys():
            # print(1)
            new_state_dict[key] = checkpoint['state_dict']["model."+key]
    model.load_state_dict(new_state_dict)
    # model = reverb_noisy_sep_signal_verse_DPRNNTasNet_subtask_all.from_pretrained(model_path)
    for name in model.state_dict():
        # sep_module,  enh_module ,  dereverb_module
        if name == "sep_module.sep_masker.net.0.intra_RNN.rnn.weight_ih_l0":
            print(model.state_dict()['sep_module.sep_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
        if name == "enh_module.enh_masker.net.0.intra_RNN.rnn.weight_ih_l0":
            print(model.state_dict()['enh_module.enh_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
        if name == "dereverb_module.reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0":
            print(model.state_dict()['dereverb_module.reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
    # for name in model.state_dict():
    #     # sep_module,  enh_module ,  dereverb_module
    #     if name == "sep_module.sep_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(model.state_dict()['sep_module.sep_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
    #     if name == "enh_module.enh_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(model.state_dict()['enh_module.enh_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])
    #     if name == "dereverb_module.reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0":
    #         print(model.state_dict()['dereverb_module.reverb_masker.net.0.intra_RNN.rnn.weight_ih_l0'][0][0])

    # Handle device placement
    # for name in model.state_dict():
    #     print(name)

    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = LibriMix_noise_reverb(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "test/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    csig, cbak, covl, count = 0, 0, 0, 0
    count = 0
    erle=0
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, reverb_source, noise,ids = test_set[idx]
        mix, sources, reverb_source = tensors_to_device([mix, sources,reverb_source], device=model_device)
        # noi_free_st_mask, est_sources = model(mix.unsqueeze(0))
        _,_,est_sources = model(mix.unsqueeze(0))
        # print()
        # print(est_sources.size())
        # print(sources[None].size())
        loss,_, reordered_sources,_ = loss_func(est_sources, sources[None], return_est=True)
        #loss,_, reordered_denoise_sources,_ = loss_func(denoise_source, reverb_source[None], return_est=True)
        # print(loss)
        mix_np = mix.cpu().data.numpy()
        reverb_source_np = reverb_source.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # reordered_denoise_sources_np = reordered_denoise_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        for i in np.arange(len(est_sources_np)):
            res = eval_composite(sources_np[i], est_sources_np[i])
            csig += res['csig']
            cbak += res['cbak']
            covl += res['covl']
            count += 1
            
            # normFactor = np.dot(est_sources_np[i], reverb_source_up[i])/np.dot(reverb_source_up[i], reverb_source_up[i])
            # reverb_source_up_temp = normFactor*reverb_source_up[i]
            # array_2d = array[:, np.newaxis]
            # print(sources_np[i][None,:])
            # print(reverb_source_np[i][None,:].shape)
            # normal_vector = np.cross(sources_np[i],reverb_source_np[i])
            # projection_matrix = np.outer(normal_vector, normal_vector) / np.dot(normal_vector, normal_vector)
            # projection = np.dot(projection_matrix, est_sources_np[i])


            # normFactor = np.dot(est_sources_np[i], reordered_denoise_sources_np[i])/np.dot(reordered_denoise_sources_np[i], reordered_denoise_sources_np[i])
            # reverb_source_up_temp = normFactor*reordered_denoise_sources_np[i]

            # signal_energy = np.mean(est_sources_np[i] ** 2)
            # noise_energy = np.mean(reverb_source_up_temp ** 2)
            # erle += 10 * np.log10(signal_energy / (noise_energy + 0.00000001)+ 0.00000001)

            # predict_near_end_wav = np.mean(reverb_source_up_temp**2)
            # mic_near = np.mean(reverb_source_np[i]**2)
            # erle  += 10 * np.log10(mic_near/predict_near_end_wav)
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.mixture_path
        est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
        # utt_metrics.update(
        #     **wer_tracker(
        #         mix=mix_np,
        #         clean=sources_np,
        #         estimate=est_sources_np_normalized,
        #         wav_id=ids,
        #         sample_rate=conf["sample_rate"],
        #     )
        # )
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np_normalized):
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)
    print(f'CSIG: {csig/count}, CBAK: {cbak/count}, COVL: {covl/count}')
    # print(f'erle: {erle/count}')
    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)
    # if conf["compute_wer"]:
    #     print("\nWER report")
    #     wer_card = wer_tracker.final_report_as_markdown()
    #     print(wer_card)
    #     # Save the report
    #     with open(os.path.join(eval_save_dir, "final_wer.md"), "w") as f:
    #         f.write(wer_card)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    publishable = save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict,
        metrics=final_results,
        train_conf=train_conf,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
