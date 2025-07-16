import os

from signxai.methods.wrappers import calculate_relevancemap

from utils.data import load_and_preprocess_ecg
from utils.explainability import normalize_ecg_relevancemap
from utils.model import load_models_from_paths
from utils.viz import plot_ecg


def run(model_id, record_id, subsample_start=0, posthresh=0.2, cmap_adjust=0.3, fltr=True, last_conv_layer_name='activation_4', model_dir='../data/models/', plot_dir='../plots/ecgs/', filetype='png'):
    # Generate plot dir
    os.makedirs('{}{}'.format(plot_dir, model_id) , exist_ok=True)

    # Methods to use
    methods = ['grad_cam_timeseries', 'gradient', 'gradient_x_input', 'gradient_x_sign', 'lrp_alpha_1_beta_0', 'lrp_epsilon_0_5_std_x', 'lrpsign_epsilon_0_5_std_x']

    # Load models
    model, model_wo_softmax = load_models_from_paths(
        modelpath='{}{}/model.json'.format(model_dir, model_id),
        weightspath='{}{}/weights.h5'.format(model_dir, model_id)
    )

    # Load ECG
    ecg = load_and_preprocess_ecg(record_id=record_id,
                                  ecg_filters=['BWR', 'BLA', 'AC50Hz', 'LP40Hz'],
                                  subsampling_window_size=2000,
                                  subsample_start=subsample_start)

    # Iterate over methods
    for method in methods:
        # Calculate relevancemap based on SIGN-XAI package
        R = calculate_relevancemap(method, ecg, model_wo_softmax, last_conv_layer_name=last_conv_layer_name)

        # Use only positives
        R[R < 0] = 0

        # Normalize relevance map
        Rn = normalize_ecg_relevancemap(R)

        if fltr:
            # Discard values <= posthresh
            Rn[Rn <= posthresh] = 0

            # Amplify positives for better visualisation
            Rn[Rn > posthresh] = Rn[Rn > posthresh] + cmap_adjust

        # Plot preparation
        os.makedirs(plot_dir, exist_ok=True)
        save_to = '{}{}/{}_{}.{}'.format(plot_dir, model_id, model_id, method, filetype)

        # Plot ECG with explanation
        plot_ecg(ecg=ecg,
                 explanation=Rn,
                 title=record_id,
                 save_to=save_to)



if __name__ == '__main__':
    run('AVB', '03509_hr')
    run('ISCH', '12131_hr')
    run('LBBB', '14493_hr')
    run('RBBB', '02906_hr')