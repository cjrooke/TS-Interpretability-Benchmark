import numpy as np


def getSaliencyTop1Accuracy(dataset_type, data_generation_types, models, saliency_methods, imp_timestep, imp_feature, saliency_dir):
    """For a given (feature, timestep), what % of the time is it recognized as the most important by a saliency method"""
    for generation_type in data_generation_types:
        for model in models:
            for saliency_method in saliency_methods:
                # TODO: Refac so this is less hardcoded
                filename = f"Simulated{dataset_type}_{generation_type}_{model}_{saliency_method}_rescaled.npy"
                saliency_data = np.load(saliency_dir + '/' + filename)
                num_samples, _, num_features = saliency_data.shape
                correct_count = 0
                for i in range(num_samples):
                    correct_count += np.argmax(saliency_data[i]) == imp_timestep * num_features + imp_feature
                accuracy = correct_count / num_samples
                print(f"{dataset_type} {generation_type} {model} {saliency_method} has Top1 accuracy {accuracy} ({correct_count} / {num_samples})")
