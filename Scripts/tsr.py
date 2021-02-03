import pathlib

import numpy as np
from sklearn import preprocessing

from .Helper import givenAttGetRescaledSaliency
from .Plotting.plot import plotExampleBox


def getTwoStepRescaling(Grad, input, TestingLabel, hasBaseline=None, hasFeatureMask=None,
                        hasSliding_window_shapes=None, return_time_ft_contributions=False, ft_dim_last=True):
    batch_size, sequence_length, input_size = input.shape if ft_dim_last else (input.shape[0], input.shape[2], input.shape[1])
    assignment = input[0, 0, 0]
    timeGrad = np.zeros((batch_size, sequence_length))
    inputGrad = np.zeros((batch_size, input_size))
    newGrad = np.zeros(input.shape)
    if hasBaseline is None:
        ActualGrad = Grad.attribute(input, target=TestingLabel).data.cpu().numpy()
    else:
        if (hasFeatureMask != None):
            ActualGrad = Grad.attribute(input, baselines=hasBaseline, target=TestingLabel,
                                        feature_mask=hasFeatureMask).data.cpu().numpy()
        elif (hasSliding_window_shapes != None):
            ActualGrad = Grad.attribute(input, sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline,
                                        target=TestingLabel).data.cpu().numpy()
        else:
            ActualGrad = Grad.attribute(input, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()

    timeGrad[:] = np.mean(np.absolute(ActualGrad), axis=2 if ft_dim_last else 1)

    timeContribution = preprocessing.minmax_scale(timeGrad, axis=1)
    # meanTime = np.quantile(timeContribution, .55)

    time_contributions = np.zeros((batch_size, sequence_length, input_size))
    time_contributions[:, :] = timeContribution[:, :, None]
    feature_contributions = np.zeros((batch_size, sequence_length, input_size))

    for t in range(sequence_length):
        # TODO: Improve performance by only computing ft contribution if above alpha threshold
        for c in range(input_size):
            newInput = input.clone()
            i1, i2 = (t, c) if ft_dim_last else (c, t)
            newInput[:, i1, i2] = assignment

            if hasBaseline is None:
                inputGrad_perInput = Grad.attribute(newInput, target=TestingLabel).data.cpu().numpy()
            else:
                if hasFeatureMask is not None:
                    inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline, target=TestingLabel,
                                                        feature_mask=hasFeatureMask).data.cpu().numpy()
                elif hasSliding_window_shapes is not None:
                    inputGrad_perInput = Grad.attribute(newInput, sliding_window_shapes=hasSliding_window_shapes,
                                                        baselines=hasBaseline,
                                                        target=TestingLabel).data.cpu().numpy()
                else:
                    inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline,
                                                        target=TestingLabel).data.cpu().numpy()

            inputGrad_perInput = np.absolute(ActualGrad - inputGrad_perInput)
            inputGrad[:, c] = np.sum(inputGrad_perInput, axis=(1, 2))
        featureContribution = preprocessing.minmax_scale(inputGrad, axis=-1)
        feature_contributions[:, t, :] = featureContribution

        for c in range(input_size):
            i1, i2 = (t, c) if ft_dim_last else (c, t)
            newGrad[:, i1, i2] = timeContribution[:, t] * featureContribution[:, c]

    return newGrad, time_contributions, feature_contributions if return_time_ft_contributions else newGrad


def get_tsr_saliency(saliency, input, labels, baseline=None, inputs_to_graph=(), graph_dir=None,
                     graph_name='TSR', cur_batch=None, ft_dim_last=True):
    batch_size, num_timesteps, num_features = input.shape

    TSR_attributions, time_contributions, ft_contributions = getTwoStepRescaling(saliency, input, labels, hasBaseline=baseline,
                                                                                 return_time_ft_contributions=True, ft_dim_last=ft_dim_last)

    assert len(inputs_to_graph) == 0 or (graph_dir is not None and cur_batch is not None)
    for index in inputs_to_graph:
        index_within_batch = index - batch_size * cur_batch
        if 0 <= index_within_batch < batch_size:
            pathlib.Path(graph_dir).mkdir(parents=True, exist_ok=True)
            plotExampleBox(TSR_attributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_attr', greyScale=True, flip=True)
            plotExampleBox(time_contributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_time_cont', greyScale=True, flip=True)
            plotExampleBox(ft_contributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_ft_cont', greyScale=True, flip=True)

    return givenAttGetRescaledSaliency(num_timesteps, num_features, TSR_attributions, isTensor=False)
