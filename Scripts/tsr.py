import pathlib

import numpy as np
from sklearn import preprocessing

import Helper
from Plotting import plotExampleBox


def getTwoStepRescaling(Grad, input, sequence_length, input_size, TestingLabel, hasBaseline=None, hasFeatureMask=None,
                        hasSliding_window_shapes=None, return_time_ft_contributions=False):
    assignment = input[0, 0, 0]
    timeGrad = np.zeros(sequence_length)
    inputGrad = np.zeros(input_size)
    newGrad = np.zeros((sequence_length, input_size))
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

    for t in range(sequence_length):
        timeGrad[t] = np.mean(np.absolute(ActualGrad[0, t, :]))

    timeContribution = preprocessing.minmax_scale(timeGrad, axis=0)
    meanTime = np.quantile(timeContribution, .55)

    time_contributions = np.zeros((sequence_length, input_size))
    time_contributions[:] = timeContribution[:, None]
    feature_contributions = np.zeros((sequence_length, input_size))

    for t in range(sequence_length):
        if timeContribution[t] > meanTime:
            for c in range(input_size):
                newInput = input.clone()
                newInput[:, t, c] = assignment

                if (hasBaseline == None):
                    inputGrad_perInput = Grad.attribute(newInput, target=TestingLabel).data.cpu().numpy()
                else:
                    if (hasFeatureMask != None):
                        inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline, target=TestingLabel,
                                                            feature_mask=hasFeatureMask).data.cpu().numpy()
                    elif (hasSliding_window_shapes != None):
                        inputGrad_perInput = Grad.attribute(newInput, sliding_window_shapes=hasSliding_window_shapes,
                                                            baselines=hasBaseline,
                                                            target=TestingLabel).data.cpu().numpy()
                    else:
                        inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline,
                                                            target=TestingLabel).data.cpu().numpy()

                inputGrad_perInput = np.absolute(ActualGrad - inputGrad_perInput)
                inputGrad[c] = np.sum(inputGrad_perInput)
            featureContribution = preprocessing.minmax_scale(inputGrad, axis=0)
        else:
            featureContribution = np.ones(input_size) * 0.1
        for c in range(input_size):
            newGrad[t, c] = timeContribution[t] * featureContribution[c]
        feature_contributions[t, :] = featureContribution
    return newGrad, time_contributions, feature_contributions if return_time_ft_contributions else newGrad


def get_tsr_saliency(saliency, input, num_features, num_timesteps, labels, baseline=None, inputs_to_graph=(), graph_dir=None,
                     graph_name='TSR', cur_batch=None):
    batch_size = input.shape[0]
    TSR_attributions = np.zeros(input.shape)
    time_contributions = np.zeros(input.shape)
    ft_contributions = np.zeros(input.shape)

    if baseline is not None:
        baseline = baseline.reshape(batch_size, -1, num_timesteps, num_features)

    for i in range(batch_size):
        cur_baseline = baseline[i] if baseline is not None else None
        TSR_attributions[i], time_contributions[i], ft_contributions[i] = \
            getTwoStepRescaling(saliency, input[i:i+1], num_timesteps, num_features, labels[i:i+1], hasBaseline=cur_baseline, return_time_ft_contributions=True)

    assert len(inputs_to_graph) == 0 or (graph_dir is not None and cur_batch is not None)
    for index in inputs_to_graph:
        index_within_batch = index - batch_size * cur_batch
        if 0 <= index_within_batch < batch_size:
            pathlib.Path(graph_dir).mkdir(parents=True, exist_ok=True)
            plotExampleBox(TSR_attributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_attr', greyScale=True, flip=True)
            plotExampleBox(time_contributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_time_cont', greyScale=True, flip=True)
            plotExampleBox(ft_contributions[index_within_batch], f'{graph_dir}/{graph_name}_{index}_ft_cont', greyScale=True, flip=True)

    return Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, TSR_attributions, isTensor=False)
