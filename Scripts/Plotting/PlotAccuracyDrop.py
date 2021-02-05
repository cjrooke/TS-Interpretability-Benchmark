import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

from .. import Helper

def getSamples(args,fileName):


  Box_model= Helper.load_CSV(fileName)
  Saliency_rows=[]

  if(args.GradFlag):
      Saliency_rows.append(0)
  if(args.IGFlag):
      Saliency_rows.append(1)
  if(args.DLFlag):
      Saliency_rows.append(2)
  if(args.GSFlag):
      Saliency_rows.append(3)
  if(args.DLSFlag):
      Saliency_rows.append(4)
  if(args.SGFlag):
      Saliency_rows.append(5)
  if(args.ShapleySamplingFlag):
      Saliency_rows.append(6)
  if(args.FeaturePermutationFlag):
      Saliency_rows.append(7)
  if(args.FeatureAblationFlag):
      Saliency_rows.append(8)
  if(args.OcclusionFlag):
      Saliency_rows.append(9)
  Saliency_rows.append(10)


  Box_model=Box_model[Saliency_rows,1:]

  return  Box_model





index=[i for i in range(0,101,10)]


colors = [
          "red",
          "green",
          "cyan",
          "blue",
          "purple" , 
          "lime" , 
          "orange",
          "deeppink",
          "maroon",
          "pink",
          "brown",
          "black"]


DatasetsTypes= ["Moving_SmallMiddle","Middle","SmallMiddle", "Moving_Middle",  "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
DatasetsNames=['Small Moving Box', 'Middle Box', 'Small Middle Box','Moving Box','Rare Time','Moving Rare Time','Rare Feature', 'Moving Rare Features', 'Time Postional Box', 'Feature Postional Box']
DataGenerationTypes=[None,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]
DataGenerationNames=['Gaussian','Harmonic','Gaussian \n Process', 'Pseudo\nPeriodic','AR','Continuous\nAR','NARMA']
models=["LSTM" ,"LSTMWithInputCellAttention","TCN","Transformer"]

models_name=["LSTM","LSTM+\nin.cell At.","TCN","Transformer"]


def main(args):

    Saliency_Methods=[]

    if(args.GradFlag):
        Saliency_Methods.append("Grad")
    if(args.IGFlag):
        Saliency_Methods.append("IG")
    if(args.DLFlag):
        Saliency_Methods.append("DL")
    if(args.GSFlag):
        Saliency_Methods.append("GS")
    if(args.DLSFlag):
        Saliency_Methods.append("DLS")
    if(args.SGFlag):
        Saliency_Methods.append("SG")
    if(args.ShapleySamplingFlag):
        Saliency_Methods.append("SVS")
    if(args.FeaturePermutationFlag):
        Saliency_Methods.append("FP")
    if(args.FeatureAblationFlag):
        Saliency_Methods.append("FA")
    if(args.OcclusionFlag):
        Saliency_Methods.append("FO")
    Saliency_Methods.append("Random")

    for x in range(len(DatasetsTypes)):


      #figsize Wxh
      row=len(DataGenerationTypes)
      measurmentsCount=len(models)

      fig, (DS) = plt.subplots(row, measurmentsCount,sharex=True, figsize=(measurmentsCount*3+1,2*row))
      print("Plotting", DatasetsTypes[x])

      for y in range(len(DataGenerationTypes)):
          args.DataGenerationProcess=DataGenerationTypes[y]
          

          if(DataGenerationTypes[y]==None):
              args.DataName=DatasetsTypes[x]+"_Box"
          else:
              args.DataName=DatasetsTypes[x]+"_"+DataGenerationTypes[y]


          for m in range(len(models)):
              FileName=args.Masked_Acc_dir + args.DataName+"_"+models[m]+"_0_10_20_30_40_50_60_70_80_90_100_percentSal_rescaled.csv"
              first_row_flag=True
              try:
                data = getSamples(args,FileName)
                # print(data.shape)
                for i in range(data.shape[0]):
                  if(i==len(Saliency_Methods)-1):
                     DS[y,m].plot(index,data[i,:],color = colors[i],label=Saliency_Methods[i])
                  else:

                    DS[y,m].plot(index,data[i,:],color = colors[i],linestyle=':',label=Saliency_Methods[i])

                  DS[y,m].set_ylim([0, 100])
                if(y==0):
                  DS[y,m].set_title(models_name[m],fontsize=16)

                if(not first_row_flag):
                    DS[y,m].tick_params(labelleft=False)  
                else:
                  first_row_flag=False
                if(m==len(models)-1):
                    DS[y,m].set_ylabel(DataGenerationNames[y], fontsize=16)
                    DS[y,m].yaxis.set_label_position("right")
                if(y!=len(DataGenerationTypes)-1):
                    DS[y,m].tick_params(labelbottom=False)
              except:
                print("ignoring",args.DataName+"_"+models[m]  )
                DS[y,m].set_yticklabels([])
                DS[y,m].set_xticklabels([])
                DS[y,m].set_xticks([])
                DS[y,m].set_yticks([])

                DS[y,m].set_ylim([0, 100])

                if(m==len(models)-1):
                  DS[y,m].set_ylabel(DataGenerationNames[y], fontsize=16)
                  DS[y,m].yaxis.set_label_position("right")

                if(y==0):
                  DS[y,m].set_title(models_name[m],fontsize=16)
                continue

      handles, labels = DS[-1,-1].get_legend_handles_labels()
      fig.legend(handles, labels, loc='center right',fontsize=16)

      # # fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0),
      # #           fancybox=True, shadow=True, ncol=len(labels), fontsize=25)
      # #(left, bottom, right, top)

      fig.tight_layout(rect=[0.11, 0.15,0.85,0.95])

      fig.text(0.5, 0.11, '% of features masked', ha='center',fontsize=16)
      fig.text(0.09, 0.5, 'Model Accuracy', va='center', rotation='vertical',fontsize=16)
      fig.suptitle(DatasetsNames[x],fontsize=22)

      plt.savefig(args.Graph_dir+DatasetsTypes[x]+'_AccuracyDrop_rescaled.png',  bbox_inches = 'tight',pad_inches = 0)

   




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--NumTimeSteps',type=int,default=50)
    parser.add_argument('--NumFeatures',type=int,default=50)
    parser.add_argument('--DataGenerationProcess', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default="Datasets/")
    parser.add_argument('--Mask_dir', type=str, default='Results/Saliency_Masks/')
    parser.add_argument('--Masked_Acc_dir', type=str, default= "Results/Masked_Accuracy/")

    parser.add_argument('--Saliency_Distribution_dir', type=str, default= "Results/Saliency_Distribution/")
    parser.add_argument('--Saliency_dir', type=str, default='Results/Saliency_Values/')



    parser.add_argument('--GradFlag', type=bool, default=True)
    parser.add_argument('--IGFlag', type=bool, default=True)
    parser.add_argument('--DLFlag', type=bool, default=True)
    parser.add_argument('--GSFlag', type=bool, default=True)
    parser.add_argument('--DLSFlag', type=bool, default=True)
    parser.add_argument('--SGFlag', type=bool, default=True)
    parser.add_argument('--ShapleySamplingFlag', type=bool, default=True)
    parser.add_argument('--FeaturePermutationFlag', type=bool, default=True)
    parser.add_argument('--FeatureAblationFlag', type=bool, default=True)
    parser.add_argument('--OcclusionFlag', type=bool, default=True)


    parser.add_argument('--Graph_dir', type=str, default='Graphs/Accuracy_Drop/')

    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
plt.clf()