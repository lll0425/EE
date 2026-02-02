#extract.py
import mat73
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from model.resnet import ResNet
from dataset.CISDataset import CISDataset
from cfg_ex import *

def extration(dis):

    formatted_dis = "{:.1f}".format((0.1*(dis)))
    saved_model_path = cfg["saved_model_path"]
    # data_root = cfg['extration_root']
    data_root = '/home/jovyan/jupyter/data_in/FE_out/RawIQ_to_Qmatrix_0to2cm/Q_matrix' #the folder you want to extract the output vector
    # data_root = './data/20PCD1batch1(560,56,f26t7,1cm)'
    
    device = torch.device('cpu')
    dataset = CISDataset(data_root=data_root)
    model = ResNet()
    print(device)
    model.load_state_dict(torch.load(saved_model_path,map_location=device)) #load trained model
    print("model loaded.")

    

    model.eval()
    num = 1
    previous_label = None
    previous_distance = None
    RFF = []

    # for folders in dataset.subfolders:

    #     # folder_path = os.path.join('./data/20PCD1batch11124RFF(6peakRMSprop(0cm),hop112)', str(folders))

    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)    
    # folder_path = os.path.join('./RFF/20sPCD1batch11208RFF((5distance),560,56,smaller,wholefreq,0.1cm)')
        folder_path = '/home/jovyan/jupyter/data_in/FE/EXTRACTOR/RFF/ExpA_0909_TripletLoss_DeepResNet_440E'  # output vector saved path
 #output vector saved path
    # folder_path = os.path.join('./RFF/20PCD1batch10418RFF(560,56,f94t7,padding)')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with torch.no_grad():
        for i in range(len(dataset)):
            CIS, label, distance_folder = dataset[i]  # Include distance folder
            CIS = CIS.unsqueeze(0).to(device)  # CI=1*1*N*M
            embeddings = model(CIS).cpu().numpy()
            
            # Ensure correct path creation and file saving
            if previous_label != label or previous_distance_folder != distance_folder:
                if previous_label is not None and len(RFF) > 0:
                    saved_folder_path = os.path.join(folder_path, str(previous_distance_folder))
                    if not os.path.exists(saved_folder_path):
                        os.makedirs(saved_folder_path)
                    np.save(os.path.join(saved_folder_path, f'{previous_label}.npy'), RFF)
                    print(f'Saving card {previous_label} in {previous_distance_folder}')
                num = 1
            
            if num == 1:
                print(f'Extracting card {label} in {distance_folder}')
                RFF = []            
            
            RFF.append(embeddings)
            num += 1
            previous_label = label
            previous_distance_folder = distance_folder

        # Save the last RFF
        if len(RFF) > 0:
            saved_folder_path = os.path.join(folder_path, str(distance_folder))
            if not os.path.exists(saved_folder_path):
                os.makedirs(saved_folder_path)
            np.save(os.path.join(saved_folder_path, f'{label}.npy'), RFF)
            print(f'Saving card {label} in {distance_folder}')    




def main(): 
    extration(1)
    # for dis in range(1):
    #     extration(dis)

    # extration(1)
if __name__ == '__main__':
    main()
