cfg = {
    'raw_root' : '../raw/',
    'preprocessing_root': '../data',


    ## model config
    'batch_size' : 1024,
    'epoch':2000,
    ## Scheduler setting
    # 'warm_step': 200,
    'lr':0.001,
    # 'milestones': [800, 1200],
    # 'milestones_lr': [5e-6, 2e-6],

    ## dataset config
    # 'train_annot':'../WToutput/DPI60_2train/annot.pkl',
    # 'train_data_root':'../WToutput/DPI60_2train',
    #'train_data_root':'./dataset/20PCD1batch1',

    # D:/午/EXTRACTOR/dataset/0519_Allbrand1234_0to05and12cm_All/Q_matrix
    # H:\0714_data_low1MHz\Q_matrix\Q_matrix_train

    'train_data_root':'D:/午/EXTRACTOR/dataset/0929_AF16_low1MHzDB/Q_matrix', 
    #'train_data_root':'H:/data/0925_AF16_7DISDB_low1M/Q_matrix', 
    'knn_train_data_root':'K:/午/EXTRACTOR_1/RFF/0611_ExpA_AllDis20card_all',
    'saved_optimizer_path' : 'D:/午/EXTRACTOR/save/GRLGaps_0928_3dis_AF16/optim_320.pt',
    'split_ratio': 0.9,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    # 'test_annot':'../WToutput/DPI60_2test/annot.pkl',
    'test_data_root':'../WToutput/DPI60_2test',
    'margin':0.5,
    'RESUME' : True,
    'saved_model_path': 'D:/午/EXTRACTOR/save/GRLGaps_0928_3dis_AF16/model_320.pt', 

    #'saved_optimizer_path' : './save/0221Adam(560,56,f36t7,m0.2,deep)/optimizer_253.pt',

    'seed' : 1206,
    '13' : 13,
}