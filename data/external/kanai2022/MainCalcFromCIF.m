filename = 'Si-CCE1-20210127_5T-(50)';
path = ['E:\0temp\MATLAB\data\' filename '-' char(datetime('now','Format','yyyyMMdd''T''HHmmss'))];
mkdir(path)
batch_density_CCE_v0_from_cif (filename, [path '\']);