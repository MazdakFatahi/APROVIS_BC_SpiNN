import os
import shutil
import numpy as np
import numpy as np
from tonic import Dataset, transforms
import torch
from torch.utils.data import DataLoader, random_split

def uniform_noise_numpy(events: np.ndarray, sensor_size, n: int):
    """Adds a fixed number of noise events that are uniformly distributed across sensor size
    dimensions.

    Parameters:
        events: ndarray of shape (n_events, n_event_channels)
        sensor_size: 3-tuple of integers for x, y, p
        n: the number of noise events added.
    """
    noise_events = np.zeros(n, dtype=events.dtype)
    for channel in events.dtype.names:
        if channel == "x":
            low, high = 0, sensor_size[0]
        if channel == "y":
            low, high = 0, sensor_size[1]
        if channel == "p":
            low, high = 0, sensor_size[2]
        if channel == "t":
            low, high = events["t"].min(), events["t"].max()
        noise_events[channel] = np.random.uniform(low=low, high=high, size=n)
    noisy_events = np.concatenate((events, noise_events))
    return noisy_events[np.argsort(noisy_events["t"])]



def downscale(events, SCALEDOWN = 4, X_NUM = 32, Y_NUM = 32):
    # For a single event, use as data[1:2].shape
    # if events.shape[0]==1:
    #     return events[0]//SCALEDOWN+X_NUM*(events[1]//SCALEDOWN)+X_NUM*Y_NUM*events[3]
    # else:
    return (events[:,0]//SCALEDOWN+X_NUM*(events[:,1]//SCALEDOWN)+X_NUM*Y_NUM*events[:,3]).astype(int)
class AprovisDataSet(Dataset):
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    
    # dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    def __init__(
        self,
        same_size_classes = False,
        src_path = 'SrcData',
        save_path = 'NewData',  
        downsample = False, 
        max_rate = None,
        SCALEDOWN = 4, X_NUM = 32, Y_NUM = 32, 
        noise = None, 
        classes = ["sea", "gro"], 
        time_window = 10000000, 
        temporal_scale_factor = 1,
        verbose =True, sensor_size=[128,128,2],
        transform=transforms.NumpyAsType(int),
        target_transform=None, 
    ):
        super(AprovisDataSet, self).__init__(
            save_to='./', transform=transform, target_transform=target_transform
        )
        # self.train = train
        self.same_size_classes = same_size_classes
        self.sensor_size = sensor_size.copy()
        self.save_path = save_path 
        self.src_path = src_path 
        self.classes = classes
        self.time_window = time_window 
        self.temporal_scale_factor = temporal_scale_factor,
        self.verbose = verbose 
        self.downsample = downsample
        self.SCALEDOWN = SCALEDOWN
        self.X_NUM = X_NUM
        self.Y_NUM = Y_NUM
        self.max_rate = max_rate
        self.noise = noise
        print(f'same_size_classes: {self.same_size_classes}')
        # replace the strings with your training/testing file locations or pass as an argument
        
        self.data, self.targets, self.n_total_samples_in_cls = prepar_data(same_size_classes = self.same_size_classes, 
                                                                            save_path = self.save_path, 
                                                                            src_path = self.src_path, 
                                                                            downsample = self.downsample, 
                                                                            max_rate = self.max_rate, 
                                                                            SCALEDOWN = self.SCALEDOWN, X_NUM = self.X_NUM, Y_NUM = self.Y_NUM ,  
                                                                            noise=self.noise, 
                                                                            classes = self.classes, 
                                                                            time_window = self.time_window, 
                                                                            temporal_scale_factor = self.temporal_scale_factor,
                                                                            verbose =self.verbose)
        # print(f'len(data) in dataset object: {len(self.data)}')
    def __getitem__(self, index):
        if self.downsample == True:
            events, target = self.data[index], self.targets[index]
            # print('No tonic transfor is available ')
        else:
            # events = np.load(self.filenames[index])
            events, target = self.data[index], self.targets[index]
            # print(f'In dataset object events.shape: {events.shape}')
            # print(f'In dataset object target.shape: {target.shape}')
                    #this line is used in tonic package, keep and see if needed:
            events = np.lib.recfunctions.unstructured_to_structured(np.array(events), self.dtype)
            # events = np.lib.recfunctions.unstructured_to_structured(np.array(event_stream), dtype)

            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # print(f'In dataset object events.shape: {events.shape}')
            # print(f'In dataset object target.shape: {target.shape}')


        return events, target

    def __len__(self):
        return len(self.data)
def prepar_data(same_size_classes = False, save_path = 'NewData', src_path = 'SrcData', 
                downsample = False, max_rate = None, 
                SCALEDOWN = 4, X_NUM = 32, Y_NUM = 32, 
                noise=None, 
                classes = ["sea", "gro"], 
                time_window = 1000000, 
                temporal_scale_factor = None,
                verbose =True):
# npy_array = np.load(f'Data/only_gro_sessions/DVS128-2023-03-24T08-47-28+0200-Bern13XX-0.npy')
    # print(f'in prepar_data: same_size_classes: {same_size_classes}')
    if  os.path.exists(f'{save_path}'):
        shutil.rmtree(f'{save_path}')
    os.mkdir(f'{save_path}')

    for cls in classes:
        if  os.path.exists(f'{save_path}/{cls}'):
            shutil.rmtree(f'{save_path}/{cls}')
        os.mkdir(f'{save_path}/{cls}')

    int_classes = dict(zip(classes, range(len(classes))))
    if verbose: print(int_classes)
    data = []
    targets = []
    file_names = []
    paths = []
    sub_data = []
    n_samples_from_file = []
    reminder_size = []
    n_total_samples = 0
    n_total_samples_in_cls = dict(zip(classes, (0, 0)))
    if verbose: print(f'n_total_samples_in_cls: {n_total_samples_in_cls}')
    for path, _, files in os.walk(src_path):
        files.sort()
        for i, file in enumerate(files):
            if file.endswith("npy"):
                if verbose: print(f'====================={file}===================')
                if verbose: print(f"class :{path.split('_')[1]}")
                # if verbose: print(path[-6:])
                # if verbose: print(path[1:])
                data.append(np.load(os.path.join(path, file)))
                # targets.append(self.int_classes[path[-3:]])
                cls = path.split('_')[1]
                # targets.append(int_classes[cls])#path[-6:]])
                file_names.append(file)
                paths.append(path)


                if noise != None: # Using a a function from tonic, will added 'n' unoform events (a uniform events with x,y,t,p ) to all the events from the current loaded npy file
                                  # n is a fraction of the 'noise' parameter
                    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
                    ordering = dtype.names
                    # print(len(data[i]))
                    # print(np.floor(noise*(len(data[i]))).astype(np.int32))
                    events = np.lib.recfunctions.unstructured_to_structured(np.array(data[i]), dtype)
                    data[i] = uniform_noise_numpy(events, sensor_size=[128,128,2], n=np.floor(noise*(len(data[i]))).astype(np.int32))
                    # print(type(data[i]))
                    data[i] = np.array(data[i].tolist())
                    # print(type(data[i]))
                
                
                if temporal_scale_factor != None:
                    data[i][:,2] = np.floor(data[i][:,2]/temporal_scale_factor)

                if verbose: print(data[i].shape)
                max_x = int(max(data[i][:,0]))
                max_y = int(max(data[i][:,0]))
                if verbose: print(f'max_x: {max_x}')
                if verbose: print(f'max_y: {max_y}')
                start_time = int(min(data[i][:,2]))
                if verbose: print(f'start_time : {start_time }')
                end_time = int(max(data[i][:,2]))
                if verbose: print(f'end_time : {end_time }')



                
                n_samples = int((end_time- start_time)// time_window)# NOTE: It's deviding based on the time window (using the timestamps) not based on the number of events
                reminder_size.append(end_time - n_samples*time_window)
                if verbose: print(f'reminder_size_{i} : {reminder_size[i]}')
                if verbose: print(f'n_samples : {n_samples}')

                if reminder_size[i]>0:
                    n_samples+=1
                n_samples_from_file.append(n_samples)
                if verbose: print(f'n_samples : {n_samples}')
                # (events_patch[:,t_index]>=t*time_limit)&(events_patch[:,t_index]<(t+1)*time_limit)
                for s in range(n_samples):
                    inxs = (data[i][:,2]>=s*time_window )& (data[i][:,2]<(s+1)*time_window)
                    if verbose: print(f'last row of data[{i}] : {data[i][inxs][-1]}  len data[{i}: {len(data[i][inxs])} ]')# for s in range(n_samples)] }')
                    if downsample == False:# convert to `flatten frames` instead of `sliced events-stream `
                        sub_data.append( data[i][(data[i][:,2]>=s*time_window )& (data[i][:,2]<(s+1)*time_window)])
                    else:
                        if max_rate !=None:# add up the frames by step size as `step` (add up the `1`s from each frame at it's position )
                            # print(f'max_rate = {max_rate}')
                            # time_window = 1/max_rate
                            step = np.ceil((1/max_rate)*1e9).astype(int)

                            # x_1 = torch.zeros(X_NUM*Y_NUM*2)
                            # tmp = downscale(data[i][(data[i][:,2]>=s*time_window )& (data[i][:,2]<(s+1)*time_window)])

                            d = data[i][(data[i][:,2]>=s*time_window )& (data[i][:,2]<(s+1)*time_window)]

                            x_1 = torch.zeros(X_NUM*Y_NUM*2)
                            
                            for t in range(min(d[:,2]), max(d[:,2]),step):
                                tmp = torch.zeros(X_NUM*Y_NUM*2)
                                tmp[downscale(d[(t<=d[:,2])&(d[:,2]< (t+step))],SCALEDOWN = SCALEDOWN, X_NUM = X_NUM, Y_NUM = Y_NUM )]=1
                                x_1+=tmp
                            sub_data.append( x_1)

                        else:
                            # print(f'max_rate = {max_rate}')
                            x_1 = torch.zeros(X_NUM*Y_NUM*2)
                            tmp = downscale(data[i][(data[i][:,2]>=s*time_window )& (data[i][:,2]<(s+1)*time_window)], SCALEDOWN = SCALEDOWN, X_NUM = X_NUM, Y_NUM = Y_NUM )
                            x_1[tmp]=1
                            sub_data.append( x_1)
                    targets.append(int_classes[cls])#path[-6:]])
                saved_data = [np.save(f'{save_path}/{cls}/{n_total_samples_in_cls[cls]+s}.npy',
                                    data[i][(data[i][:,2]>=s*time_window )& (data[i][:,2]<(s+1)*time_window)]) 
                                    for s in range(n_samples)]
                # saved_data.append(np.save(f'{save_path}/{cls}/{n_samples}.npy',data[i][n_samples*time_window:end_time,:]))
                
                if verbose: print(f'len(saved_data) = {len(saved_data)}')
                n_total_samples_in_cls[cls]+= n_samples
                n_total_samples += n_samples
                if verbose: print('=================================')

                # break
        # break

    if verbose: print(f'n_total_samples_in_cls:{n_total_samples_in_cls}')
    # if verbose: print(f'sub_data : {sub_data}')
    if verbose: print(f'number of events in each sub_data : {[len(sub_data[i]) for i in range(len(sub_data))]}')
    if verbose: print(f'n_total_samples:{n_total_samples}')
    if verbose: print(targets)
    npy_files_info = dict(zip(file_names, paths))
    if verbose: print(f'npy_files_info:{npy_files_info}')
    # print(same_size_classes)
    print(f'n_total_samples_in_cls 1: {n_total_samples_in_cls}')
    if same_size_classes == True:
        # print('in if 1')
        # print(targets)
        n_samples_per_class = [len(np.where(np.array(targets)== c)[0]) for c in range(len(int_classes))]
        # print(f'n_samples_per_class: {n_samples_per_class}')
        inx_class_samples_in_all_data = [np.where(np.array(targets)== c)[0] for c in range(len(int_classes))]
        # print(f'inx_class_samples_in_all_data: {inx_class_samples_in_all_data}')
        arg_min_n_samples_per_class = np.argmin(n_samples_per_class)
        # print(f'arg_min_n_samples_per_class: {arg_min_n_samples_per_class}')
        min_n_samples_per_class = np.min(n_samples_per_class)
        # print(f'min_n_samples_per_class: {min_n_samples_per_class}')
        # print(f'inx_class_samples_in_all_data[0][:min_n_samples_per_class]: {inx_class_samples_in_all_data[0][:min_n_samples_per_class]}')
        selected_indexes_with_same_size_per_class = [inx_class_samples_in_all_data[c][:min_n_samples_per_class] for c in range(len(int_classes))]
        # print(len(selected_indexes_with_same_size_per_class))
        # print(f'selected_indexes_with_same_size_per_class: {selected_indexes_with_same_size_per_class}')
        new_inxs = np.concatenate(selected_indexes_with_same_size_per_class)
        # print(f'new_inxs: {new_inxs}')
        # for cls in range(len(int_classes)):
        # print(f'same_size_classes=True: {len(sub_data)}')
        sub_data = np.array(sub_data)[new_inxs]
        # print(f'same_size_classes=True: {len(sub_data)}')
        targets = np.array(targets)[new_inxs]
        for c in int_classes:
            n_total_samples_in_cls[c] = min_n_samples_per_class 
    # print(len(sub_data))
        # print('in if 2')
    print(f'n_total_samples_in_cls 2: {n_total_samples_in_cls}')
    return sub_data, targets, n_total_samples_in_cls

# data, targets, n_total_samples_in_cls = prepar_data(same_size_classes=True, verbose=False)
# data.shape

def split_train(full_dataset, train_fraction,transform):
    # print(train_data_folder)
    # data_train = full_dataset#ImageFolder(train_data_folder, transform)
    fractions = [int(train_fraction*len(full_dataset)),len(full_dataset)-int(train_fraction*len(full_dataset))]
    print(len(full_dataset))
    print(f'fractions: {fractions}')
    train, test = random_split(full_dataset, fractions, generator=torch.Generator().manual_seed(42))
    return train, test
