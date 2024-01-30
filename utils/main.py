import math
from tqdm import tqdm
from os import listdir
from random import *
seed(0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import wandb

def load_data(batch_size, images_tensor, d_in, d_out):
    in_batch, out_batch = [], []
    for i in range(batch_size):
        start_point = choice(range(len(images_tensor) - d_in - d_out))
        in_batch.append(images_tensor[start_point:start_point+d_in])
        out_batch.append(images_tensor[start_point+d_in:start_point+d_in+d_out])
    return torch.stack(in_batch), torch.stack(out_batch)

def train(model,
          images_tensor_train,
          images_tensor_test,
          alpha,
          ultimate_mask,
          criterion,
          d_in, d_out,
          epochs,
          batch_size,
          lr,
          eval_step,
          is_persistance):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()
#weight_decay=1e-8, momentum=0.9
#    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
#    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    for epoch in range(epochs):
        model.train()
        
        if not is_persistance:
            images_in, images_out = load_data(batch_size, images_tensor_train, d_in, d_out)
            optimizer.zero_grad()
            model_out = alpha*-model(images_in) + images_in[:,-1, None, :, :].repeat(1, 3, 1, 1)
            
            adapted_mask = ~ultimate_mask[None, None, :].repeat(batch_size,3,1,1)
            
            loss = criterion(model_out[adapted_mask], images_out[adapted_mask])
            loss.backward() # retain_graph=True
        
#        print(model_out.shape, images_out.shape)
        
            optimizer.step()
        
        if epoch % eval_step == 0:
            with torch.no_grad():
                model.eval()

                mae_total, rmse_total, mape_total = [], [], []

                total_test_len = len(images_tensor_test)
                start_point, steps = 0, int(total_test_len/(d_in + d_out))

                for i in range(steps):
                    images_in, images_out = images_tensor_test[start_point:start_point+d_in],\
                                            images_tensor_test[start_point+d_in:start_point+d_in+d_out]

                    start_point += d_in + d_out

                    if is_persistance:
                        model_out = images_in[-1][None, :, :].repeat(3, 1, 1)[None, :, : , :]
                    else:
                        model_out = alpha*model(images_in[None, :, :, :]) + images_in[-1].repeat(3, 1, 1)[None, :, :, :] 

                    #                print(model_out.shape)
                
                    adapted_mask = ~ultimate_mask[None, None, :].repeat(1,3,1,1)

                    loss_mse = criterion_mse(model_out[adapted_mask], images_out[None, :, :, :][adapted_mask])
                    loss_mae = criterion(model_out[adapted_mask], images_out[None, :, :, :][adapted_mask])

                    mae_total.append(loss_mae.detach().cpu().numpy())
                    rmse_total.append(math.sqrt(loss_mse.detach().cpu().numpy()))
                
            
                wandb.log(
                    {
                     "test/test_MAE": np.average(mae_total), 
                     "test/test_RMSE": np.average(rmse_total)
                    }
                )
            
            print('Epoch ', epoch, ', test MAE - ', np.average(mae_total))