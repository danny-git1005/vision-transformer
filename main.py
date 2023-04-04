import torch
import torch.nn as nn
from torch.utils.data import DataLoader , random_split
from torch.cuda import amp
from utils import get_data , train , valid
from DataClass import BirdDataset
from ViT import vit
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchsummary import summary
import os
import configparser
# from vit_pytorch import ViT


os.environ['CUDA_VISIBLE_DEVICES'] = '7'

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    
if __name__ == "__main__":
    torch.manual_seed(torch.initial_seed())

    config = configparser.ConfigParser()
    config.read('config_data_bird.ini')
    
    train_dir = config['data']['train_dir']
    valid_dir = None
    test_dir  = None
    if config['data']['val_dir'] != 'None':
        valid_dir = config['data']['val_dir']
    if config['data']['test_dir'] != 'None':
        test_dir  = config['data']['test_dir']
    
    train_imgs , valid_imgs , test_imgs , classInt = get_data( train_dir=train_dir , val_dir=valid_dir , test_dir=test_dir )
    
    train_dataset = BirdDataset( train_imgs , classInt , transforms=None )
    if valid_imgs and test_imgs:
        val_dataset   = BirdDataset( valid_imgs , classInt , transforms=None )
        test_dataset  = BirdDataset( test_imgs  , classInt , transforms=None )
    else:
        train_size = int(len(train_dataset)*0.8)
        val_size   = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    model = vit(
                img_shape   = config['model'].getint('img_shape'),  
                patch_size  = config['model'].getint('patch_size'),
                dim         = config['model'].getint('dim'),
                classes     = config['model'].getint('classes'),  
                block_depth = config['model'].getint('block_depth'),
                heads       = config['model'].getint('heads'),
                mlp_dim     = config['model'].getint('mlp_dim'),
                channels    = config['model'].getint('channels'),
                dropout     = config['model'].getfloat('dropout'),
                emb_dropout = config['model'].getfloat('emb_dropout'),
            ).to(device)  

    summary( model, (3,224,224) )
    
    batch         = 144
    epoch         = 400
    learning_rate = 0.0004

    train_loader = DataLoader( train_dataset , batch_size=batch , shuffle=True , num_workers=6 , pin_memory=True , drop_last=True)
    val_loader   = DataLoader( val_dataset   , batch_size=batch  , shuffle=True , num_workers=6 , pin_memory=True , drop_last=True)

    loss      = nn.CrossEntropyLoss(reduction='mean' , label_smoothing=0)
    loss_test = nn.CrossEntropyLoss(reduction='mean')

    optimizer    = torch.optim.AdamW(model.parameters() , lr=learning_rate)
    lr_scheduler = CosineLRScheduler(
                        optimizer      = optimizer, 
                        t_initial      = config['cosLrScheduler'].getint('t_initial'),
                        warmup_t       = config['cosLrScheduler'].getint('warmup_t'), 
                        warmup_lr_init = config['cosLrScheduler'].getfloat('warmup_lr_init'),
                        k_decay        = config['cosLrScheduler'].getfloat('k_decay'),
                        lr_min         = config['cosLrScheduler'].getfloat('lr_min')
                    )
  
    writer = SummaryWriter( log_dir = '/code/tb_logs/MyVit_lr-' + str(learning_rate)+
                                                  '_batch-' + str(batch)+ 
                                                  '_label_smooth-0'+
                                                  '_CosineLRScheduler'+
                                                  '_accumlate-2_ep3')


    for i in range(epoch):
        print('epoch:{}/{}'.format(i,epoch))
        scaler = amp.GradScaler()
 
        train_loss , train_acc = train( dataloader=train_loader , model=model , loss_function=loss , optimizer=optimizer , lr_scheduler=lr_scheduler , scaler=scaler  )
        val_loss   , val_acc   = valid( dataloader=val_loader , model=model , loss_function=loss_test )


        writer.add_scalar( "Accuracy / Train" , train_acc  , i )
        writer.add_scalar( "Loss     / Train" , train_loss , i )
        writer.add_scalar( "Accuracy / val"   , val_acc    , i )
        writer.add_scalar( "Loss     / val"   , val_loss   , i )

        if val_acc > 0.6 :
            modelname = './model/model_' + str(i) + '.pt'
            torch.save(model.state_dict(), modelname)
            print('Save model {}'.format(i))
            print('\n')

# # nohup python main.py > /dev/null 2>&1
