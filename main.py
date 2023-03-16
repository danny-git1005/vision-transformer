import torch
import torch.nn as nn
from torch.utils.data import DataLoader , random_split
from torch.cuda import amp
from utils import get_data , train , valid
from DataClass import BirdDataset
from vit import vit
from torch.utils.tensorboard import SummaryWriter
from vit_pytorch import ViT
from torchsummary import summary

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    
if __name__ == "__main__":
    torch.manual_seed(torch.initial_seed())

    train_dir = '/data/practice_data/birds/train/'
    val_dir   = '/data/practice_data/birds/valid/'
    test_dir  = '/data/practice_data/birds/test/'

    train_imgs , valid_imgs , test_imgs , classInt = get_data( train_dir=train_dir , val_dir=val_dir , test_dir=test_dir )
    train_dataset = BirdDataset( train_imgs , classInt , transforms=None )
    val_dataset   = BirdDataset( valid_imgs , classInt , transforms=None )
    test_dataset  = BirdDataset( test_imgs  , classInt , transforms=None )

    # model.load_state_dict(torch.load('model_35_93.942.pt'))
    model         = vit(
                        img_shape = 224,
                        patch_size = 32,
                        classes = 500,
                        dim = 1024,
                        block_depth = 6,
                        heads = 16,
                        head_dim = 64,
                        mlp_dim = 2048,
                        channels=3,
                        dropout = 0.2,
                        emb_dropout = 0.2
                    ).to(device)
    
    # model         = ViT(
    #                     image_size = 224,
    #                     patch_size = 32,
    #                     num_classes = 500,
    #                     dim = 1024,
    #                     depth = 6,
    #                     heads = 16,
    #                     dim_head = 64,
    #                     mlp_dim = 2048,
    #                     channels=3,
    #                     dropout = 0.1,
    #                     emb_dropout = 0.1
    #                 ).to(device)    

    summary( model, (3,224,224) )
    # model.to(device)
    batch         = 48
    epoch         = 30
    learning_rate = 0.0001

    train_loader = DataLoader( train_dataset , batch_size=batch , shuffle=True , num_workers=2 , pin_memory=True , drop_last=True)
    val_loader   = DataLoader( val_dataset   , batch_size=batch  , shuffle=True , num_workers=2 , pin_memory=True , drop_last=True)

    loss      = nn.CrossEntropyLoss(reduction='mean' , label_smoothing=0.1)
    loss_test = nn.CrossEntropyLoss(reduction='mean')

    optimizer    = torch.optim.AdamW(model.parameters() , lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.995, patience=8, threshold=0.0001, eps=1e-8)
  

    writer = SummaryWriter( log_dir = '/code/tb_logs/mytransformer_lr-' + str(learning_rate)+
                                                  '_batch-' + str(batch)+ 
                                                  '_label_smooth-0.1'+
                                                  '_accumlate-2_ep1')

    for i in range(epoch):
        print('epoch:{}'.format(i))
        scaler = amp.GradScaler()
 
        train_loss , train_acc = train( dataloader=train_loader , model=model , loss_function=loss , optimizer=optimizer , lr_scheduler=lr_scheduler , scaler=scaler  )
        val_loss , val_acc = valid( dataloader=val_loader , model=model , loss_function=loss_test )


        writer.add_scalar( "Accuracy / Train" , train_acc  , i )
        writer.add_scalar( "Loss     / Train" , train_loss , i )
        writer.add_scalar( "Accuracy / val"   , val_acc    , i )
        writer.add_scalar( "Loss     / val"   , val_loss   , i )

        # writer.flush()

        # torch.save({
        #     'epoch': i,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        #     }, './train_ckpt/')


        if val_acc > 85 :
            modelname = 'model_' + str(i) + '.pt'
            torch.save(model.state_dict(), modelname)
            print('Save model {}'.format(i))
            print('\n')

# # nohup python main.py > /dev/null 2>&1
# data:
# https://www.kaggle.com/code/jainamshah17/pytorch-starter-image-classification
# https://www.kaggle.com/code/lonnieqin/bird-classification-with-pytorch
# https://www.kaggle.com/code/stpeteishii/bird-species-classify-torch-conv2d
# https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code?datasetId=534640&searchQuery=torch

# ViT
# https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
# https://medium.com/ai-blog-tw/detr%E7%9A%84%E5%A4%A9%E9%A6%AC%E8%A1%8C%E7%A9%BA-%E7%94%A8transformer%E8%B5%B0%E5%87%BAobject-detection%E6%96%B0pipeline-a039f69a6d5d
# https://juejin.cn/post/6924173141409267726
# https://www.cnblogs.com/wxkang/p/16150868.html
