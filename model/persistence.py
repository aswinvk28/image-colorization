import torch

def save_model(net, optimizer, epoch, batch_num):
    torch.save({
        'model_state_dict': net.state_dict(),
    }, 'checkpoints/img_color-'+str(epoch)+'_'+str(batch_num)+'.checkpoint.pth')

def load_model(net, optimizer, epoch, batch_num):
    checkpoint = torch.load('checkpoints/img_color-'+str(epoch)+'_'+str(batch_num)+'.checkpoint.pth')
    net.load_state_dict(checkpoint['model_state_dict'])

    return net