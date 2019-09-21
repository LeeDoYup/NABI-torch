import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable


def bce_with_logits(logits, gts):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, gts)
    return loss


def train(model, train_loader, eval_loader, save_path, args):
    num_epoch = args.epoch
    utils.create_dir(save_path)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(save_path, 'log.txt'))
    
    best_eval_loss = 0
    if os.path.exists(os.path.join(save_path, 'model.pth')):
        checkpoint = torch.load(os.path.join(save_path, 'model.pth'))
        model.load_state_dict(checkpoint)
        print('[*] Saved model is loaded:\t', save_path+'/model.pth')
    
    for epoch in range(num_epochs):
        total_loss = 0
        t = time.time()

        for (vital, demo, gt) in enumerate(train_loader):
            vital = Variable(vital).cuda()
            demo = Variable(demo).cuda()
            gt = Variable(gt).cuda()

            pred = model(vital, demo)
            loss = bce_with_logits(pred, gt)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_loss += loss.data * v.size(0)

        total_loss /= len(train_loader.dataset)
        model.train(False)
        eval_loss, _ = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f' % (total_loss))
        
        print('epoch %d, time: %.2f' %(epoch, time.time()-t))
        print('\ttrain_loss: %.2f, \teval_loss: %.2f' %(total_loss, eval_loss))
        if eval_loss < best_eval_loss:
            model_path = os.path.join(save_path, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_loss = eval_loss

def evaluate(model, dataloader, reload=False, save_path=None):
    if reload:
        try:
            checkpoint = torch.load(os.path.join(save_path, 'model.pth'))
            model.load_state_dict(checkpoint)
            print('[*] Saved model is loaded:\t', save_path+'/model.pth')
        except 
            raise 

    loss = 0
    num_data = 0
    preds = []
    for vital, demo, gt in iter(dataloader):
        vital = Variable(vital, volatile=True).cuda() 
        demo = Variable(demo, volatile=True).cuda() 
        gt = Variable(gt, volatile=True).cuda()         
        pred = model(vital, demo)
        preds.extend(pred)
        loss += bce_with_logits(pred, gt)
        
        num_data += pred.size(0)
    
    return loss / num_data, np.array(preds)
