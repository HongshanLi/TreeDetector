import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import pickle


def _accuracy():
    return

def _top_one_accuracy():
    return

def _top_five_accuracy():
    return


# for regression related tasks



# checkpoints
def _save_checkpoint(state_dict, ckp_path):
    torch.save(state_dict, ckp_path)
    return
    

# print out progress
def _print(epoch, epochs, step, steps, loss, **kwargs):
    '''
    Args:
        args: global arguments
    '''
    message = 'Epoch: [{}/{}] '.format(epoch, epochs)
    message = message + 'Step: [{}/{}] '.format(step, steps)
    message = message + 'Loss: {:0.2f} '.format(loss)    
    
    for k, v in kwargs.items():
        message = message + "{}: {}".format(k, v)

    print(message)

    return message

def _pixelwise_accuracy(output, target, threshold=0.5):
    '''
    Args: 
        output: predicted mask
        target: true mask
        threshold: threshold to turn softmax into one-hot encode
    '''
    predicted_mask = (output > threshold).float()
    correct = (predicted_mask == target).float()

    acc = torch.sum(correct) / torch.sum(torch.ones(target.shape))
    acc = acc.cpu().item()
    return acc
        
class Logger(object):
    '''log the training process'''
    def __init__(self, log_dir, resume):
        '''
        Args:
            log_dir: directory to save the log file
        '''
        # read the log files if not the starting from 0 th 
        self.log_dir = log_dir
        
        if resume is not None:
            log_path = os.path.join(log_dir, 'log.pickle')
            with open(log_path, 'rb') as f:
                self.log = pickle.load(f)
        else:
            self.log = {}
    
    def new_epoch(self,epoch):
        '''make log for new epoch'''
        self.log[epoch] = {}
        return

    def __call__(self, epoch,  **kwargs):

        # might need a try except bk if kwargs is empty
        for k,v in kwargs.items():
            self.log[epoch][k] = v
        return

    def is_best(self, epoch):
        '''check if the model (validation) is the best from all epochs
        '''
        if epoch == 1:
            self.log['min_loss'] = self.avg_loss
            self.log['best_model'] = 0

        avg_loss = self.avg_loss
        min_loss = self.log['min_loss']

        if avg_loss <= min_loss:
            min_loss = avg_loss
            self.log['min_loss'] = avg_loss
            self.log['best_model'] = epoch
        return 


    def save_log(self):
        log_path = os.path.join(
                self.log_dir, 'log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.log, f)
        return


class Trainer(object):
    def __init__(self, train_dataset, val_dataset, model, 
            criterion, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                else "cpu")
        print("Device: ", self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers)
        
        self.val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers)

        self.args = args
        self.total_steps = self.total_steps()

        self.model = model.train().to(self.device)

        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), 
                lr=self.args.lr, weight_decay=1e-5)

        self.logger = Logger(log_dir=self.args.log_dir, 
                resume=self.args.resume)
        return

    def total_steps(self):
        '''Compute total steps per epoch'''
        x = len(self.train_dataset) / float(self.args.batch_size)
        if x % 1 == 0:
            total_steps = int(x)
        else:
            total_steps = int(x) + 1
        return total_steps

    def __call__(self, epoch):
        """Train one epoch"""
        self.model = self.model.train()
        self.logger.new_epoch(epoch)

        train_loss = 0
        train_acc = 0
        for step, (feature, elv, mask) in enumerate(self.train_loader):
            step = step + 1
            feature = feature.to(self.device)
            elv = elv.to(self.device)
            mask = mask.to(self.device)

            try:
                self.optimizer.zero_grad()
                output = self.model(feature)
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()            
                loss = loss.detach().cpu().item()
                acc = _pixelwise_accuracy(output, mask, 
                        threshold=self.args.threshold)
                
                train_loss = train_loss + loss
                train_acc = train_acc + acc

                if step % self.args.print_freq == 0:
                    epochs = self.args.start_epoch + self.args.epochs-1

                    _print(epoch=epoch, epochs=epochs, 
                            step=step, steps=self.total_steps,
                            loss=loss, acc=acc)


            # if KeyboardInterrupt happens during training
            # remove logs from current epoch and save the
            # logs from other epochs
            except KeyboardInterrupt:
                print("Saving logs from previous epochs before shutting down the process")
                self.logger.log[epoch] = None
                self.logger.save_log()
        
        train_loss = train_loss / self.total_steps
        train_acc = train_acc / self.total_steps
        self.logger(epoch=epoch, train_loss=train_loss, train_acc=train_acc)
        self.save_ckp(epoch)
        return
    
    def validate(self, epoch):
        print('validating model after epoch: {}'.format(epoch))
        self.model = self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0

            for step, (feature, elv, mask) in enumerate(self.val_loader):
                step = step + 1
                feature = feature.to(self.device)
                mask = mask.to(self.device)

                output = self.model(feature)
                loss = self.criterion(output, mask)
                
                acc = _pixelwise_accuracy(output, mask, 
                    threshold=self.args.threshold)

                val_loss = val_loss + loss.cpu().item()
                val_acc = val_acc + acc
            
            val_loss = val_loss / step
            val_acc = val_acc / step

            self.logger(epoch=epoch, val_loss=val_loss, val_acc=val_acc)
            print("Val Loss: {:0.2f}, Val Acc: {:0.2f}".format(val_loss, val_acc))

        return

    def save_ckp(self, epoch):
        ckp_file = 'model_{}.pth'.format(epoch)
        try:
            state_dict = self.model.state_dict()
            ckp_path = os.path.join(self.args.ckp_dir, 
                    ckp_file)

            _save_checkpoint(state_dict, ckp_path)
        # KeyboardInterrupt occurs while saving
        except KeyboardInterrupt:
            self.save_ckp()
        return







