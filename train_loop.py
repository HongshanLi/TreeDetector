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
    def __init__(self, model, args, train=True):
        '''
        Args:
            args: global arguments
            train:(boolean) log for training?
        '''
        self.train = train
        self.model = model
        self.args = args
        # read the log files if not the starting from 0 th 
        self.log = {}
    
    def new_epoch(self,epoch):
        '''make log for new epoch'''
        self.log[epoch] = []
        return


    def __call__(self, epoch, step, loss, lr, **kwargs):
        step_log = {}

        step_log['step'] = step
        step_log['loss'] = loss
        step_log['lr'] = lr
        
        # might need a try except bk if kwargs is empty
        for k,v in kwargs.items():
            step_log[k] = v

        self.log[epoch].append(step_log)
        return

    def _epoch_avg_loss(self, epoch):
        '''compute the average loss of current epoch'''
        loss = 0 
        for step in self.log[epoch]:
            loss = loss + step['loss']

        loss = loss / len(self.log[epoch])
        return loss

    def compute_epoch_avg_loss(self, epoch):
        loss = 0
        for step in self.log[epoch]:
            loss = loss + step['loss']
        loss = loss / len(self.log[epoch])

        self.avg_loss = loss
        return loss


    def compute_epoch_avg_acc(self, epoch):
        '''compute avg acc of the epoch'''
        acc = 0
        for step in self.log[epoch]:
            acc = acc + step['acc']

        acc = acc / len(self.log[epoch])
        return acc

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
        try:
            if self.train:
                logfile=os.path.join(
                        self.args.log_dir, 
                        'train_log.pickle')
            else:
                logfile=os.path.join(
                        self.args.log_dir, 
                        'val_log.pickle')

            with open(logfile, 'wb') as f:
                pickle.dump(self.log, f)
        # KeyboardInterrupt occurs while saving
        except KeyboardInterrupt:
            self.save_log()
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

class Validator(object):
    """Wrapper for validation
    includes save ckps find best model and checkpoint
    """
    def __init__(self, val_dataset, model, criterion, args):
        self.device = torch.device("cuda:0")
        self.val_dataset = val_dataset
        self.val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.workers)

        self.model = model.eval().to(self.device)

        self.criterion = criterion
        self.args = args
        self.logger = Logger(train=False, model=self.model, 
                args=self.args)
        
        return

    def __call__(self, epoch):
        self.logger.new_epoch(epoch)
        for step, (feature, elv, mask) in enumerate(self.val_loader):
            step = step + 1
            feature = feature.to(self.device)
            elv = elv.to(self.device)
            mask = mask.to(self.device)

            try:
                output = self.model(feature)
                loss = self.criterion(output, mask)
                acc = _pixelwise_accuracy(output, mask, 
                        threshold=self.args.threshold)
                self.logger(epoch=epoch, step=step, loss=loss, 
                        lr=self.args.lr, acc=acc)
            except KeyboardInterrupt:
                # remove logs from the current epoch and save
                # the logs from the rest
                self.logger.log[epoch] = None 
                self.logger.save_log()

        self.logger.save_log()
        self.logger.save_ckp(epoch)

        print("Validation result after epoch:{}".format(epoch))

        val_loss = self.logger.compute_epoch_avg_loss(epoch)
        val_acc = self.logger.compute_epoch_avg_acc(epoch)
        
        print("Loss: {:0.2f}, Accuracy : {:0.2f}".format(val_loss, val_acc)) 
        return

    def is_best(self, epoch):
        '''determine if the model after current epoch is the best'''
        self.logger.is_best(epoch)
        return
    

class Trainer(object):
    def __init__(self, train_dataset, model, 
            criterion, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                else "cpu")
        print("Device: ", self.device)

        self.train_dataset = train_dataset
        self.train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers)

        self.args = args
        self.total_steps = self.total_steps()

        self.model = model.train().to(self.device)

        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), 
                lr=self.args.lr, weight_decay=1e-5)
        self.logger = Logger(model=self.model, args=self.args)
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
        self.logger.new_epoch(epoch)
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

                self.logger(epoch=epoch, step=step, loss=loss, 
                        lr=self.args.lr, acc=acc)

                if step % self.args.print_freq == 0:
                    _print(epoch=epoch, epochs=self.args.epochs, 
                            step=step, steps=self.total_steps,
                            loss=loss, acc=acc)


            # if KeyboardInterrupt happens during training
            # remove logs from current epoch and save the
            # logs from other epochs
            except KeyboardInterrupt:
                print("Saving logs from previous epochs before shutting down the process")
                self.logger.log[epoch] = None
                self.logger.save_log()

        self.logger.save_log()
        self.logger.save_ckp(epoch)
        return







