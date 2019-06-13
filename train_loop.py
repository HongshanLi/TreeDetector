# define Trainer here, 
# trainer should contain logics that trains the model for one epoch
# Define some helper functions like: logging loss/error printing 
# loss and accuracy
# for classification related tasks

from torch.utils.data import DataLoader
import torch.optim as optim

def _accuracy():
    return

def _top_one_accuracy():
    return

def _top_five_accuracy():
    return


# for regression related tasks



# checkpoints
def _save_checkpoints(state, is_best, ckp_dir, ckpfile):
    ckp_path = os.path.join(ckp_dir, ckpfile)
    torch.save(state, ckp_path)
    best_model_path = os.path.join(ckp_dir, 'model_best.pth')
    if is_best:
        shutil.copyfile(ckp_path, best_model_path)
    return

# print out progress
def _print(epoch, step, steps, args, **kwargs):
    '''
    Args:
        args: global arguments
    '''
    epochs = args.epochs
    message = 'Epoch: [{}/{}]'.format(epoch, epochs)
    message = message + 'Step: [{}/{}]'.format(step, steps)
        
    # TODO add kwarg print        

    return message


class Logger(object):
    '''log the training process'''
    def __init__(self, model, args, train=True):
        '''
        Args:
            args: global arguments
            train:(boolean) log for training?
        '''
        self.model = model
        self.args = args
        self.log = {}
    
    def new_epoch(self,epoch):
        '''make log for new epoch'''
        self.log[epoch] = []
        return

    def __call__(self, epoch, **kwargs):
        step_log = {}
        for k,v in kwargs.items():
            step_log[k] = v
        self.log[epoch].append(step_log)
        return


    def is_best(self, epoch):
        '''check if the model (validation) is the best from all epochs'''
        epoch_pfm = self.log[epoch]
        # compute average loss
        loss = 0
        for i, item in enumerate(epoch_pfm):
            loss += item['loss']

        avg_loss=float(loss) / float(i+1)

        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self.log[epoch].append(
                    {'is_best': True}
                    )
        else:
            self.log[epoch].append(
                    {'is_best': False}
                    )
        return 


    def save_logs(self):
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
                pickle.dump(self.step_log, f)
        # KeyboardInterrupt occurs while saving
        except KeyboardInterrupt:
            self.save_logs()
        return


    def save_ckp(self, epoch):
        ckpfile = 'model_{}.pth'.format(epoch)
        is_best = self.log[epoch][-1]['is_best']
        try:
            _save_checkpoint(state, is_best, self.args.ckp_dir, ckpfile)
        # KeyboardInterrupt occurs while saving
        except KeyboardInterrupt:
            self.save_ckp()
        return




class Validator(object):
    """Wrapper for validation
    includes save ckps find best model and checkpoint
    """
    def __init__(self, val_dataset, model, criterion, args):
        self.val_dataset = val_dataset
        self.val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.workers)

        self.model = model.eval()
        self.criterion = criterion
        self.args = args
        self.logger = Logger(train=False, model=self.model, 
                args=self.args)
        
        return

    def __call__(self, epoch):
        self.logger.new_epoch(epoch)
        for step, (feature, _, mask) in enumerate(self.val_loader):
            try:
                output = self.model(features)
                loss = self.criterion(output, mask)
                self.logger(step=step, loss=loss, lr=self.args.lr)
            except KeyboardInterrupt:
                # remove logs from the current epoch and save
                # the logs from the rest
                self.logger.log[epoch] = None 
                self.logger.save_log()

        self.logger.save_log()
        self.logger.save_ckp()

        return
    

class Trainer(object):
    def __init__(self, train_dataset, model, 
            criterion, args):
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers)

        self.args = args
        self.total_steps = self.total_steps()

        self.model = model.train()
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters())
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
            print(feature.shape, elv.shape, mask.shape)

            try:
                self.optimizer.zero_grad()
                output = self.model(feature)
                print(mask.shape, output.shape)
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()            
                loss = loss.detach().cpu().item()
                self.logger(epoch=epoch, step=step, loss=loss, lr=self.args.lr)
                if step % self.args.print_freq==0:
                    _print(epoch=epoch, step=step, steps=self.total_steps,
                            loss=loss, args=self.args, lr=self.args.lr)


            # if KeyboardInterrupt happens during training
            # remove logs from current epoch and save the
            # logs from other epochs
            except KeyboardInterrupt:
                print("Saving logs from previous epochs before shutting down the process")
                self.logger.log[epoch] = None
                self.logger.save_logs()

        self.logger.save_logs()
        self.logger.save_ckp()
        return





