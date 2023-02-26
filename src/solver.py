"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

In this script the class "Solver" is created, which trains the chosen model.
"""

# import necessary modules
import os
from torch import optim
from evaluation import *
from networks import *
from loss import *
import csv
import torch

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Initialize data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.image_size = config.image_size
        # Initialize the model
        self.net = None
        # Initialize the optimizer
        self.optimizer = None
        # Initialize the number of input and output channels
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # initialize the loss function
        if config.lossfunc == 'DiceLoss':
            self.loss_name = 'DiceLoss'
            self.criterion = DiceLoss()
        # initialize the augmentation probability
        self.augmentation_prob = config.augmentation_prob
        # initialize hyper-parameters of the optimization scheme
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        # initialize training settings
        self.batch_size = config.batch_size

        # initialize path of the model to be saved
        self.model_path = config.model_path
        self.result_path = config.result_path
        # using cpu or gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.pretrained = bool(config.pretrained)
        self.number_layers_freeze = config.number_layers_freeze
        # initialize the path to save the network
        self.net_path = None
        
        self.build_model()


    def build_model(self):
        self.net = TernausNet16(pretrained=self.pretrained)
        self.optimizer = optim.Adam(list(self.net.parameters()),
                                    self.lr, [self.beta1, self.beta2], weight_decay=self.weight_decay)
        self.net.to(self.device)

        if self.number_layers_freeze != 0:
            layer_names = ['conv'+str(i) for i in range(1,self.number_layers_freeze+1)]
            for layer_name in layer_names:
                for param in getattr(self.net,layer_name).parameters():
                    param.requires_grad = False

    def print_network(self):
        """Print out the network information."""
        print(self.net)
        print(self.model_type)

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.net.zero_grad()

    def train(self):
        """Function to train the network"""
        # path to save the best model
        self.net_path = os.path.join(self.model_path, '%s-%s-%d-%d-%d-%d-%d-%d-%.6f-%.2f-%.4f-%.10f-%.2f-%d.pkl' % (
            self.model_type, self.loss_name, self.image_size, self.img_ch, self.output_ch,
            self.pretrained, self.num_epochs, self.batch_size, self.lr, self.beta1, self.beta2,
            self.weight_decay, self.augmentation_prob, self.number_layers_freeze))
        # print the model information
        self.print_network()
        # ====================================== Training =============================================================#
        lowest_valid_loss = 1e5  # initialize the lowest validation loss.
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)  # decay learning rate every 100
        # epochs by a factor of 0.5
        print('Start learning ...')
        # main loop through the training data
        for epoch in range(self.num_epochs):
            current_lr = [group['lr'] for group in self.optimizer.param_groups]
            print('Epoch [%d/%d], lr:%f' % (epoch, self.num_epochs-1, current_lr[0]))
            self.net.train(True)

            epoch_train_loss = 0  # initialize the epoch training loss
            acc = 0.  # initialize accuracy
            SE = 0.  # initialize sensitivity (recall)
            SP = 0.  # initialize specificity
            PC = 0.  # initialize precision
            JS = 0.  # initialize Jaccard Similarity
            DC = 0.  # initialize dice Coefficient
            length = 0  # counter
            for i, (images, GT) in enumerate(self.train_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)  # GT : Ground Truth
                GT_flat = GT.view(GT.size(0), -1)
            
                SR = self.net(images)  # SR : Segmentation Result
                SR_probs = torch.sigmoid(SR)  # segmentation results as probabilities
                SR_flat = SR_probs.view(SR_probs.size(0), -1)  # flatten the results
                loss = self.criterion(SR_flat, GT_flat)  # compute loss
                epoch_train_loss += loss.item()  # accumulate the loss in the epoch
                # compute metrics for each batch and accumulate them
                acc += get_accuracy(SR_probs, GT)
                SE += get_sensitivity(SR_probs, GT)
                SP += get_specificity(SR_probs, GT)
                PC += get_precision(SR_probs, GT)
                JS += get_JS(SR_probs, GT)
                DC += get_DC(SR_probs, GT)
                length += images.shape[0]

                # backprop + Update the parameters
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

            # compute average metrics for training data
            train_acc = acc / length
            train_SE = SE / length
            train_SP = SP / length
            train_PC = PC / length
            train_JS = JS / length
            train_DC = DC / length
            # compute the average training loss
            train_loss = epoch_train_loss / length
            # Print the log info
            print('[Training] Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, JS: %.4f, DC: %.4f' % (
                train_loss, train_acc, train_SE, train_SP, train_PC, train_JS, train_DC))

            # ===================================== Validation ====================================================#
            self.net.train(False)  # set the training to false
            self.net.eval()  # set the module in evaluation mode.
            
            epoch_valid_loss = 0  # initialize the epoch validation loss
            acc = 0.  # initialize accuracy
            SE = 0.  # initialize sensitivity (Recall)
            SP = 0.  # initialize specificity
            PC = 0.  # initialize precision
            JS = 0.  # initialize Jaccard Similarity
            DC = 0.  # initialize dice Coefficient
            length = 0  # initialize counter
            with torch.no_grad():
                for i, (images, GT) in enumerate(self.valid_loader):
                    images = images.to(self.device)
                    GT = GT.to(self.device)  # GT : Ground Truth
                    GT_flat = GT.view(GT.size(0), -1)
                    SR = self.net(images)  # SR : Segmentation Result
                    SR_probs = torch.sigmoid(SR)  # segmentation results as probabilities
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)  # flatten the results
                    loss = self.criterion(SR_flat, GT_flat) # compute loss
                    epoch_valid_loss += loss.item()  # accumulate the loss in the epoch

                    # compute metrics for each batch and accumulate them
                    acc += get_accuracy(SR_probs, GT)
                    SE += get_sensitivity(SR_probs, GT)
                    SP += get_specificity(SR_probs, GT)
                    PC += get_precision(SR_probs, GT)
                    JS += get_JS(SR_probs, GT)
                    DC += get_DC(SR_probs, GT)
                    length += images.shape[0]

            # compute average metrics for validation data
            valid_acc = acc / length
            valid_SE = SE / length
            valid_SP = SP / length
            valid_PC = PC / length
            valid_JS = JS / length
            valid_DC = DC / length
            valid_loss = epoch_valid_loss / length
            scheduler.step()

            print(
                '[Validation] Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, JS: %.4f, DC: %.4f' % (
                    valid_loss, valid_acc, valid_SE, valid_SP, valid_PC, valid_JS, valid_DC))
            # Save the best model
            if valid_loss < lowest_valid_loss:
                lowest_valid_loss = valid_loss
                best_epoch = epoch
                best_net = self.net.state_dict()
                print('Best %s model loss : %.4f' % (self.model_type, lowest_valid_loss))
                torch.save(best_net, self.net_path)
            print('----------------------------------------------------------------------------------------------')
            if epoch == 0:
                f = open(os.path.join(self.result_path, '%s-%s-%d-%d-%d-%d-%d-%d-%.6f-%.2f-%.4f-%.10f-%.2f-%d.csv' % (
                        self.model_type, self.loss_name, self.image_size, self.img_ch, self.output_ch,
                        self.pretrained, self.num_epochs, self.batch_size, self.lr, self.beta1, self.beta2,
                        self.weight_decay, self.augmentation_prob, self.number_layers_freeze)), 'w', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(['Architecture', 'epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'train_SE',
                             'valid_SE', 'train_SP', 'valid_SP', 'train_PC', 'valid_PC',
                             'train_JS', 'valid_JS', 'train_DC', 'valid_DC', 'lr',
                             'best_epoch', 'num_epochs','batch_size' ,'augmentation_prob','pretrained', 'weight_decay',
                             'number_layers_freeze'])
                f.close()
            f = open(os.path.join(self.result_path, '%s-%s-%d-%d-%d-%d-%d-%d-%.6f-%.2f-%.4f-%.10f-%.2f-%d.csv' % (
                    self.model_type, self.loss_name, self.image_size, self.img_ch, self.output_ch,
                    self.pretrained, self.num_epochs, self.batch_size, self.lr, self.beta1, self.beta2,
                    self.weight_decay, self.augmentation_prob, self.number_layers_freeze)), 'a', encoding='utf-8', newline='')
                
            wr = csv.writer(f)
            wr.writerow(
                [self.model_type, epoch, train_loss, valid_loss, train_acc, valid_acc, train_SE, valid_SE, train_SP,
                 valid_SP, train_PC, valid_PC, train_JS, valid_JS, train_DC, valid_DC,
                 current_lr[0], best_epoch, self.num_epochs, self.batch_size, self.augmentation_prob,
                 self.pretrained, self.weight_decay, self.number_layers_freeze])
            f.close()

    def pred(self):
        """Function to obtain the results on test data"""
        # load the trained model
        model = self.net
        model.load_state_dict(torch.load(self.net_path))
        model.train(False)
        model.eval()

        epoch_test_loss = 0.  # initialize the epoch validation loss
        acc = 0.  # initialize accuracy
        SE = 0.  # initialize sensitivity (Recall)
        SP = 0.  # initialize specificity
        PC = 0.  # initialize precision
        JS = 0.  # initialize Jaccard Similarity
        DC = 0.  # initialize dice Coefficient
        length = 0  # initialize counter
        with torch.no_grad():
            for i, (images, GT) in enumerate(self.test_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)  # GT : Ground Truth
                GT_flat = GT.view(GT.size(0), -1)

                SR = self.net(images)  # SR : Segmentation Result
                SR_probs = torch.sigmoid(SR)  # segmentation results as probabilities
                SR_flat = SR_probs.view(SR_probs.size(0), -1)  # flatten the results
                loss = self.criterion(SR_flat, GT_flat)  # compute loss
                epoch_test_loss += loss.item()  # accumulate the loss in the epoch
                    

                # compute metrics for each batch and accumulate them
                acc += get_accuracy(SR_probs, GT)
                SE += get_sensitivity(SR_probs, GT)
                SP += get_specificity(SR_probs, GT)
                PC += get_precision(SR_probs, GT)
                JS += get_JS(SR_probs, GT)
                DC += get_DC(SR_probs, GT)
                length += images.shape[0]

        # compute average metrics for test data
        test_acc = acc / length
        test_SE = SE / length
        test_SP = SP / length
        test_PC = PC / length
        test_JS = JS / length
        test_DC = DC / length
        test_loss = epoch_test_loss / length
        print(
            '[Test] Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, JS: %.4f, DC: %.4f' % (
                test_loss, test_acc, test_SE, test_SP, test_PC, test_JS, test_DC))
        print('----------------------------------------------------------------------------------------------')
        
        f = open(os.path.join(self.result_path, '%s-%s-%d-%d-%d-%d-%d-%d-%.6f-%.2f-%.4f-%.10f-%.2f-%d.csv' % (
            self.model_type, self.loss_name, self.image_size, self.img_ch, self.output_ch,
            self.pretrained, self.num_epochs, self.batch_size, self.lr, self.beta1, self.beta2,
            self.weight_decay, self.augmentation_prob, self.number_layers_freeze)), 'a', encoding='utf-8', newline='')

        wr = csv.writer(f)
        wr.writerow(['Architecture', 'test_loss', 'test_acc',
                     'test_SE', 'test_SP', 'test_PC',
                     'test_JS', 'test_DC'])
        wr.writerow(
            [self.model_type, test_loss, test_acc, test_SE,
             test_SP, test_PC, test_JS, test_DC])
        f.close()
