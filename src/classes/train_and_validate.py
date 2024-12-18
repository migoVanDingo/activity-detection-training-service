from datetime import datetime
import traceback
from sklearn.metrics import accuracy_score
from barbar import Bar
import time, json, math, os, torch




from utility.logger_config import setup_logger

system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

class TrainAndValidate:
    def __init__(self,
                 params,
                 net,
                 optimizer,
                 criterion,
                 train_loader,
                 val_loader,
                 cuda_device):
        
        self.params = params
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cuda_device = cuda_device

    def do_process(self):
        try:
            print(f"{self.__class__.__name__} -- Training started")
            # Start timer
            start_time = datetime.now()
            system_logger.info(f"{self.__class__.__name__} -- Training started at {start_time}")

            train_list = []
            val_list = []
            
            train_list.append({"mode": "info", "time":f"{start_time}", "action": "start"})
            val_list.append({"mode": "info", "time":f"{start_time}", "action": "start"})
            
            # Load model to GPU
            self.net.to(self.cuda_device)

            # Start training loop
            train_log, val_log = self.epoch_cycle(self.params['training']['max_epochs'])
            train_list.extend(train_log)
            val_list.extend(val_log)

            # End timer
            end_time = datetime.now()
            train_list.append({"mode": "info", "time":f"{end_time}", "action": "end"})
            val_list.append({"mode": "info", "time":f"{end_time}", "action": "end"})

            # Write to training logs
            train_log, val_log = self.init_logs()
            self.write_logs(train_log, train_list)
            self.write_logs(val_log, val_list)

            train_log.close()
            val_log.close()


    
            return None
        except Exception as e:
            print(f"{self.__class__.__name__} -- {traceback.format_exc()} -- ERROR: {e}")
            system_logger.error(f"{self.__class__.__name__} -- {traceback.format_exc()} - {e} Error: {e}")
            return e



    def epoch_cycle(self, max_epochs):
        best_validation_accuracy = 0
        best_validation_loss = math.inf
        stop_limit = 0
        train_log_list = []
        val_log_list = []
        for epoch in range(max_epochs):

            self.net.train()
            training_loss, training_accuracy = self.cycle_metrics()

            if epoch % self.params['training']['epoch_save_interval'] == 0:
                self.save_checkpoint(epoch, training_loss, os.path.join(self.params['prepare_environment']['checkpoint_dir'], f"checkpoint_epoch_{epoch}.pth"))

            self.net.eval()
            validation_loss, validation_accuracy = self.cycle_val_metrics()

            if epoch > self.params['training']['epoch_skip'] and validation_loss >= best_validation_loss:
                stop_limit += 1
            else:
                checkpoint = os.path.join(self.params['prepare_environment']['checkpoint_dir'], f"best_epoch_{epoch}.pth")
                self.save_checkpoint(epoch, validation_loss, checkpoint)
                best_validation_loss = validation_loss
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch
                num_epochs_since_best = 0
                stop_limit = 0

            train_metrics = {
                "mode": "train",
                "epoch": epoch,
                "training_loss": training_loss,
                "training_accuracy": training_accuracy
            }
            train_log_list.append(train_metrics)
            

            val_metrics = {
                "mode": "validation",
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy
            }
            val_log_list.append(val_metrics)

            

            system_logger.info(f"{self.__class__.__name__} -- Epoch: {epoch}, Training loss: {training_loss}, Training accuracy: {training_accuracy}, Validation loss: {validation_loss}, Validation accuracy: {validation_accuracy}, Best validation loss: {best_validation_loss}, Best validation accuracy: {best_validation_accuracy}, Best epoch: {best_epoch}")
            print(f"{self.__class__.__name__} -- Epoch: {epoch}, Training loss: {training_loss}, Training accuracy: {training_accuracy}, Validation loss: {validation_loss}, Validation accuracy: {validation_accuracy}, Best validation loss: {best_validation_loss}, Best validation accuracy: {best_validation_accuracy}, Best epoch: {best_epoch} ---- Stop limit: {stop_limit}")

            if stop_limit >= self.params['training']['early_stopping']:
                    break
        
        return train_log_list, val_log_list


    ### Training
    def cycle_metrics(self):
        gt_list = []
        time_taken_list = []
        loss_list = []
        prediction_list = []
        for data in self.train_loader:
            loss, prediction, labels, time_taken = self.train(data)
            
            time_taken_list.append(time_taken)
            gt_list += labels
            loss_list.append(loss)
            prediction_list += prediction

        gt_list = [round(x) for x in gt_list]
        prediction_list = [round(x) for x in prediction_list]
        total_time = round(sum(time_taken_list), 5)
        accuracy = accuracy_score(gt_list, prediction_list)
        average_loss = round(sum(loss_list) / len(loss_list), 2)

        system_logger.info(f"{self.__class__.__name__} -- Training metrics -- accuracy:{accuracy}, average_loss:{average_loss}, total_time:{total_time}")
        return average_loss, accuracy
        



    def train(self, data):
        # Label and input tensors
        inputs, labels = (data[0].to(self.cuda_device, non_blocking=True),
                          data[1].to(self.cuda_device, non_blocking=True))
        labels = torch.reshape(labels.float(), (-1, 1))

        start_time = time.time()
        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        end_time = time.time()
        total_time = round(end_time - start_time, 5)


        prediction = outputs.data.clone().to('cpu').numpy().flatten().tolist()
        loss = loss.data.clone().to('cpu').numpy().tolist()
        labels = labels.data.clone().to('cpu').numpy().flatten().tolist()

        return loss, prediction, labels, total_time



    ### Validation
    def cycle_val_metrics(self):
        with torch.no_grad():
            loss_list = []
            prediction_list = []
            gt_list = []
            time_taken_list = []
            for data in Bar(self.val_loader):
                loss, prediction, labels, time_taken = self.validate(data)
                loss_list.append(loss)
                prediction_list += prediction
                gt_list += labels
                time_taken_list.append(time_taken)
            
            gt_list = [round(x) for x in gt_list]
            prediction_list = [round(x) for x in prediction_list]
            total_time = round(sum(time_taken_list), 5)
            accuracy = accuracy_score(gt_list, prediction_list)
            average_loss = round(sum(loss_list) / len(loss_list), 2)

            system_logger.info(f"{self.__class__.__name__} -- Validation metrics -- accuracy:{accuracy}, average_loss:{average_loss}, total_time:{total_time}")
            return average_loss, accuracy


    def validate(self, data):
        inputs, labels = (data[0].to(self.cuda_device, non_blocking=True),
                        data[1].to(self.cuda_device, non_blocking=True))
        labels = torch.reshape(labels.float(), (-1, 1))  # Ensure it's the correct shape
        
        start_time = time.time()
        outputs = self.net(inputs)
        end_time = time.time()
        total_time = round(end_time - start_time, 5)
        
        # Convert predictions to list for further processing
        prediction = outputs.data.clone().to('cpu').numpy().flatten().tolist()
        
        # Compute the loss
        loss = self.criterion(outputs, labels)  # Keep labels as a tensor here
        loss = loss.item()  # Convert to a scalar if needed for logging
        
        # If you still need labels as a list, convert it AFTER computing the loss
        labels_list = labels.data.clone().to('cpu').numpy().flatten().tolist()
        
        return loss, prediction, labels_list, total_time

        


    ### Helper functions
    def init_logs(self):
        system_logger.info(f"{self.__class__.__name__} -- Initializing logs")
        
        files = self.params['prepare_environment']['logs']
        file_objects = []
        for file in files:
            file_path = os.path.join(self.params['prepare_environment']['working_dir'], file)
            file_objects.append(open(file_path, "w"))

        return file_objects
    
    def write_logs(self, log_file, log_data):
        time = datetime.now()
        log = {
            "time": f"{time}",
            "data": log_data
        }
        system_logger.info(f"{self.__class__.__name__} -- FILE: {log_file} ----- LOGGING: {log}")
        
        log_file.write(json.dumps(log))


    def save_checkpoint(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss
        }, path)


        