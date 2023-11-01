import torchvision.models as models
import torch
import os

class BaseModel():
    def name(self):
        return 'base_model'

    def __init__(self, config):
        self.config = config
        self.use_weighted_loss = config["use_weighted_loss"]
        self.continue_train = config["continue_train"]

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_errors(self):
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass

class DeepModel(BaseModel):
    def name(self):
        n = [self.deep_model]
        if self.use_pretrained:
            n += ['pretrained_weights']
        if self.use_weighted_loss:
            n += ['weighted_loss']
        if self.use_antialias:
            n += ['antialias']

        return '_'.join(n)

    def __init__(self, config, is_eval=False, class_weight=None):
        super().__init__(config)
        self.is_eval = is_eval
        self.deep_model = self.config["deep_model"]
        self.class_weight = class_weight if self.use_weighted_loss else None

        model = getattr(models, self.deep_model)
        model = model(**self.config["parameters"])

        # Modified the original HACK! part into an if condition to avoid the need to comment/uncomment lines
        # when using vgg vs resnet/inception
        if 'vgg' in self.deep_model:
            model.classifier._modules['6'] = torch.nn.Linear(4096, self.config["num_subtypes"])
        else:
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, self.config["num_subtypes"])

        self.model = model.cuda()

        if not self.is_eval:
            if self.use_weighted_loss:
                self.criterion = torch.nn.CrossEntropyLoss(
                    reduction='mean', weight=torch.from_numpy(self.class_weight).cuda())
                #self.criterion = torch.nn.BCEWithLogitsLoss(
                #    reduction='mean', weight=torch.from_numpy(self.class_weight).cuda())
            else:
                self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
                #self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        optimizer = getattr(torch.optim, self.config["optimizer"]["type"])
        self.optimizer = optimizer(model.parameters(), **self.config["optimizer"]["parameters"])

        if self.continue_train:
            self.load_state(config["load_deep_model_id"])

        if self.is_eval:
            self.load_state(config["load_deep_model_id"])
            self.model = self.model.eval()

    def forward(self, input_data):
        output = self.model.forward(input_data)

        if type(output).__name__ in ['GoogLeNetOutputs', 'InceptionOutputs'] and config["parameters"]["aux_logits"]:
            logits = output.logits
        else:
            logits = output
            
        probs = torch.softmax(logits, dim=1)
        return logits, probs, output

    def get_loss(self, logits, labels, output=None):
        if type(output).__name__ == 'GoogLeNetOutputs':
            loss = self.criterion(logits.type(torch.float), labels.type(torch.long)) + 0.4 * (self.criterion(output.aux_logits1.type(
                torch.float), labels.type(torch.long)) + self.criterion(output.aux_logits2.type(torch.float), labels.type(torch.long)))
        elif type(output).__name__ == 'InceptionOutputs':
            loss = self.criterion(logits.type(torch.float), labels.type(
                torch.long)) + 0.4 * self.criterion(output.aux_logits.type(torch.float), labels.type(torch.long))
        else:
            loss = self.criterion(logits.type(
                torch.float), labels.type(torch.long))

        return loss

    def optimize_parameters(self, logits, labels, output=None):
        self.loss = self.get_loss(logits, labels, output)
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.loss.item()

    def get_current_errors(self):
        return self.loss.item()

    def load_state(self, save_location, test_name, itr):
        filename = '{}_{}.pth'.format(str(test_name), str(itr))
        save_path = os.path.join(save_location, filename)

        if not torch.cuda.is_available():
            state = torch.load(save_path, map_location='cpu')
        else:
            state = torch.load(save_path)

        try:
            self.model.load_state_dict(state['state_dict'])
        except RuntimeError:
            pretrained_dict = state['state_dict']
            model_dict = self.model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(pretrained_dict)

        self.optimizer.load_state_dict(state['optimizer'])

        model_id = state['iter_idx']
        return model_id

    def save_state(self, save_location, train_instance_name, iter_idx, epoch):
        filename = f'{train_instance_name}.pth'
        save_path = os.path.join(save_location, filename)
        os.makedirs(save_location, exist_ok=True)
        state = {
            'epoch': epoch,
            'iter_idx': iter_idx,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, save_path)
