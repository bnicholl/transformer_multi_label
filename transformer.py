import torch
from sklearn.utils import shuffle
from transformers import AutoTokenizer
from transformers import ElectraTokenizer, ElectraModel, ElectraForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.utils import shuffle
from torch import optim
import copy


def elec(input_ids, labels, mask, epochs = 20):
    # this pre logits mask will get multiplied by the final hidden state in our model in order to zero our the
    # mask vectors
    pre_logits_mask = torch.reshape(mask, (mask.shape[0], mask.shape[1], 1) )
    pre_logits_validation_mask = torch.reshape(validation_mask, (validation_mask.shape[0], validation_mask.shape[1], 1) )
    # creater a logit layer that has 5 nodes, which will ultimately give us 5 probabilites for our multi layer neural network
    logit_layer = torch.nn.Linear(768, 5)
    ###NEW
    loss_list = []
    ###NEW
    lowest_loss = 1000000
    
    model = ElectraModel.from_pretrained('google/electra-base-discriminator', num_labels = 4,
                                                output_hidden_states=True,
                                                output_attentions=True)
    
    
    model.train()
    
    #
    model_params = [i for i in model.parameters()]
    logit_params = [i for i in logit_layer.parameters()]
    model_params.append(logit_params[0])
    model_params.append(logit_params[1])
    model_params = iter(model_params)
    #
    # below hashtagged code is for running SGD instead of adam optimizer
    #optimizer = optim.SGD(model_params, lr=0.001, momentum=0.04)
    optimizer = AdamW(model_params,
                      lr = 4e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = epochs)
    
    
   
    for batch in range(epochs):
        input_ids, labels, mask = shuffle(input_ids, labels, mask)
        model.train()
        optimizer.zero_grad()
        
        #zero's our gradient so it does not build up over each iteration
        #model.zero_grad()
        
        #forward pass
        outputs = model(input_ids, 
                            token_type_ids=None, 
                            attention_mask=mask,
                            )
        
        
        # example size is
        #torch.Size([335, 37, 768]) for below, where
        #335 relates to amount of training examples. 37 realates to max words in sequence. 
        #768 are the indiviudal word vectors
        last_hidden_state = outputs.last_hidden_state
        
        # this zeros out the mask vectors in the final hidden state, becuase for the vanilla models, it doesn't do that automatically
        # for example, say our input_ids = [[101,4769, 77, 102, 0, 0]], where the 0's represent masks. Since we running the vanillia model,
        # and the vanilla model ends on a hidden layer, as opposed to a attention mechanism, it never 0's out the last 2 word vecotors.
        # this is shown below
        #outputs.last_hidden_state
        #Out[120]: 
        #tensor([[[ 1.0132, -0.4270, -0.2964,  ..., -0.4578,  0.6050,  0.1191],
        # [ 0.7935, -0.7512, -0.2856,  ...,  0.1178,  0.0712, -0.4344],
        # [ 0.8716, -0.7868,  0.1454,  ...,  0.2290,  0.3592, -0.1717],
        # [ 1.0132, -0.4270, -0.2964,  ..., -0.4578,  0.6050,  0.1191],
        # [ 0.9389, -0.8407, -0.1254,  ...,  0.5575,  0.9512, -0.7111],       These two word vectors 
        # [ 0.9249, -0.8279, -0.1271,  ...,  0.5662,  0.9262, -0.6858]]],     need to be 0 everywhere
        last_hidden_state_zero_layer = torch.mul(last_hidden_state, pre_logits_mask)
        
        #torch.Size([335, 768]) for below is:
        summed_final_hidden_state = torch.sum(last_hidden_state_zero_layer, 1)
        
        logits = logit_layer(summed_final_hidden_state)
        
        #This loss combines a Sigmoid layer and the BCELoss in one single class
        logits_and_loss = torch.nn.BCEWithLogitsLoss()
        
        loss = logits_and_loss(logits, labels.type_as(logits))
        #########
        """compute gradient. We should now have grad.data in model.parameters()"""
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        
        ##now we run our validation data. 
        model.eval()

        validation_output = model(validation_inputs,
                                  token_type_ids=None, 
                                  attention_mask=validation_mask
                                #labels = validation_labels
                                  )
        
        val_last_hidden_state = validation_output.last_hidden_state
        val_last_hidden_state_zero_layer = torch.mul(val_last_hidden_state, pre_logits_validation_mask)
        
        val_summed_final_hidden_state = torch.sum(val_last_hidden_state_zero_layer, 1)
        
        validation_logits = logit_layer(val_summed_final_hidden_state)
            
        val_logits_and_loss = torch.nn.BCEWithLogitsLoss()
        
        validation_loss = val_logits_and_loss(validation_logits, validation_labels.type_as(logits))
        #########
        
        if validation_loss < lowest_loss:            
            lowest_loss = validation_loss
            lowest_loss_model = copy.deepcopy(model)
            lowest_loss_logit_layer = copy.deepcopy(logit_layer)
        loss_list.append(validation_loss)
    
    return lowest_loss_model, lowest_loss_logit_layer, lowest_loss, loss_list#, model

# below hashtagged code is an example for calling the transfromer function
#model, logit_layer, lowest_loss, loss_list = elec(train_inputs,  train_labels, mask)
