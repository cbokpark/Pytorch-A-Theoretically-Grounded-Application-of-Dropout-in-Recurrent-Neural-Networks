import torch
import torch.nn as nn 
from typing import Optional, Tuple 

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence 

class residual_LSTM(nn.Module):
    def __init__(self,
                input_size: int,
                cell_size: int,
                hidden_size: int,
                num_layer : int = 2,
                bidirectional: bool = True,
                batch_first: bool = True,
                recurrent_dropout_probability: float = 0.0,
                use_input_projection_bias: bool = True) -> None:
    
        super(residual_LSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.num_layer = num_layer
        self.bidirectional = bidirectional

        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.use_highway = False
        self.use_input_projection_bias = use_input_projection_bias
        self.batch_first = batch_first

        self.LSTM = LSTM_block(input_size,cell_size,hidden_size,num_layer,recurrent_dropout_probability,use_input_projection_bias,go_forward =True,skip_connection =True,batch_first =True)
        if self.bidirectional:
            self.bw_LSTM = LSTM_block(input_size,cell_size,hidden_size,num_layer,recurrent_dropout_probability,use_input_projection_bias,go_forward =False,skip_connection =True,batch_first =True) 

    def forward(self,
            inputs:PackedSequence,
            initial_state:Optional[Tuple[torch.Tensor,torch.Tensor]] = None):       
        """
        Parameters
        ----------
            inputs : (Batch,seq_len, input_dim) or (seq_len,batch,input_dim)
            initial_state : Tuple((Bacth x hidden_size) ,(batch,hidden_size))

        Returns 
        -------
            output : (Batch,seq_len,hidden_size)
            (h_n,c_n) : Tuple( (Batch,num_layer,hidden_size), (Batch,num_layer,hidden_size) )

        """ 
        output_accumulator,(h_n,c_n),batch_lengths,save_state= self.LSTM(inputs,initial_state)
        
        if self.bidirectional:
            
            bw_output_accumul,(bw_h_n,bw_c_n),_,bw_save_state=self.bw_LSTM(inputs,initial_state)
            
            tmp_h_n = h_n.new_zeros(h_n.size(0),h_n.size(1)*2,self.hidden_size)
            tmp_c_n = h_n.new_zeros(h_n.size(0),h_n.size(1)*2,self.cell_size)

            tmp_h_n[:,0::2] = h_n
            tmp_h_n[:,1::2] = bw_h_n
            tmp_c_n[:,0::2] = c_n
            tmp_c_n[:,1::2] = bw_c_n
            h_n = tmp_h_n
            c_n = tmp_c_n
            output_accumulator = torch.cat((output_accumulator,bw_output_accumul),dim=-1)
            save_state = torch.cat((save_state,bw_save_state),dim=-1)
        if self.batch_first != False:
            output = output_accumulator.permute(1,0,2)
            h_n = h_n.permute(1,0,2)
            c_n = c_n.permute(1,0,2)


        final_state = (h_n,c_n)
        output= pack_padded_sequence(output_accumulator, batch_lengths, batch_first=self.batch_first)
        return output,final_state,save_state

class LSTM_block(nn.Module):
    def __init__(self,
                input_size:int,
                cell_size:int ,
                hidden_size:int = None,
                num_layer : int = 1,
                recurrent_dropout_probability :float =0.0,
                inprojection_bias:bool = True,
                skip_connection:bool = True,
                go_forward :bool = True,
                batch_first:bool = True 
                ) -> None:

        super(LSTM_block,self).__init__()
        self.input_size = input_size
        self.cell_size = cell_size
        if hidden_size :
            self.hidden_size = hidden_size
        else:
            self.hidden_size = cell_size

        self.num_layer = num_layer
        self.inprojection_bias = inprojection_bias
        self.skip_connection = skip_connection
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.batch_first = batch_first
        self.go_forward = go_forward
        self.LSTM_layers = nn.ModuleList([])

        
        for i in range(self.num_layer):
            if i!=0:
                input_size = hidden_size
            self.LSTM_layers.append(cellLSTM(input_size,self.cell_size,self.hidden_size,inprojection_bias))
            
            
    def forward(self,
            inputs:PackedSequence,
            initial_state:Optional[Tuple[torch.Tensor,torch.Tensor]] = None):
        """
        Parameters
        ----------
            inputs : (Batch,seq_len, input_dim) # it should be batch-first Tensor.
            initial_state : Tuple((Bacth ,layer, hidden_size) ,(batch,layer ,hidden_size))

        Returns 
        -------
            output : (Batch,seq_len,hidden_size)
            (h_n,c_n) : Tuple( (Batch,num_layer,hidden_size), (Batch,num_layer,hidden_size) )

        """
        if not isinstance(inputs,PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s'.format(type(inputs)))
        sequence_tensor,batch_lengths  = pad_packed_sequence(inputs,batch_first = True)
        batch_size = sequence_tensor.size()[0] 
        max_timestep = sequence_tensor.size()[1]
        
        output_accumulator = sequence_tensor.new_zeros(batch_size,max_timestep,self.hidden_size)
        
        h_n_accumulator = sequence_tensor.new_zeros(batch_size,self.num_layer,self.hidden_size)
        c_n_accumulator = sequence_tensor.new_zeros(batch_size,self.num_layer,self.hidden_size)

        current_length_index = batch_size - 1 if self.go_forward else 0
        
        if initial_state is None: # get_initial_state ! 
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size,self.num_layer, self.cell_size)
            full_batch_previous_state = sequence_tensor.data.new_zeros(batch_size,self.num_layer,self.hidden_size)

        else:
            full_batch_previous_state = initial_state[0]
            full_batch_previous_memory = initial_state[1]
        
        #save_memory =  sequence_tensor.new_zeros(batch_size,max_timestep,self.num_layer,self.hidden_size)
        save_state = sequence_tensor.new_zeros(batch_size,max_timestep,self.num_layer+1,self.hidden_size) #include input data size 


        if self.recurrent_dropout_probability > 0.0:
            # make dropout dropout_mask_list => row -> layer ,column direction 0 | 1 in
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_state[:,0,:],self.num_layer)
        else:
            dropout_mask =  None

        for timestep in range(max_timestep):

            index = timestep if self.go_forward else max_timestep - timestep - 1
            
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -=1
            else:
                while current_length_index < (batch_size -1 ) and batch_lengths[current_length_index +1] > index:
                    current_length_index +=1
                # 
            timestep_inputs = sequence_tensor[:current_length_index+1,index]
            pdb.set_trace()
            save_state[:current_length_index+1,index,0] = timestep_inputs
            
            for j in range(self.num_layer):
                previous_state = full_batch_previous_state[:current_length_index+1,j]
                previous_memory = full_batch_previous_memory[:current_length_index+1,j]

                if dropout_mask is not None and self.training:
                    timestep_inputs = timestep_inputs*dropout_mask[j][1][0:current_length_index+1] # layer wise dropout 


                h_n,c_n= self.LSTM_layers[j](timestep_inputs,(previous_state,previous_memory))

                if self.skip_connection and self.num_layer !=0:
                    h_n = h_n + timestep_inputs # time_step dropout 
                    timestep_inputs = h_n
                
                if dropout_mask is not None and self.training:
                    h_n = h_n*dropout_mask[j][0][0:current_length_index+1]

                full_batch_previous_state = full_batch_previous_state.data.clone()
                full_batch_previous_memory = full_batch_previous_memory.data.clone()

                full_batch_previous_memory[0:current_length_index+1,j] = c_n
                full_batch_previous_state[0:current_length_index+1,j] = h_n
            
            save_state[:,index,1:] = full_batch_previous_state.data.clone()
            
                
            output_accumulator[:current_length_index+1,index] = h_n.clone()
        
        #output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)
        #final_state = (full_batch_previous_state,full_batch_previous_memory)

        return output_accumulator,(full_batch_previous_state,full_batch_previous_memory),batch_lengths,save_state


class cellLSTM(nn.Module):
    def __init__(self,
        input_size:int,
        cell_size:int ,
        hidden_size:int,
        #projection:int = None ,
        inprojection_bias:bool = True
        ) -> None:
        super(cellLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.inprojection_bias = inprojection_bias
        self.cell_size = cell_size

        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=inprojection_bias)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=inprojection_bias)
        if self.hidden_size != self.cell_size :
            self.projection_layer = nn.Linear(cell_size,hidden_size)
        else:
            self.projection_layers =  None
        self.reset_parameters()
    def reset_parameters(self):
        #block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        #block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self,
                inputs:torch.Tensor,
                initial_state:Optional[Tuple[torch.Tensor,torch.Tensor]] = None):
        """
        Parameters
        ----------
            inputs : (Batch , input_dim)
            initial_state : Tuple((Bacth x hidden_size) ,(batch,hidden_size))

        Returns 
        -------
            (ht,memory) : Tuple((Batch,hidden_size),(Batch,hidden_size))
        """
        if initial_state is None: # get_initial_state ! 
            previous_memory = sequence_tensor.new_zeros(batch_size, self.cell_size)
            previous_state = sequence_tensor.data.new_zeros(batch_size, self.hidden_size)
        else:
            previous_state = initial_state[0]
            previous_memory = initial_state[1]
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(inputs)
        projected_state = self.state_linearity(previous_state)
        
        input_gate = torch.sigmoid(projected_input[:, 0 * self.cell_size:1 * self.cell_size] +
                                       projected_state[:, 0 * self.cell_size:1 * self.cell_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.cell_size:2 * self.cell_size] +
                                        projected_state[:, 1 * self.cell_size:2 * self.cell_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.cell_size:3 * self.cell_size] +
                                     projected_state[:, 2 * self.cell_size:3 * self.cell_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.cell_size:4 * self.cell_size] +
                                        projected_state[:, 3 * self.cell_size:4 * self.cell_size])
        memory = input_gate * memory_init + forget_gate * previous_memory

        h_t = output_gate * torch.tanh(memory)
        if self.projection_layer:
            h_t = self.projection_layer(h_t)

        #if self.projection is not None:
        #    h_t = self.projection_layer(h_t)
        return (h_t,memory)


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor,num_layer :int =1 ):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Tensor, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    dropout_mask_list = []

    for i in range(num_layer):
        tmp_dropmasks = []
        for j in range(2):
            binary_mask = tensor_for_masking.new_tensor(torch.rand(tensor_for_masking.size()) > dropout_probability)
            # Scale mask by 1/keep_prob to preserve output statistics.
            dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
            tmp_dropmasks.append(dropout_mask)
        dropout_mask_list.append(tmp_dropmasks)
    return dropout_mask_list