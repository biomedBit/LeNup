require 'nn'
require 'optim'

if cuda then
    require 'cunn'
    require 'cutorch'
end
if cuda_nn then
    require 'cudnn'
end

require 'accuracy'
require 'batcher'


ConvNet = {}

function ConvNet:__init()
    obj = {}
    setmetatable(obj, self)
    self.__index = self
    return obj
end



------------------------------------------------------------------


------------------------------------------------------------------
local function Tower(layers)
  local tower = nn.Sequential()
  for i=1,#layers do
    tower:add(layers[i])
  end
  return tower
end

local function FilterConcat(towers)
  local concat = nn.DepthConcat(2)
  for i=1,#towers do
    concat:add(towers[i])
  end
  return concat
end
----------------------------------------------------------------

----------------------------------------------------------------


----------------------------------------------------------------
----gated
----------------------------------------------------------------
	local function gatedCNNLayer1_1()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(4, 128, 1, 1))
		gate:add(nn.SpatialBatchNormalization(128))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(4, 128, 1, 1))
		con:add(nn.SpatialBatchNormalization(128))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end


	local function gatedCNNLayer1_2()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(4, 128, 7, 1, 1, 1, 3, 0))
		gate:add(nn.SpatialBatchNormalization(128))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(4, 128, 7, 1, 1, 1, 3, 0))
		con:add(nn.SpatialBatchNormalization(128))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end


	local function gatedCNNLayer1_3()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(4, 128, 3, 1, 1, 1, 1, 0))
		gate:add(nn.SpatialBatchNormalization(128))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(4, 128, 3, 1, 1, 1, 1, 0))
		con:add(nn.SpatialBatchNormalization(128))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end


	local function gatedCNNLayer3_1_1()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(384, 96, 1, 1))
		gate:add(nn.SpatialBatchNormalization(96))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(384, 96, 1, 1))
		con:add(nn.SpatialBatchNormalization(96))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end
	
	local function gatedCNNLayer3_1_2()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(384, 64, 1, 1))
		gate:add(nn.SpatialBatchNormalization(64))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(384, 64, 1, 1))
		con:add(nn.SpatialBatchNormalization(64))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end

	local function gatedCNNLayer3_2()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(64, 96, 7, 1, 1, 1, 3, 0))
		gate:add(nn.SpatialBatchNormalization(96))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(64, 96, 7, 1, 1, 1, 3, 0))
		con:add(nn.SpatialBatchNormalization(96))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end


	local function gatedCNNLayer3_3()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(64, 96, 7, 1, 1, 1, 3, 0))
		gate:add(nn.SpatialBatchNormalization(96))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(64, 96, 7, 1, 1, 1, 3, 0))
		con:add(nn.SpatialBatchNormalization(96))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end


	local function gatedCNNLayer3_4()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(96, 96, 7, 1, 1, 1, 3, 0))
		gate:add(nn.SpatialBatchNormalization(96))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(96, 96, 7, 1, 1, 1, 3, 0))
		con:add(nn.SpatialBatchNormalization(96))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end



	local function gatedCNNLayer4_1_1()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(384, 96, 1, 1))
		gate:add(nn.SpatialBatchNormalization(96))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(384, 96, 1, 1))
		con:add(nn.SpatialBatchNormalization(96))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end
	
	local function gatedCNNLayer4_1_2()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(384, 64, 1, 1))
		gate:add(nn.SpatialBatchNormalization(64))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(384, 64, 1, 1))
		con:add(nn.SpatialBatchNormalization(64))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end

	local function gatedCNNLayer4_2()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(64, 96, 11, 1, 1, 1, 5, 0))
		gate:add(nn.SpatialBatchNormalization(96))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(64, 96, 11, 1, 1, 1, 5, 0))
		con:add(nn.SpatialBatchNormalization(96))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end


	local function gatedCNNLayer4_3()
        local gate = nn.Sequential()
        gate:add(nn.SpatialConvolution(64, 96, 11, 1, 1, 1, 5, 0))
		gate:add(nn.SpatialBatchNormalization(96))
		gate:add(nn.ReLU())
        gate:add(nn.Sigmoid())
	
		local con = nn.Sequential()
        con:add(nn.SpatialConvolution(64, 96, 11, 1, 1, 1, 5, 0))
		con:add(nn.SpatialBatchNormalization(96))
		con:add(nn.ReLU())

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(con)
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end


----------------------------------------------------------------
----gated
----------------------------------------------------------------

-------------------------------------------------------------

local function Inception_A()
	local inception = FilterConcat(
	{
		Tower(
        {
			gatedCNNLayer1_1(),
        }
		),
		Tower(
        {
			gatedCNNLayer1_2(),
			
		}
		),
		Tower(
        {
			gatedCNNLayer1_3(),
        }
		),
    }
	) 
	return inception
end




local function Inception_C()
	local inception = FilterConcat(
	{
		Tower(
        {
			nn.SpatialAveragePooling(3, 1, 1, 1, 1, 0),
			gatedCNNLayer3_1_1(),
        }
		),
		Tower(
        {
			gatedCNNLayer3_1_1(),
        }
		),
		Tower(
        {
			gatedCNNLayer3_1_2(),
			gatedCNNLayer3_2(),
		}
		),
		Tower(
        {
			gatedCNNLayer3_1_2(),
			gatedCNNLayer3_3(),
        }
		),
    }
	) 
	return inception
end



local function Inception_D()
	local inception = FilterConcat(
	{
		Tower(
        {
			nn.SpatialAveragePooling(3, 1, 1, 1, 1, 0),
			gatedCNNLayer4_1_1(),
        }
		),
		Tower(
        {
			gatedCNNLayer4_1_1(),
        }
		),
		Tower(
        {
			gatedCNNLayer4_1_2(),
			gatedCNNLayer4_2(),
		}
		),
		Tower(
        {
			gatedCNNLayer4_1_2(),
			gatedCNNLayer4_3(),
        }
		),
    }
	) 
	return inception
end



--------------------------------------------------------------------


----------------------------------------------------------------
-- build
--
-- Build the network using the job parameters and data
-- attributes.
----------------------------------------------------------------
function ConvNet:build(job, init_depth, init_len, num_targets)
    -- parse network structure parameters
    self:setStructureParams(job)

    -- initialize model sequential
    self.model = nn.Sequential()

    -- store useful values
    self.num_targets = num_targets
    local depth = init_depth
    local seq_len = init_len
	
	
	

    self.model:add(Inception_A())
    self.model:add(nn.SpatialMaxPooling(2,1))
    self.model:add(nn.Dropout(0.5))
	
	self.model:add(Inception_C())
	self.model:add(nn.SpatialMaxPooling(4,1))
	self.model:add(nn.Dropout(0.3))
	
	self.model:add(Inception_C())
	self.model:add(Inception_D())
    self.model:add(nn.SpatialMaxPooling(3,1))
    self.model:add(nn.Dropout(0.3))
    
	self.model:add(Inception_D())
	self.model:add(nn.Dropout(0.3))
    self.model:add(nn.SpatialAveragePooling(6, 1))


    hidden_in = 384
    self.model:add(nn.Reshape(hidden_in))
    self.model:add(nn.Linear(hidden_in, 384))
    self.model:add(nn.BatchNormalization(384))     
    self.model:add(nn.ReLU())     
    self.model:add(nn.Dropout(0.3))

    final_linear = nn.Linear(384, self.num_targets)
    self.model:add(final_linear)
	self.model:add(nn.Dropout(0.3))
    self.model:add(nn.Sigmoid())
    self.criterion = nn.BCECriterion()


    self.criterion.sizeAverage = false

    -- cuda
    if cuda then
        print("Running on GPU.")
        self.model:cuda()
        self.criterion:cuda()
    end
    if cuda_nn then
        self.model = cudnn.convert(self.model, cudnn)
    end
	
    -- retrieve parameters and gradients
    self.parameters, self.gradParameters = self.model:getParameters()

    -- print model summary
    print(self.model)
    return true
end


----------------------------------------------------------------
-- cuda
--
-- Move the model to the GPU. Untested.
----------------------------------------------------------------
function ConvNet:cuda()
    self.optim_state.m = self.optim_state.m:cuda()
    self.optim_state.tmp = self.optim_state.tmp:cuda()
    self.criterion:cuda()
    self.model:cuda()

    if cuda_nn then
        cudnn.convert(self.model, nn)
    end

    self.parameters, self.gradParameters = self.model:getParameters()
	
    cuda = true
end



----------------------------------------------------------------
-- drop_rate
--
-- Decrease the optimization learning_rate by a multiplier
----------------------------------------------------------------
function ConvNet:drop_rate()
    self.learning_rate = self.learning_rate * 0.97
    self.optim_state.learningRate = self.learning_rate
end


----------------------------------------------------------------
-- sanitize
--
-- Clear the intermediate states in the model before
-- saving to disk.
----------------------------------------------------------------
function ConvNet:sanitize()
    local module_list = self.model:listModules()
    for _,val in ipairs(module_list) do
        for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if name == 'homeGradBuffers' then val[name] = nil end
            if name == 'input_gpu' then val['input_gpu'] = {} end
            if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
            if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
            if name == 'output' or name == 'gradInput' then
                --val[name] = field.new()
            end
            -- batch normalization
            if name == 'buffer' or name == 'normalized' or name == 'centered' then
                --val[name] = field.new()
            end
            -- max pooling
            if name == 'indices' then
                --val[name] = field.new()
            end
        end
    end
end

----------------------------------------------------------------
-- setStructureParams
--
----------------------------------------------------------------
function ConvNet:setStructureParams(job)
    ---------------------------------------------
    -- training
    ---------------------------------------------
    -- number of examples per weight update
    self.batch_size = job.batch_size or 128

    -- optimization algorithm
    self.optimization = job.optimization or "rmsprop"

    -- base learning rate
    self.learning_rate = job.learning_rate or 0.002

    -- gradient update momentum
    self.momentum = job.momentum or 0.98

    -- adam momentums
    self.beta1 = job.beta1 or 0.9
    self.beta2 = job.beta2 or 0.999

    -- batch normaliztion
    if job.batch_normalize == nil then
        self.batch_normalize = true
    else
        self.batch_normalize = job.batch_normalize
    end

    self.weight_norm = job.weight_norm or 10

end


---------------------------------------------------------------
-- train_epoch
--
-- Train the model for one epoch through the data specified by
-- batcher.
----------------------------------------------------------------
function ConvNet:train_epoch(batcher)
    local total_loss = 0

    -- collect garbage occasionaly
    collectgarbage()
    local cgi = 0

    -- get first batch
    local inputs, targets = batcher:next()

    -- while batches remain
    while inputs ~= nil do
        -- cuda
        if cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= self.parameters then
                self.parameters:copy(x)
            end

            -- reset gradients
            self.gradParameters:zero()

            -- evaluate function for mini batch
            local outputs = self.model:forward(inputs)
            local f = self.criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = self.criterion:backward(outputs, targets)
            self.model:backward(inputs, df_do)
          
            return f, self.gradParameters
        end

        self.optim_state = self.optim_state or {
            learningRate = self.learning_rate,
            alpha = self.momentum
        }
        optim.rmsprop(feval, self.parameters, self.optim_state)

        -- cap weight paramaters
        self.model:maxParamNorm(self.weight_norm)

        -- accumulate loss
        total_loss = total_loss + self.criterion.output

        -- next batch
        inputs, targets = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    -- reset batcher
    batcher:reset()

    -- mean loss over examples
    avg_loss = total_loss / batcher.num_seqs
    return avg_loss
end


----------------------------------------------------------------
-- test
--
-- Predict targets for X and compare to Y.
----------------------------------------------------------------
function ConvNet:test(Xf, Yf, epoch)
    -- track the loss across batches
    local loss = 0
    -- collect garbage occasionaly
    local cgi = 0

    -- create a batcher to help
    local batch_size = batch_size or self.batch_size
    local batcher = Batcher:__init(Xf, Yf, batch_size)

    -- track predictions across batches
    local preds = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local pi = 1

    -- get first batch
    local inputs, targets = batcher:next()
	
    myTP = 0
    myTN = 0
    myFP = 0
    myFN = 0
	
    -- while batches remain
    while inputs ~= nil do
        -- cuda
        if cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end
		
		targets2 = targets

        -- predict
        local preds_batch = self.model:forward(inputs)

        loss = loss + self.criterion:forward(preds_batch, targets)

        -- copy into larger Tensor
        for i = 1,(#preds_batch)[1] do
            preds[{pi,{}}] = preds_batch[{i,{}}]:float()
            pi = pi + 1
        end
        
        tmp,myindex = torch.sort(preds_batch[{} ])
        for i = 1,preds_batch:size(1) do	
            if preds_batch[i][1] > 0.5 then
                myindex[i][1] = 1
            else
                myindex[i][1] = 0
            end
	    end

        for i = 1,preds_batch:size(1) do
            if myindex[i][1] == targets2[i][1] and myindex[i][1] ==0 then
                myTP = myTP + 1
            end
        end
    
        for i = 1,preds_batch:size(1) do
            if myindex[i][1] == targets2[i][1] and myindex[i][1] ==1 then
                myTN = myTN + 1
            end
        end
    
       for i = 1,preds_batch:size(1) do
            if myindex[i][1] ~= targets2[i][1] and myindex[i][1] ==0 then
                myFP = myFP + 1
            end
        end
    
        for i = 1,preds_batch:size(1) do
            if myindex[i][1] ~= targets2[i][1] and myindex[i][1] ==1 then
                myFN = myFN + 1
            end
        end

		-- next batch
        inputs, targets = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    mycorrectRate = (myTP+ myTN) / (myTP + myTN + myFP + myFN)
    mysen = myTP / (myTP + myFN)
    myspc = myTN / (myTN + myFP)
    mcc = (myTP * myTN) / (((myTP + myFN) * (myTP + myFP) * (myTN + myFP) * (myTN + myFN))^(1 / 2))
	io.write(string.format("acc = %7.5f, sen = %7.5f, spc = %7.5f, mcc = %7.5f, ", mycorrectRate, mysen, myspc, mcc))


	-- mean loss over examples
    local avg_loss = loss / batcher.num_seqs

    -- save pred means and stds
    self.pred_means = preds:mean(1):squeeze()
    self.pred_stds = preds:std(1):squeeze()

    local Ydim = batcher.num_targets
  
        -- compute AUC
    local AUCs = torch.Tensor(Ydim)
    local roc_points = {}
    for yi = 1,Ydim do
        -- read Yi from file
        local Yi = Yf:partial({1,batcher.num_seqs},{yi,yi}):squeeze()

        if(Yi:sum() == 0) then
            roc_points[yi] = nil
            AUCs[yi] = 1.0

        else
                -- compute ROC points
            roc_points[yi] = ROC.points(preds[{{},yi}], Yi)

                -- compute AUCs
            AUCs[yi] = ROC.area(roc_points[yi])
        end

        collectgarbage()
    end
		

    return avg_loss, AUCs, roc_points, mycorrectRate

end

----------------------------------------------------------------
-- get_final
--
-- Return the module representing the final layer.
----------------------------------------------------------------
function ConvNet:get_final()
    local layers = #self.model
    return self.model.modules[layers-1]
end


function ConvNet:load(cnn)
    for k, v in pairs(cnn) do
        self[k] = v
    end
end


----------------------------------------------------------------
-- predict
--
-- Predict targets for a new set of sequences.
----------------------------------------------------------------
function ConvNet:predict(Xf, batch_size)
    local bs = batch_size or self.batch_size
    local batcher

    batcher = Batcher:__init(Xf, nil, bs)

    -- find final model layer
    local final_i = #self.model.modules - 1
    if self.target_type == "continuous" then
        final_i = final_i + 1
    end

    -- track predictions across batches
    local preds = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local scores = torch.FloatTensor(batcher.num_seqs, self.num_targets)
    local pi = 1

    -- collect garbage occasionaly
    local cgi = 0

    -- get first batch
    local Xb = batcher:next()

    -- while batches remain
    while Xb ~= nil do
        -- cuda
        if cuda then
            Xb = Xb:cuda()
        end

        -- predict
        local preds_batch = self.model:forward(Xb)
        local scores_batch = self.model.modules[final_i].output
        -- copy into larger Tensor
        for i = 1,(#preds_batch)[1] do
            preds[{pi,{}}] = preds_batch[{i,{}}]:float()
            scores[{pi,{}}] = scores_batch[{i,{}}]:float()
            pi = pi + 1
			
        end

        -- next batch
        Xb = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    return preds, scores, xx, xxx
end
